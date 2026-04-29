#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def is_none_like(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"none", "null", ""}
    return False


def is_variant_failed(variant_payload: dict) -> bool:
    predicted_letter = variant_payload.get("predicted_letter")
    answer = variant_payload.get("answer")
    return is_none_like(predicted_letter) or is_none_like(answer) or str(answer).strip().upper() == "ERROR"


def question_has_failure(question_payload: dict) -> bool:
    variants = question_payload.get("variants", {}) or {}
    if not isinstance(variants, dict) or not variants:
        return True

    for _, variant_payload in variants.items():
        if not isinstance(variant_payload, dict):
            return True
        if is_variant_failed(variant_payload):
            return True
    return False


def compute_mode_stats(records: list[dict]) -> dict:
    stats = defaultdict(
        lambda: {
            "total": 0,
            "valid": 0,
            "skipped_none_or_error": 0,
            "correct": 0,
            "accuracy": 0.0,
        }
    )

    for rec in records:
        variants = rec.get("variants", {}) or {}
        if not isinstance(variants, dict):
            continue

        for mode, payload in variants.items():
            if not isinstance(payload, dict):
                continue

            stats[mode]["total"] += 1

            if is_variant_failed(payload):
                stats[mode]["skipped_none_or_error"] += 1
                continue

            stats[mode]["valid"] += 1
            pred = str(payload.get("predicted_letter", "")).strip()
            gold = str(payload.get("mcq_correct_letter", "")).strip()
            if pred == gold and pred != "":
                stats[mode]["correct"] += 1

    for _, s in stats.items():
        if s["valid"] > 0:
            s["accuracy"] = s["correct"] / s["valid"]
        else:
            s["accuracy"] = 0.0

    return dict(stats)


CURRENT_BENCHMARK_YEAR = 2026
ISO_DATE_RE = re.compile(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?:[T\s]\d{2}:\d{2}:\d{2}(?:Z)?)?(?!\d)")
DMY_SLASH_RE = re.compile(r"(?<!\d)(\d{1,2})/(\d{1,2})/(\d{4})(?!\d)")
DMY_DASH_RE = re.compile(r"(?<!\d)(\d{1,2})-(\d{1,2})-(\d{4})(?!\d)")
YEAR_RE = re.compile(r"(?<!\d)(1\d{3}|20[0-1]\d|202[0-5])(?!\d)")


def has_past_timestamp(text: str) -> bool:
    if not text:
        return False

    years = []
    spans = []
    for pat, yg in ((ISO_DATE_RE, 1), (DMY_SLASH_RE, 3), (DMY_DASH_RE, 3)):
        for m in pat.finditer(text):
            y = int(m.group(yg))
            if y < CURRENT_BENCHMARK_YEAR:
                years.append(y)
                spans.append(m.span())

    def in_spans(i: int) -> bool:
        return any(a <= i < b for a, b in spans)

    for m in YEAR_RE.finditer(text):
        if in_spans(m.start()):
            continue
        y = int(m.group(1))
        if y < CURRENT_BENCHMARK_YEAR:
            years.append(y)

    return len(years) > 0


def build_question_group_map(lcquad_file: Path) -> dict[str, str]:
    if not lcquad_file.exists():
        return {}

    raw = json.loads(lcquad_file.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return {}

    result: dict[str, str] = {}
    for row in raw:
        if not isinstance(row, dict):
            continue
        uid = str(row.get("uid", "")).strip()
        if not uid:
            continue
        text = (row.get("question") or row.get("NNQT_question") or "").strip()
        result[uid] = "timestamp" if has_past_timestamp(text) else "non_timestamp"
    return result


def print_mode_stats(title: str, records: list[dict]) -> None:
    print(f"\n=== {title} ===")
    print(f"Questions in scope: {len(records)}")
    mode_stats = compute_mode_stats(records)
    if not mode_stats:
        print("No mode statistics available.")
        return

    for mode in sorted(mode_stats.keys()):
        s = mode_stats[mode]
        print(
            f"{mode}: valid {s['valid']}/{s['total']} | "
            f"accuracy {s['accuracy']:.4f} ({s['correct']}/{s['valid']}) | "
            f"skipped_none_or_error {s['skipped_none_or_error']}"
        )


def make_backup(input_path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = input_path.with_suffix(input_path.suffix + f".bak_{ts}")
    backup_path.write_text(input_path.read_text(encoding="utf-8"), encoding="utf-8")
    return backup_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Clean benchmark prediction JSON in-place by removing question records with None/error variants, "
            "then report temporary progress and per-mode accuracy."
        )
    )
    parser.add_argument(
        "results_file",
        help="Path to benchmark results JSON (framework-agnostic).",
    )
    parser.add_argument(
        "--lcquad-file",
        default="data/lcquad_test.json",
        help="Path to lcquad test JSON used for time/non-time grouping.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create timestamped backup before overwriting input file.",
    )
    args = parser.parse_args()

    results_path = Path(args.results_file)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    output_path = results_path

    raw = json.loads(results_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Expected benchmark results JSON to be a list")

    question_group_map = build_question_group_map(Path(args.lcquad_file))

    total_questions = len(raw)
    removed_qids = []
    kept = []

    for rec in raw:
        qid = str(rec.get("question_id", "UNKNOWN"))
        if question_has_failure(rec):
            removed_qids.append(qid)
        else:
            kept.append(rec)

    if not args.no_backup:
        backup_path = make_backup(results_path)
        print(f"Backup created: {backup_path}")

    output_path.write_text(json.dumps(kept, ensure_ascii=False, indent=4), encoding="utf-8")

    success_questions = len(kept)
    removed_questions = len(removed_qids)
    question_success_rate = (success_questions / total_questions * 100.0) if total_questions else 0.0

    print("\n=== Question-level Progress ===")
    print(
        "Successful questions (no None/error variants): "
        f"{success_questions}/{total_questions} ({question_success_rate:.2f}%)"
    )
    print(f"Removed questions for rerun: {removed_questions}")

    if removed_qids:
        preview = ", ".join(removed_qids[:20])
        print(f"Removed question IDs (first 20): {preview}")

    # 1) Overall metrics as before
    print_mode_stats("Per-mode Accuracy (ALL, ignoring None/error)", raw)

    # 2) Split metrics: timestamp vs non_timestamp
    if question_group_map:
        raw_timestamp = []
        raw_non_timestamp = []
        raw_unknown = []

        for rec in raw:
            qid = str(rec.get("question_id", ""))
            group = question_group_map.get(qid)
            if group == "timestamp":
                raw_timestamp.append(rec)
            elif group == "non_timestamp":
                raw_non_timestamp.append(rec)
            else:
                raw_unknown.append(rec)

        print_mode_stats("Per-mode Accuracy (TIMESTAMP only)", raw_timestamp)
        print_mode_stats("Per-mode Accuracy (NON_TIMESTAMP only)", raw_non_timestamp)
        if raw_unknown:
            print(f"\nUnmapped question IDs (not found in lcquad file): {len(raw_unknown)}")
    else:
        print("\nSkip split metrics: lcquad file not found or invalid.")

    print(f"\nCleaned results written to: {output_path}")


if __name__ == "__main__":
    main()