"""
convert_to_mcq.py — Sinh MCQ options cho từng query trong test_perturbed_subgraphs/

THIẾT KẾ:
  Mỗi query có 4 options CỐ ĐỊNH, giống nhau ở TẤT CẢ variants:
    - GT:           label(s) của clean answer (đáp án gốc)
    - Swap entity:  label của swapping entity (đáp án của graph bị hoán đổi)
    - No context:   "Insufficient information to answer" (đáp án khi graph bị cắt - broken)
    - Distractor:   1 entity ngẫu nhiên không trùng 3 đáp án trên

  Đáp án đúng (mcq_correct_letter) khác nhau tuỳ variant:
    - clean, topological, type_matching → GT
    - broken                            → No context
    - swapping                          → Swap entity

  Tất cả 4 options được shuffle một lần và dùng cho mọi variant → đảm bảo cùng ABCD trên mọi mode.
"""

import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

RANDOM_SEED = 42  # seed để đảm bảo reproducible shuffle


def load_wikidata_labels():
    cache_file = "data/wikidata_labels.json"
    with open(cache_file, "r", encoding="utf-8") as f:
        return json.load(f)


GLOBAL_LABELS = load_wikidata_labels()

NO_CONTEXT_TEXT = "Insufficient information to answer"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_entity_label(uri: str, global_labels: dict) -> str:
    """Resolve một Wikidata URI thành label (human-readable string)."""
    if not isinstance(uri, str):
        return str(uri)
    eid = uri.split("/")[-1]
    val = global_labels.get(eid, {})
    if isinstance(val, dict):
        return val.get("label", eid) or eid
    return str(val) or eid


def get_gt_label(answers: list, global_labels: dict) -> str:
    """
    Chuyển danh sách URIs của ground truth thành chuỗi label đơn.
    Nếu nhiều answers (e.g. 9 sister cities), nối bằng ', '.
    """
    if not answers:
        return "Unknown"
    labels = []
    for uri in answers:
        labels.append(get_entity_label(uri, global_labels))
    return ", ".join(labels)


def get_random_distractor(global_labels: dict, exclude_texts: set) -> str:
    """Lấy 1 entity label ngẫu nhiên không trùng với exclude_texts."""
    all_keys = list(global_labels.keys())
    for _ in range(200):
        k = random.choice(all_keys)
        val = global_labels.get(k, {})
        label = val.get("label", k) if isinstance(val, dict) else str(k)
        if label and label not in exclude_texts:
            return label
    return "Random Distractor"


def build_options_for_query(
    gt_text: str,
    swap_text: str,
    rng: random.Random,
) -> tuple[dict, str, str, str]:
    """
    Xây dựng 4 options cố định cho một query.

    Returns:
        options_dict: {"A": ..., "B": ..., "C": ..., "D": ...}
        letter_gt: chữ cái của GT
        letter_swap: chữ cái của swap entity
        letter_no_context: chữ cái của no context
    """
    exclude = {gt_text, swap_text, NO_CONTEXT_TEXT}
    distractor = get_random_distractor(GLOBAL_LABELS, exclude)

    # Đảm bảo 4 slots: GT, swap, no_context, distractor
    slots = [
        ("gt", gt_text),
        ("swap", swap_text),
        ("no_context", NO_CONTEXT_TEXT),
        ("distractor", distractor),
    ]

    # Shuffle thứ tự nhưng dùng rng riêng để reproducible theo query
    rng.shuffle(slots)

    letters = ["A", "B", "C", "D"]
    options_dict = {}
    letter_gt = letter_swap = letter_no_context = None

    for i, (role, text) in enumerate(slots):
        letter = letters[i]
        options_dict[letter] = text
        if role == "gt":
            letter_gt = letter
        elif role == "swap":
            letter_swap = letter
        elif role == "no_context":
            letter_no_context = letter

    return options_dict, letter_gt, letter_swap, letter_no_context


def build_options_text(options_dict: dict) -> str:
    """Sinh chuỗi options text để nối vào câu hỏi."""
    lines = ["\n\nOptions:"]
    for letter in ["A", "B", "C", "D"]:
        lines.append(f"{letter}. {options_dict[letter]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------

VARIANT_TO_ROLE = {
    "clean": "gt",
    "topological": "gt",
    "type_matching": "gt",
    "broken": "no_context",
    "swapping": "swap",
}


def process_single_file(file_path: Path, verbose: bool = True):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return

    # Chỉ xử lý file multi-variant (phải có cả clean lẫn broken)
    if "clean" not in data or "broken" not in data:
        if verbose:
            print(f"[SKIP] {file_path.name}: không phải multi-variant file.")
        return

    # -------------------------------------------------------------------
    # Bước 1: Thu thập GT và swap labels từ đúng từng variant
    # -------------------------------------------------------------------
    clean_answers = data["clean"].get("answers", [])
    swap_answers = data.get("swapping", {}).get("answers", [])

    gt_text = get_gt_label(clean_answers, GLOBAL_LABELS)

    # Nếu swapping có đáp án mới (swap entity), dùng nó; ngược lại fallback ra GT
    if swap_answers:
        swap_text = get_gt_label(swap_answers, GLOBAL_LABELS)
    else:
        # Swapping không tạo ra entity mới → dùng distractor giả
        swap_text = get_random_distractor(GLOBAL_LABELS, {gt_text, NO_CONTEXT_TEXT})

    # Nếu swap_text == gt_text (edge case), tạo distractor khác
    if swap_text == gt_text:
        swap_text = get_random_distractor(GLOBAL_LABELS, {gt_text, NO_CONTEXT_TEXT})

    # -------------------------------------------------------------------
    # Bước 2: Xây dựng 4 options CỐ ĐỊNH cho query này
    # -------------------------------------------------------------------
    # Dùng question_id làm seed để đảm bảo shuffle reproducible
    qid = data["clean"].get("question_id", file_path.stem)
    rng = random.Random(int(qid) if str(qid).isdigit() else hash(qid))

    options_dict, letter_gt, letter_swap, letter_no_context = build_options_for_query(
        gt_text, swap_text, rng
    )
    options_text = build_options_text(options_dict)

    if verbose:
        print(f"\n=== Question {qid} ===")
        print(f"  GT          : {repr(gt_text)} → {letter_gt}")
        print(f"  Swap entity : {repr(swap_text)} → {letter_swap}")
        print(f"  No context  : {repr(NO_CONTEXT_TEXT)} → {letter_no_context}")
        print(f"  Options     : {options_dict}")

    # -------------------------------------------------------------------
    # Bước 3: Gán options và correct_letter vào từng variant
    # -------------------------------------------------------------------
    for variant, content in data.items():
        if not isinstance(content, dict):
            continue

        role = VARIANT_TO_ROLE.get(variant, "gt")  # default = GT nếu variant lạ

        if role == "gt":
            correct_letter = letter_gt
        elif role == "no_context":
            correct_letter = letter_no_context
        elif role == "swap":
            correct_letter = letter_swap
        else:
            correct_letter = letter_gt

        content["mcq_options_text"] = options_text
        content["mcq_options_dict"] = options_dict
        content["mcq_correct_letter"] = correct_letter

        if verbose:
            print(f"  [{variant}] correct_letter = {correct_letter}")

    # -------------------------------------------------------------------
    # Bước 4: Ghi lại file
    # -------------------------------------------------------------------
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def process_dir(dir_name: str, verbose: bool = True):
    path = Path(f"data/{dir_name}")
    if not path.exists():
        print(f"[WARN] Không tìm thấy thư mục: {path}")
        return

    files = sorted(path.glob("*.json"))
    print(f"\n[INFO] Xử lý {len(files)} file trong '{dir_name}'...")
    for fp in files:
        process_single_file(fp, verbose=verbose)
    print(f"[DONE] Hoàn thành '{dir_name}'.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Verbose = False khi xử lý toàn bộ dataset (tránh spam)
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    process_dir("test_perturbed_subgraphs", verbose=verbose)
    process_dir("perturbed_subgraphs", verbose=verbose)
