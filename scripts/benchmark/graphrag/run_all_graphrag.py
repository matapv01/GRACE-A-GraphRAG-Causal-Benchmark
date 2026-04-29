import asyncio
import json
import logging
import os
import shutil
import sys
import traceback
from pathlib import Path

import nest_asyncio
from dotenv import load_dotenv
from rich.console import Console

nest_asyncio.apply()
load_dotenv()

# Prevent GraphRAG logging from spamming too much during mass execution
logging.getLogger("graphrag").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "benchmark" / "graphrag"))

from benchmark_graphrag import GLOBAL_LABELS, RESULTS_FILE, WORKSPACE_ROOT, run_graphrag_scenario
from graphrag_benchmark.use_cases.evaluation_module import EvaluationModule

console = Console()


def iter_variant_items(data: dict):
    """Yield only valid variant payloads that contain triples."""
    for variant, payload in data.items():
        if isinstance(payload, dict) and "triples" in payload:
            yield variant, payload


async def evaluate_single_file(test_file: Path, evaluator: EvaluationModule):
    q_id = test_file.stem
    console.print(f"\n[bold blue]>>> Bắt đầu ID: {q_id} <<<[/bold blue]")

    with open(test_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    question_text = "What is the relation between these entities?"
    try:
        with open(DATA_DIR / "lcquad_test.json", "r", encoding="utf-8") as f:
            all_q = json.load(f)
            for q in all_q:
                if str(q.get("uid")) == str(q_id):
                    question_text = q.get("question")
                    break
    except FileNotFoundError:
        pass

    results = {"question_id": q_id, "question": question_text, "variants": {}}

    for variant, payload in iter_variant_items(data):
        triples = payload.get("triples", [])
        ground_truth = payload.get("answers", [])

        options_text = payload.get("mcq_options_text", "")
        correct_letter = payload.get("mcq_correct_letter", "None")
        mcq_question = question_text + options_text

        work_dir = str(WORKSPACE_ROOT / q_id / variant)

        if os.path.exists(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        os.makedirs(work_dir, exist_ok=True)

        try:
            rag_output = await run_graphrag_scenario(
                question=mcq_question,
                triples=triples,
                variant_name=variant,
                workspace_dir=work_dir,
            )
            ans = rag_output.get("answer", "")
            retrieved_context = rag_output.get("retrieved_context", "")

            score, predicted_letter = await evaluator.compute_llm_based_score(
                predicted_text=str(ans),
                ground_truths=[correct_letter],
                variant=variant,
                global_labels=GLOBAL_LABELS,
                question=mcq_question,
            )

            results["variants"][variant] = {
                "ground_truth": ground_truth,
                "mcq_correct_letter": correct_letter,
                "mcq_options": options_text,
                "retrieved_context": retrieved_context,
                "answer": str(ans),
                "score": score,
                "predicted_letter": predicted_letter,
            }
            console.print(f"[{variant.upper()}] Correct Letter: {correct_letter} | Model Output: {ans} | Score: {score}")
        except Exception as e:
            console.print(f"[red]Lỗi ở ID {q_id} Mode {variant}: {e}[/red]")
            traceback.print_exc()
            results["variants"][variant] = {
                "answer": "ERROR",
                "predicted_letter": "None",
                "score": 0.0,
            }

    return results


async def main():
    if not os.environ.get("OPENAI_API_KEY"):
        console.print("[red]Thiếu cấu hình OPENAI_API_KEY!. Chạy huỷ bỏ.[/red]")
        return

    files = list((DATA_DIR / "test_perturbed_subgraphs").glob("*.json"))
    console.print(f"[bold cyan]Tìm thấy {len(files)} subgraphs để chạy đánh giá.[/bold cyan]")

    evaluator = EvaluationModule()

    all_results = []
    output_file = RESULTS_FILE
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                all_results = json.load(f)
        except Exception:
            all_results = []

    completed_ids = {res["question_id"] for res in all_results}

    for idx, test_file in enumerate(files):
        q_id = test_file.stem
        if q_id in completed_ids:
            console.print(f"Skipping {q_id} (Đã có kết quả)")
            continue

        console.print(f"[bold yellow]Progess: {idx+1}/{len(files)}[/bold yellow]")

        res = await evaluate_single_file(test_file, evaluator)
        all_results.append(res)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)

    console.print(f"\n[bold green]Hoàn thành Benchmark toàn bộ dữ liệu! Kết quả lưu ở {output_file}[/bold green]")


if __name__ == "__main__":
    asyncio.run(main())
