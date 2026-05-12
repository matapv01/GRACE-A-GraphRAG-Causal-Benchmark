# File: scripts/benchmark/graphrag/run_all_graphrag.py

import asyncio
import json
import logging
import os
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

from benchmark_graphrag import evaluate_single_file, RESULTS_FILE
from graphrag_benchmark.use_cases.evaluation_module import EvaluationModule

console = Console()


async def main():
    if not os.environ.get("OPENAI_API_KEY"):
        console.print("[red]Thiếu cấu hình OPENAI_API_KEY!. Chạy huỷ bỏ.[/red]")
        return

    files = sorted((DATA_DIR / "test_perturbed_subgraphs").glob("*.json"))
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
            console.print(f"[yellow]Skipping {q_id} (Đã có kết quả)[/yellow]")
            continue

        console.print(f"[bold yellow]Progress: {idx+1}/{len(files)}[/bold yellow]")

        try:
            res = await evaluate_single_file(test_file, evaluator)
            all_results.append(res)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=4)
        except Exception as e:
            console.print(f"[red]Lỗi khi chạy ID {q_id}: {e}[/red]")
            traceback.print_exc()

    console.print(f"\n[bold green]Hoàn thành Benchmark toàn bộ dữ liệu! Kết quả lưu ở {output_file}[/bold green]")


if __name__ == "__main__":
    asyncio.run(main())
