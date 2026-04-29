import asyncio
import gc
import inspect
import json
import os
import random
import shutil
import subprocess
import traceback
from pathlib import Path
from urllib.parse import urlparse

import nest_asyncio
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
nest_asyncio.apply()

# Prevent GraphRAG logging from spamming too much during ad-hoc runs
import logging

logging.getLogger("graphrag").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
GRAPHRAG_DIR = DATA_DIR / "graphrag"
WORKSPACE_ROOT = GRAPHRAG_DIR / "workspace"
RESULTS_FILE = GRAPHRAG_DIR / "benchmark_results.json"

sys.path.insert(0, str(REPO_ROOT / "src"))

from graphrag.api import local_search
from graphrag.config.load_config import load_config
from graphrag_benchmark.prompts import BENCHMARK_MULTIPLE_CHOICE_INSTRUCTION
from graphrag_benchmark.use_cases.evaluation_module import EvaluationModule

console = Console()


def iter_variant_items(data: dict):
    """Yield only valid variant payloads that contain triples."""
    for variant, payload in data.items():
        if isinstance(payload, dict) and "triples" in payload:
            yield variant, payload


def load_wikidata_labels():
    cache_file = DATA_DIR / "wikidata_labels.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


GLOBAL_LABELS = load_wikidata_labels()
WIKIDATA_LABELS = GLOBAL_LABELS

API_KEY = os.environ.get("OPENAI_API_KEY", "")
os.environ["NVIDIA_NIM_API_KEY"] = API_KEY
os.environ["NVIDIA_NIM_API_BASE"] = "https://integrate.api.nvidia.com/v1"
os.environ["LITELLM_PARAMS_encoding_format"] = "float"
os.environ["LITELLM_PARAMS_input_type"] = "passage"

LLM_MODEL = os.environ.get("LLM_MODEL_NAME", "gpt-4o-mini")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-3-small")


def serialize_context_data(context_data, max_rows: int = 8):
    if not isinstance(context_data, dict):
        return {"raw": str(context_data)}

    serialized = {}
    for key, value in context_data.items():
        if isinstance(value, pd.DataFrame):
            rows = value.head(max_rows).to_dict(orient="records")
            serialized[key] = {
                "row_count": int(len(value)),
                "columns": list(value.columns),
                "rows": rows,
            }
        else:
            serialized[key] = value

    return serialized


def build_context_summary(context_data, max_rows: int = 3):
    if not isinstance(context_data, dict):
        return str(context_data)

    parts = []
    for key, value in context_data.items():
        if isinstance(value, pd.DataFrame):
            preview = value.head(max_rows).to_string(index=False)
            parts.append(f"[{key}]\n{preview}")
        else:
            parts.append(f"[{key}]\n{value}")

    return "\n\n".join(parts)


def map_ground_truth_to_labels(ground_truth: list, global_labels: dict) -> list:
    labels = []
    for gt in ground_truth:
        if not ("://" in gt or "/" in gt):
            labels.append(gt)
            continue

        gt_id = gt.split("/")[-1]
        val = global_labels.get(gt_id, gt_id)
        if isinstance(val, dict):
            label = val.get("label", gt_id)
        else:
            label = str(val)
        labels.append(label)
    return labels


def resolve_wikidata_value(value: str) -> str:
    if not value:
        return ""

    key = value
    if "/entity/" in value or "/prop/direct/" in value:
        key = value.rsplit("/", 1)[-1]

    if key in WIKIDATA_LABELS:
        entry = WIKIDATA_LABELS[key]
        label = entry.get("label") or ""
        aliases = entry.get("aliases") or []
        if label:
            return label
        if aliases:
            return aliases[0]

    if "/entity/" in value and key.startswith("Q"):
        return f"Unknown entity ({key})"

    if "/prop/direct/" in value and key.startswith("P"):
        return f"Unknown property ({key})"

    if value.startswith("http://") or value.startswith("https://"):
        parsed = urlparse(value)
        host = parsed.netloc or value
        return f"URL({host})"

    return value


def render_fact(subject: str, predicate: str, obj: str) -> str:
    subject_label = resolve_wikidata_value(subject)
    predicate_label = resolve_wikidata_value(predicate)
    object_label = resolve_wikidata_value(obj)
    return f"{subject_label} -- {predicate_label} --> {object_label}"


def setup_graphrag_workspace(workspace_dir: str):
    os.makedirs(workspace_dir, exist_ok=True)
    os.makedirs(os.path.join(workspace_dir, "input"), exist_ok=True)

    with open(os.path.join(workspace_dir, "input", "dummy.txt"), "w", encoding="utf-8") as f:
        f.write("Dummy data to bypass input validation.")

    llm_base = os.environ.get("LLM_BASE_URL", "http://localhost:8001/v1")
    emb_base = os.environ.get("EMBEDDING_BASE_URL", "https://integrate.api.nvidia.com/v1")

    settings_content = f"""
encoding_model: cl100k_base

workflows: ["create_communities","create_final_text_units","create_community_reports","generate_text_embeddings"]

cluster_graph:
  max_cluster_size: 10
  use_lcc: false

storage:
  type: file
  base_dir: "artifacts"

completion_models:
  default_completion_model:
    model_provider: openai
    model: {LLM_MODEL}
    api_base: "{llm_base}"
    api_key: "{API_KEY}"
    raw_completion_model: true

embedding_models:
  default_embedding_model:
    model_provider: openai
    model: "{EMBEDDING_MODEL}"
    api_base: "{emb_base}"
    api_key: "{API_KEY}"
    call_args:
        encoding_format: float
"""
    with open(os.path.join(workspace_dir, "settings.yaml"), "w", encoding="utf-8") as f:
        f.write(settings_content)


async def run_graphrag_scenario(question: str, triples: list, variant_name: str, workspace_dir: str) -> dict:
    console.print(f"  [cyan]-> [GraphRAG] Thiết lập workspace: {workspace_dir}[/cyan]")
    setup_graphrag_workspace(workspace_dir)

    artifacts_dir = os.path.join(workspace_dir, "artifacts")
    output_dir = os.path.join(workspace_dir, "output")
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    nodes_dict = {}
    edges_list = []
    text_unit_rows = []

    for idx, triple in enumerate(triples):
        s = triple.get("subject")
        p = triple.get("predicate")
        o = triple.get("object")

        if s and o:
            s = str(s)
            p = str(p)
            o = str(o)
            s_label = resolve_wikidata_value(s)
            p_label = resolve_wikidata_value(p)
            o_label = resolve_wikidata_value(o)
            text_unit_id = f"tu_{idx}"
            relationship_id = f"rel_{idx}"

            triple_text = render_fact(s, p, o)
            text_unit_rows.append(
                {
                    "id": text_unit_id,
                    "text": triple_text,
                    "n_tokens": max(1, len(triple_text.split())),
                    "document_id": f"doc_{variant_name}",
                }
            )

            if s not in nodes_dict:
                nodes_dict[s] = {
                    "id": s,
                    "title": s_label,
                    "type": "ENTITY",
                    "description": f"Wikidata entity {s_label}",
                    "text_unit_ids": [],
                    "human_readable_id": len(nodes_dict),
                    "frequency": 1,
                    "degree": 0,
                    "raw_id": s,
                }
            if o not in nodes_dict:
                nodes_dict[o] = {
                    "id": o,
                    "title": o_label,
                    "type": "ENTITY",
                    "description": f"Wikidata entity {o_label}",
                    "text_unit_ids": [],
                    "human_readable_id": len(nodes_dict),
                    "frequency": 1,
                    "degree": 0,
                    "raw_id": o,
                }

            nodes_dict[s]["text_unit_ids"].append(text_unit_id)
            nodes_dict[o]["text_unit_ids"].append(text_unit_id)

            edges_list.append(
                {
                    "id": relationship_id,
                    "human_readable_id": idx,
                    "source": s_label,
                    "target": o_label,
                    "weight": 1.0,
                    "description": f"{s_label} is connected to {o_label} via property {p_label}",
                    "combined_degree": 0,
                    "text_unit_ids": [text_unit_id],
                    "raw_source": s,
                    "raw_target": o,
                    "raw_predicate": p,
                }
            )

    for node in nodes_dict.values():
        node["text_unit_ids"] = sorted(set(node["text_unit_ids"]))

    entity_columns = ["id", "human_readable_id", "title", "type", "description", "text_unit_ids", "frequency", "degree", "raw_id"]
    relationship_columns = ["id", "human_readable_id", "source", "target", "description", "weight", "combined_degree", "text_unit_ids", "raw_source", "raw_target", "raw_predicate"]
    text_unit_columns = ["id", "text", "n_tokens", "document_id"]

    nodes_df = pd.DataFrame(list(nodes_dict.values()), columns=entity_columns)
    edges_df = pd.DataFrame(edges_list, columns=relationship_columns)
    text_units_df = pd.DataFrame(text_unit_rows, columns=text_unit_columns)

    for seed_dir in [output_dir, artifacts_dir]:
        nodes_df.to_parquet(os.path.join(seed_dir, "nodes.parquet"))
        nodes_df.to_parquet(os.path.join(seed_dir, "entities.parquet"))
        edges_df.to_parquet(os.path.join(seed_dir, "relationships.parquet"))
        nodes_df.to_parquet(os.path.join(seed_dir, "create_final_nodes.parquet"))
        nodes_df.to_parquet(os.path.join(seed_dir, "create_final_entities.parquet"))
        edges_df.to_parquet(os.path.join(seed_dir, "create_final_relationships.parquet"))

    for seed_dir in [output_dir, artifacts_dir]:
        text_units_df.to_parquet(os.path.join(seed_dir, "text_units.parquet"))
        text_units_df.to_parquet(os.path.join(seed_dir, "create_final_text_units.parquet"))

    console.print("  [cyan]-> [GraphRAG] Chạy Indexing Pipeline (Clustering & Summarization)...[/cyan]")
    try:
        process = subprocess.run(
            [sys.executable, "-m", "graphrag", "index", "--root", workspace_dir, "--verbose"],
            check=True,
            capture_output=True,
            text=True,
        )
        console.print(process.stdout)
    except subprocess.CalledProcessError as e:
        console.print("[red]  [LỖI INDEXING GRAPHRAG - CHI TIẾT]:[/red]")
        console.print(f"[yellow]--- STDOUT ---[/yellow]\n{e.stdout}")
        console.print(f"[yellow]--- STDERR ---[/yellow]\n{e.stderr}")
        return {"answer": "INDEX_ERROR", "retrieved_context": ""}

    artifact_dirs = [output_dir, artifacts_dir]
    if os.path.isdir(output_dir):
        subdirs = [
            os.path.join(output_dir, d, "artifacts")
            for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d, "artifacts"))
        ]
        if subdirs:
            artifact_dirs.insert(0, sorted(subdirs)[-1])

    def load_pq(names, default_df):
        for base_dir in artifact_dirs:
            for name in names:
                p = os.path.join(base_dir, name)
                if os.path.exists(p):
                    return pd.read_parquet(p)
        return default_df

    reports_df = load_pq(["community_reports.parquet", "create_final_community_reports.parquet"], pd.DataFrame())
    communities_df = load_pq(["communities.parquet", "create_final_communities.parquet"], pd.DataFrame())
    final_nodes_df = load_pq(["nodes.parquet", "create_final_nodes.parquet"], nodes_df)
    final_entities_df = load_pq(["entities.parquet", "create_final_entities.parquet"], final_nodes_df)
    final_edges_df = load_pq(["relationships.parquet", "create_final_relationships.parquet"], edges_df)
    _loaded_text_units = load_pq(["text_units.parquet", "create_final_text_units.parquet"], text_units_df)
    # Keep only core columns to preserve query behavior while ensuring data consistency
    final_text_units_df = _loaded_text_units[["id", "text", "n_tokens", "document_id"]] if not _loaded_text_units.empty else text_units_df

    if reports_df.empty:
        console.print("[yellow]  [CẢNH BÁO]: Không sinh được Reports. Chạy fallback.[/yellow]")
    else:
        console.print(f"  [green]-> [GraphRAG] Tạo thành công {len(reports_df)} Community Reports.[/green]")

    console.print("  [cyan]-> [GraphRAG] Truy xuất dữ liệu (Local Search API)...[/cyan]")
    try:
        config = load_config(root_dir=workspace_dir)

        covariates_df = pd.DataFrame(
            columns=[
                "id",
                "human_readable_id",
                "subject_id",
                "type",
                "description",
                "object_id",
                "status",
                "start_date",
                "end_date",
                "text_unit_ids",
            ]
        )

        search_params = {
            "config": config,
            "entities": final_entities_df,
            "communities": communities_df,
            "community_reports": reports_df,
            "text_units": final_text_units_df,
            "relationships": final_edges_df,
            "covariates": covariates_df,
            "community_level": 2,
            "response_type": "Single Paragraph",
            "query": question,
        }

        sig = inspect.signature(local_search)
        if "nodes" in sig.parameters:
            search_params["nodes"] = final_nodes_df

        response, context_data = await local_search(**search_params)

        console.print("[bold magenta]--- LOCAL SEARCH ANSWER ---[/bold magenta]")
        console.print(str(response))
        console.print("[bold magenta]--- LOCAL SEARCH CONTEXT ---[/bold magenta]")
        console.print(build_context_summary(context_data))

        return {
            "answer": str(response),
            "retrieved_context": serialize_context_data(context_data),
            "retrieved_context_summary": build_context_summary(context_data),
        }
    except Exception as e:
        console.print(f"[red]  [LỖI QUERY GRAPHRAG API]: {e}[/red]")
        traceback.print_exc()
        return {"answer": "QUERY_ERROR", "retrieved_context": ""}


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

        if len(triples) > 1000:
            console.print(f"[bold yellow]CẢNH BÁO: Bỏ qua biến thể {variant} do quá lớn ({len(triples)} edges)[/bold yellow]")
            ground_truth_labels = map_ground_truth_to_labels(ground_truth, GLOBAL_LABELS)
            results["variants"][variant] = {
                "ground_truth": ground_truth,
                "ground_truth_labels": ground_truth_labels,
                "mcq_correct_letter": payload.get("mcq_correct_letter", "None"),
                "mcq_options": payload.get("mcq_options_text", ""),
                "retrieved_context": "",
                "answer": "SKIPPED_TOO_LARGE",
                "predicted_letter": "None",
                "score": 0.0,
            }
            continue

        options_text = payload.get("mcq_options_text", "")
        correct_letter = payload.get("mcq_correct_letter", "None")
        mcq_question = question_text + options_text + BENCHMARK_MULTIPLE_CHOICE_INSTRUCTION
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

            ground_truth_labels = map_ground_truth_to_labels(ground_truth, GLOBAL_LABELS)
            results["variants"][variant] = {
                "ground_truth": ground_truth,
                "ground_truth_labels": ground_truth_labels,
                "mcq_correct_letter": correct_letter,
                "mcq_options": options_text,
                "retrieved_context": retrieved_context,
                "answer": str(ans),
                "predicted_letter": predicted_letter,
                "score": score,
            }
            console.print(f"[{variant.upper()}] Score: {score}")
        except Exception as e:
            console.print(f"[red]Lỗi ở ID {q_id} Mode {variant}: {e}[/red]")
            traceback.print_exc()
            results["variants"][variant] = {
                "answer": "ERROR",
                "predicted_letter": "None",
                "score": 0.0,
            }

    return results


def main():
    if not API_KEY:
        console.print("[red]Thiếu cấu hình OPENAI_API_KEY!. Chạy huỷ bỏ.[/red]")
        return

    test_json = list((DATA_DIR / "test_perturbed_subgraphs").glob("*.json"))
    if not test_json:
        console.print("[red]Không có file JSON nào trong test_perturbed_subgraphs để thử.[/red]")
        return

    import sys as _sys

    if len(_sys.argv) > 1:
        q_id_target = _sys.argv[1]
        target_file = DATA_DIR / "test_perturbed_subgraphs" / f"{q_id_target}.json"
        if target_file.exists():
            test_file = target_file
        else:
            console.print(f"[red]Không tìm thấy file {q_id_target}.json[/red]")
            return
    else:
        test_file = random.choice([f for f in test_json if f.stem != "2466"])

    with open(test_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    q_id = test_file.stem
    console.print(f"[bold cyan]Đang test trên ID: {q_id}[/bold cyan]")
    console.print(f"[bold yellow]Model đang sử dụng: {os.environ.get('LLM_MODEL_NAME', 'gpt-4o-mini')}[/bold yellow]")

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

    console.print(f"Câu hỏi: {question_text}")

    for variant, payload in iter_variant_items(data):
        triples = payload.get("triples", [])
        ground_truth = payload.get("answers", [])

        options_text = payload.get("mcq_options_text", "")
        correct_letter = payload.get("mcq_correct_letter", "None")
        mcq_question = question_text + options_text + BENCHMARK_MULTIPLE_CHOICE_INSTRUCTION

        work_dir = str(WORKSPACE_ROOT / q_id / variant)

        if os.path.exists(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        os.makedirs(work_dir, exist_ok=True)

        rag_output = asyncio.run(
            run_graphrag_scenario(
                question=mcq_question,
                triples=triples,
                variant_name=variant,
                workspace_dir=work_dir,
            )
        )

        ans = rag_output.get("answer", "")
        retrieved_context = rag_output.get("retrieved_context", "")

        evaluator = EvaluationModule()
        score, predicted_letter = asyncio.run(
            evaluator.compute_llm_based_score(
                predicted_text=str(ans),
                ground_truths=[correct_letter],
                variant=variant,
                global_labels=GLOBAL_LABELS,
                question=mcq_question,
            )
        )

        console.print(f"\n[bold magenta]--- KNOWLEDGE CONTEXT ({variant}) ---[/bold magenta]")
        console.print(retrieved_context)
        console.print(f"[bold cyan]Ground Truth Entities ({variant}):[/bold cyan] {ground_truth}")
        console.print(f"[bold red]MCQ Correct Letter:[/bold red] {correct_letter}")
        console.print(f"[bold blue]MCQ Options:[/bold blue] {options_text}")
        console.print(f"[bold green]Đáp án từ mô hình ({variant}):[/bold green] {ans}")
        console.print(f"[bold yellow]Predicted Letter:[/bold yellow] {predicted_letter}")
        console.print(f"[bold red]LLM Judge Score ({variant}):[/bold red] {score}\n")

    del data
    gc.collect()


if __name__ == "__main__":
    main()
