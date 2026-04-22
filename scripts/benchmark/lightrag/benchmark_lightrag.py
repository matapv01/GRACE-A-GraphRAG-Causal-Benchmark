import os
import json
import asyncio
from pathlib import Path
from rich.console import Console

from dotenv import load_dotenv
load_dotenv()

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from graphrag_benchmark.prompts import BENCHMARK_MULTIPLE_CHOICE_INSTRUCTION, generate_abcd_options
from graphrag_benchmark.use_cases.evaluation_module import EvaluationModule


console = Console()

def load_wikidata_labels():
    cache_file = "data/wikidata_labels.json"
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

GLOBAL_LABELS = load_wikidata_labels()

async def custom_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    model_name = os.environ.get("LLM_MODEL_NAME", "gpt-4o-mini")
    base_url = os.environ.get("LLM_BASE_URL", os.environ.get("OPENAI_BASE_URL"))
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url=base_url,
        **kwargs
    )

async def custom_embedding_func(texts, **kwargs):
    base_url = os.environ.get("EMBEDDING_BASE_URL", os.environ.get("OPENAI_BASE_URL"))
    embed_model = os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
    
    # Ở đây chúng ta phải gọi ".func" để vượt qua wrapper mặc định của LightRAG (vốn đang lock ở 1536 chiều)
    return await openai_embed.func(
        texts,
        model=embed_model, 
        base_url=base_url,
        **kwargs
    )

async def run_lightrag_scenario(question: str, triples: list, variant_name: str, workspace_dir: str):
    verbalized_text = "Here is the knowledge context:\n"
    for t in triples:
        s, p, o = t.get("subject", ""), t.get("predicate", ""), t.get("object", "")
        
        s_id = s.split("/")[-1]
        p_id = p.split("/")[-1]
        o_id = o.split("/")[-1]
        
        def format_entity(eid):
            val = GLOBAL_LABELS.get(eid, eid)
            if isinstance(val, dict):
                text = val.get("label", eid)
                desc = val.get("description", "")
                aliases = val.get("aliases", [])
                
                parts = []
                if desc:
                    parts.append(desc)
                if aliases:
                    parts.append(f"also known as {', '.join(aliases)}")
                    
                if parts:
                    text += f" ({'; '.join(parts)})"
                return text
            return str(val)

        s_text = format_entity(s_id)
        p_text = format_entity(p_id)
        o_text = format_entity(o_id)
        
        verbalized_text += f"{s_text} {p_text} {o_text}.\n"

    console.print(f"[{variant_name.upper()}] Thư mục làm việc: {workspace_dir}")
    
    rag = LightRAG(
        working_dir=workspace_dir,
        llm_model_func=custom_llm_func,
        llm_model_name=os.environ.get("LLM_MODEL_NAME", "meta/llama-3.1-405b-instruct"),
        embedding_func=EmbeddingFunc(
            embedding_dim=int(os.environ.get("EMBEDDING_DIM", 4096)),
            max_token_size=16384,
            func=custom_embedding_func
        ),
    )
    
    # Ở phiên bản mới, bắt buộc phải initialize_storages() trước khi insert
    await rag.initialize_storages()

    console.print(f"[{variant_name.upper()}] Đang xây dựng Index ...")
    await rag.ainsert(verbalized_text)

    console.print(f"[{variant_name.upper()}] Đang Query ...")
    
    question = question if question is not None else "What is the relation between these entities?"

    # Lấy Context
    retrieved_context = await rag.aquery(question, param=QueryParam(mode="local", only_need_context=True))

    # Ép LLM trả về Short Answer để phục vụ tính toán Causal Benchmark
    strict_question = question + BENCHMARK_MULTIPLE_CHOICE_INSTRUCTION
    
    answer = await rag.aquery(strict_question, param=QueryParam(mode="local"))
    
    # Dọn dẹp RAM
    del rag
    import gc
    gc.collect()
    
    return {
        "answer": answer,
        "retrieved_context": retrieved_context
    }

def main():
    import nest_asyncio
    import sys
    import random
    nest_asyncio.apply()

    if not os.environ.get("OPENAI_API_KEY"):
        console.print("[red]Thiếu cấu hình OPENAI_API_KEY![/red]")
        console.print("Vui lòng sao chép file [cyan]`.env.template`[/cyan] thành [cyan]`.env`[/cyan] và điền khóa API của bạn.")
        return

    test_json = list(Path("data/test_perturbed_subgraphs").glob("*.json"))
    if not test_json:
        console.print("[red]Không có file JSON nào trong test_perturbed_subgraphs để thử.[/red]")
        return
        
    # Cho phép truyền ID qua tham số dòng lệnh, nếu không thì lấy random 1 file bất kỳ
    if len(sys.argv) > 1:
        q_id_target = sys.argv[1]
        target_file = Path(f"data/test_perturbed_subgraphs/{q_id_target}.json")
        if target_file.exists():
            test_file = target_file
        else:
            console.print(f"[red]Không tìm thấy file {q_id_target}.json[/red]")
            return
    else:
        test_file = random.choice([f for f in test_json if f.stem != "2466"]) # Chắc chắn né file 2466 đã dùng

    with open(test_file, "r") as f:
        data = json.load(f)

    q_id = test_file.stem
    console.print(f"[bold cyan]Đang test trên ID: {q_id}[/bold cyan]")
    console.print(f"[bold yellow]Model đang sử dụng: {os.environ.get('LLM_MODEL_NAME', 'gpt-4o-mini')}[/bold yellow]")
    
    question_text = "What is the relation between these entities?" 
    try:
        with open("data/lcquad_test.json", "r") as f:
            all_q = json.load(f)
            for q in all_q:
                if str(q.get("uid")) == str(q_id):
                    question_text = q.get("question")
                    break
    except FileNotFoundError:
        pass

    console.print(f"Câu hỏi: {question_text}")

    for variant in data.keys():
        triples = data[variant].get("triples", [])
        ground_truth = data[variant].get("answers", [])
        
        # Đọc MCQ Options trực tiếp từ file JSON thay vì sinh real-time
        options_text = data[variant].get("mcq_options_text", "")
        correct_letter = data[variant].get("mcq_correct_letter", "None")
        mcq_question = question_text + options_text
        
        work_dir = f"data/lightrag/workspace/{q_id}/{variant}"
        
        import shutil
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        os.makedirs(work_dir, exist_ok=True)
        
        import asyncio
        rag_output = asyncio.run(run_lightrag_scenario(
            question=mcq_question, 
            triples=triples, 
            variant_name=variant, 
            workspace_dir=work_dir
        ))
        
        ans = rag_output.get("answer", "")
        retrieved_context = rag_output.get("retrieved_context", "")

        # Chấm điểm LLM Judge bằng Đáp án là 1 mảng chứa chữ cái chuẩn ["A"]
        evaluator = EvaluationModule()
        score = asyncio.run(evaluator.compute_llm_based_score(
            predicted_text=str(ans),
            ground_truths=[correct_letter],
            variant=variant,
            global_labels=GLOBAL_LABELS,
            question=mcq_question
        ))
        
        console.print(f"\n[bold magenta]--- KNOWLEDGE CONTEXT ({variant}) ---[/bold magenta]")
        console.print(retrieved_context)
        console.print(f"[bold cyan]Ground Truth Entities ({variant}):[/bold cyan] {ground_truth}")
        console.print(f"[bold red]MCQ Correct Letter:[/bold red] {correct_letter}")
        console.print(f"[bold blue]MCQ Options:[/bold blue] {options_text}")
        console.print(f"[bold green]Đáp án từ mô hình ({variant}):[/bold green] {ans}")
        console.print(f"[bold red]LLM Judge Score ({variant}):[/bold red] {score}\n")

if __name__ == "__main__":
    main()
