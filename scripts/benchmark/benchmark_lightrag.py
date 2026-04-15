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
    base_url = os.environ.get("OPENAI_BASE_URL")
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url=base_url,
        **kwargs
    )

async def custom_embedding_func(texts, **kwargs):
    base_url = os.environ.get("OPENAI_BASE_URL")
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
        
        s_label = GLOBAL_LABELS.get(s_id, s_id)
        p_label = GLOBAL_LABELS.get(p_id, p_id)
        o_label = GLOBAL_LABELS.get(o_id, o_id)
        
        verbalized_text += f"{s_label} {p_label} {o_label}.\n"

    console.print(f"[{variant_name.upper()}] Thư mục làm việc: {workspace_dir}")
    
    rag = LightRAG(
        working_dir=workspace_dir,
        llm_model_func=custom_llm_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=int(os.environ.get("EMBEDDING_DIM", 4096)),
            max_token_size=8192,
            func=custom_embedding_func
        ),
    )
    
    # Ở phiên bản mới, bắt buộc phải initialize_storages() trước khi insert
    await rag.initialize_storages()

    console.print(f"[{variant_name.upper()}] Đang xây dựng Index ...")
    rag.insert(verbalized_text)

    console.print(f"[{variant_name.upper()}] Đang Query ...")
    answer = rag.query(question, param=QueryParam(mode="local"))
    
    return answer

def main():
    import nest_asyncio
    nest_asyncio.apply()

    if not os.environ.get("OPENAI_API_KEY"):
        console.print("[red]Thiếu cấu hình OPENAI_API_KEY![/red]")
        console.print("Vui lòng sao chép file [cyan]`.env.template`[/cyan] thành [cyan]`.env`[/cyan] và điền khóa API của bạn.")
        return

    test_json = list(Path("data/test_perturbed_subgraphs").glob("*.json"))
    if not test_json:
        console.print("[red]Không có file JSON nào trong test_perturbed_subgraphs để thử.[/red]")
        return
        
    test_file = test_json[0]
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

    for variant in ["clean", "literal"]:
        if variant in data:
            triples = data[variant].get("triples", [])
            work_dir = f"./lightrag_workspace/{q_id}/{variant}"
            os.makedirs(work_dir, exist_ok=True)
            
            import asyncio
            ans = asyncio.run(run_lightrag_scenario(
                question=question_text, 
                triples=triples, 
                variant_name=variant, 
                workspace_dir=work_dir
            ))
            console.print(f"[bold green]Đáp án từ mô hình ({variant}):[/bold green] {ans}\n")

if __name__ == "__main__":
    main()
