import json
import random
import requests
from pathlib import Path
from rich.console import Console
from rich.tree import Tree
from graphrag_benchmark.infrastructure.dataset_loader import DatasetLoader

def get_wikidata_labels(uris):
    q_ids = []
    uri_mapping = {}
    for uri in uris:
        q_id = uri.split('/')[-1]
        if q_id:
            q_ids.append(q_id)
            uri_mapping[q_id] = uri
            
    if not q_ids: return {}
    
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": "|".join(q_ids[:50]), # Wikidata API cho phép tối đa 50 ID mỗi request
        "languages": "en",
        "props": "labels",
        "format": "json"
    }
    headers = {
        "User-Agent": "GraphRAG-Benchmark/1.0 (https://github.com/MinhPV/GraphRAG-Benchmark)"
    }
    try:
        resp = requests.get(url, params=params, headers=headers).json()
        labels = {}
        for q_id, info in resp.get("entities", {}).items():
            if "labels" in info and "en" in info["labels"]:
                labels[uri_mapping[q_id]] = info["labels"]["en"]["value"]
            else:
                labels[uri_mapping[q_id]] = q_id
        return labels
    except Exception:
        return {}

def main():
    console = Console()
    perturb_dir = Path("data/perturbed_subgraphs")
    
    if not perturb_dir.exists():
        console.print("[red]Không tìm thấy thư mục: data/perturbed_subgraphs/[/red]")
        return
        
    json_files = list(perturb_dir.glob("*.json"))
    if not json_files:
        console.print("[red]Chưa có file perturbed subgraph nào được tạo ra.[/red]")
        return
        
    # Tải bộ dataset gốc để tra cứu lại Question text và Answers
    console.print("Đang tra cứu lại nội dung câu hỏi gốc từ dataset...")
    loader = DatasetLoader("data/lcquad_train.json")
    questions = loader.load_dataset()
    q_dict = {q.id: q for q in questions}
    
    # Lấy ngẫu nhiên 2 mẫu để in ra
    num_samples = min(2, len(json_files))
    sample_files = random.sample(json_files, num_samples)
    
    for file_path in sample_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if "error" in data:
            continue
            
        clean_graph = data.get("clean")
        if not clean_graph:
            continue
            
        q_id = clean_graph.get("question_id", file_path.stem)
        
        # Lấy câu hỏi từ dataset gốc
        orig_q = q_dict.get(q_id)
        question_text = orig_q.question if orig_q else "Unknown Question"
        
        # Lấy answers trực tiếp từ file JSON
        answers = clean_graph.get("answers", [])
        
        # Query Wikidata API để lấy text cho dễ nhìn
        answer_labels = get_wikidata_labels(answers)
        answer_texts = [f"{ans} ({answer_labels.get(ans, ans.split('/')[-1])})" for ans in answers]

        console.print(f"\n[bold cyan]=== SAMPLE ID: {q_id} ===[/bold cyan]")
        console.print(f"[bold yellow]Câu hỏi (Query):[/bold yellow] {question_text}")
        console.print(f"[bold yellow]Đáp án đích (Links):[/bold yellow] {answers}")
        console.print(f"[bold yellow]Đáp án đích (Text) :[/bold yellow] {answer_texts}\n")
        
        # In ra cấu trúc của từng bản variant
        for variant_name, v_data in data.items():
            triples = v_data.get("triples", [])
            
            # Sử dụng Tree của thư viện 'rich' để in cho tường minh
            tree = Tree(f"[bold green]Biểu đồ {variant_name.upper()}[/bold green] (Tổng số {len(triples)} cạnh)")
            for t in triples:
                s = t.get('subject', '').split('/')[-1] # Lọc bớt URI dài, lấy id cuối
                p = t.get('predicate', '').split('/')[-1]
                o = t.get('object', '').split('/')[-1]
                tree.add(f"{s} [blue]--({p})-->[/blue] {o}")
            
            console.print(tree)

if __name__ == "__main__":
    main()
