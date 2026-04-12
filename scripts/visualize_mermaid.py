import os
import json
import random
import requests
from pathlib import Path
from rich.console import Console
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
        "ids": "|".join(q_ids[:50]),
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

def clean_text(uri):
    """Trích xuất ID hoặc dọn dẹp nhãn để không làm vỡ syntax của Mermaid"""
    if not uri: return "UNKNOWN"
    # Lấy đoạn định danh cuối cùng
    text = uri.split('/')[-1]
    # Xóa các ký tự có thể làm lùi cú pháp Mermaid
    for char in ['"', "'", "[", "]", "(", ")", "{", "}"]:
        text = text.replace(char, "")
    return text

def generate_mermaid_diagram(graph_data, answers):
    """Sinh chuỗi mã nguồn Mermaid.js từ danh sách các cạnh (triples)"""
    lines = ["graph TD"]
    triples = graph_data.get("triples", [])
    
    nodes_in_graph = set()
    
    for t in triples:
        s_uri = t.get("subject", "")
        p_uri = t.get("predicate", "")
        o_uri = t.get("object", "")
        
        s = clean_text(s_uri)
        p = clean_text(p_uri)
        o = clean_text(o_uri)
        
        if s and o:
            # Format của mermaid: Node1["Hiển thị_1"] -- "Cạnh" --> Node2["Hiển thị_2"]
            lines.append(f'    {s}["{s}"] -- "{p}" --> {o}["{o}"]')
            nodes_in_graph.add(s)
            nodes_in_graph.add(o)
            
    # Tô màu (Highlight) riêng cho các Entity là Đáp án bằng màu Xanh lá
    for ans in answers:
        a_id = clean_text(ans)
        if a_id in nodes_in_graph:
            lines.append(f'    style {a_id} fill:#9f9,stroke:#333,stroke-width:4px')
            
    return "\n".join(lines)

def main():
    console = Console()
    perturb_dir = Path("data/perturbed_subgraphs")
    
    # Tạo thư mục lưu kết quả ảnh / markdown
    out_dir = Path("visualizations")
    out_dir.mkdir(exist_ok=True)
    
    json_files = list(perturb_dir.glob("*.json"))
    if not json_files:
        console.print("[red]Không tìm thấy graph nào trong data/perturbed_subgraphs/[/red]")
        return
        
    console.print("Đang tra cứu lại nội dung câu hỏi gốc từ dataset...")
    loader = DatasetLoader("data/lcquad_train.json")
    original_questions = loader.load_dataset()
    q_dict = {q.id: q for q in original_questions}
    
    sample_file = random.choice(json_files)
    with open(sample_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if "error" in data:
        console.print("[red]File được chọn đang trúng file báo lỗi (thiếu Answer). Hãy chạy lại lệnh để lấy file khác.[/red]")
        return
        
    clean = data.get("clean", {})
    q_id = str(clean.get("question_id", sample_file.stem))
    
    orig_q = q_dict.get(q_id)
    question_text = orig_q.question if orig_q else "Unknown"
    
    # Lấy trực tiếp từ field 'answers' của JSON
    answers = clean.get("answers", [])
    
    answer_labels = get_wikidata_labels(answers)
    answer_texts = [f"`{ans}` ({answer_labels.get(ans, ans.split('/')[-1])})" for ans in answers]
    
    md_content = f"# Trực quan hóa Biểu đồ: `{q_id}`\n\n"
    md_content += f"**Câu hỏi:** {question_text}\n\n"
    md_content += f"**Đáp án đích (tô <span style='color:green'>màu Xanh</span>):**\n"
    for at in answer_texts:
        md_content += f"- {at}\n"
    md_content += "\n---\n\n"
    
    # Duyệt qua các biến thể (Clean, Broken, ...)
    for variant, v_data in data.items():
        if variant == "error": continue
        md_content += f"## 1. Bản thể `{variant.upper()}`\n\n"
        md_content += "```mermaid\n"
        md_content += generate_mermaid_diagram(v_data, answers)
        md_content += "\n```\n\n"
        
    out_md_file = out_dir / f"{sample_file.stem}.md"
    with open(out_md_file, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    console.print(f"[bold green]Thành công![/bold green] Đã xuất sơ đồ biểu diễn các đồ thị sang định dạng Markdown.")
    console.print(f"File lưu tại: [bold cyan]{out_md_file}[/bold cyan]")
    console.print("=> [yellow]Trong VS Code, bạn hãy mở file trên và bấm tổ hợp phím `Ctrl + Shift + V` (Mở thanh Preview Markdown) để xem sơ đồ mạng lưới cực kì đẹp mắt nhé![/yellow]")

if __name__ == "__main__":
    main()
