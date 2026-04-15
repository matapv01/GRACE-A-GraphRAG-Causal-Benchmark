import json
import random
import argparse
import requests
from pathlib import Path
from rich.console import Console
import re


def get_wikidata_labels(uris):
    q_ids = []
    uri_mapping = {}
    for uri in uris:
        q_id = uri.split("/")[-1]
        if q_id and re.match(r"^[PQ]\d+$", q_id):
            q_ids.append(q_id)
            uri_mapping[q_id] = uri

    if not q_ids:
        return {}

    labels = {}
    # Lấy nhãn cho từng lô 50 ID để tránh lỗi URL quá dài
    for i in range(0, len(q_ids), 50):
        batch = q_ids[i : i + 50]
        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbgetentities",
            "ids": "|".join(batch),
            "languages": "en",
            "props": "labels",
            "format": "json",
        }
        headers = {
            "User-Agent": "GraphRAG-Benchmark/1.0 (https://github.com/MinhPV/GraphRAG-Benchmark)"
        }
        try:
            resp = requests.get(url, params=params, headers=headers).json()
            for q_id, info in resp.get("entities", {}).items():
                if "labels" in info and "en" in info["labels"]:
                    labels[uri_mapping[q_id]] = info["labels"]["en"]["value"]
                else:
                    labels[uri_mapping[q_id]] = q_id
        except Exception:
            pass

    return labels


def clean_id(uri):
    """Lấy ID thuần túy làm định danh node cho Mermaid (không chứa khoảng trắng/ký tự đặc biệt)"""
    if not uri:
        return "UNKNOWN"
    text = uri.split("/")[-1]
    # Replaces everything that isn't a letter, number, or underscore with an underscore
    import re

    text = re.sub(r"[^a-zA-Z0-9_]", "_", text)
    return text


def clean_label(text):
    """Dọn dẹp text hiển thị để không làm vỡ syntax của Mermaid"""
    if not text:
        return "UNKNOWN"
    for char in ['"', "'", "[", "]", "(", ")", "{", "}"]:
        text = text.replace(char, " ")
    return text


def generate_mermaid_diagram(graph_data, answers, labels_dict):
    """Sinh chuỗi mã nguồn Mermaid.js từ danh sách các cạnh (triples)"""
    lines = [
        "%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '18px', 'fontFamily': 'Arial'}, 'flowchart': {'nodeSpacing': 70, 'rankSpacing': 100}}}%%",
        "graph LR",
    ]
    triples = graph_data.get("triples", [])

    nodes_in_graph = set()

    for t in triples:
        s_uri = t.get("subject", "")
        p_uri = t.get("predicate", "")
        o_uri = t.get("object", "")

        s_id = clean_id(s_uri)
        p_id = clean_id(p_uri)
        o_id = clean_id(o_uri)

        s_label = clean_label(labels_dict.get(s_uri, s_id))
        p_label = clean_label(labels_dict.get(p_uri, p_id))
        o_label = clean_label(labels_dict.get(o_uri, o_id))

        if s_id and o_id:
            # Format của mermaid: Node1["Hiển thị_1"] -- "Cạnh" --> Node2["Hiển thị_2"]
            lines.append(
                f'    {s_id}["{s_label}"] -- "{p_label}" --> {o_id}["{o_label}"]'
            )
            nodes_in_graph.add(s_id)
            nodes_in_graph.add(o_id)

    # Tô màu (Highlight) riêng cho các Entity là Đáp án bằng màu Xanh lá
    for ans in answers:
        a_id = clean_id(ans)
        # Chỉ highlight nếu đáp án thực sự tồn tại trong đồ thị
        if a_id in nodes_in_graph:
            lines.append(f"    style {a_id} fill:#9f9,stroke:#333,stroke-width:4px")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize perturbed subgraphs as Mermaid diagrams"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="Chỉ vẽ các đồ thị thuộc loại câu hỏi subgraph tương ứng (vd: 'simple question right', 'statement_property')",
    )
    parser.add_argument(
        "--single_id", type=str, default=None, help="Chỉ vẽ cho question_id cụ thể"
    )
    args = parser.parse_args()

    console = Console()
    perturb_dir = Path("data/perturbed_subgraphs")

    # Tạo thư mục lưu kết quả ảnh / markdown
    out_dir = Path("visualizations")
    out_dir.mkdir(exist_ok=True)

    json_files = list(perturb_dir.glob("*.json"))
    if not json_files:
        console.print(
            "[red]Không tìm thấy graph nào trong data/perturbed_subgraphs/[/red]"
        )
        return

    console.print("Đang tra cứu lại nội dung câu hỏi gốc từ dataset...")
    try:
        with open("data/lcquad_train.json", "r", encoding="utf-8") as f:
            original_questions = json.load(f)
    except Exception as e:
        console.print(f"[red]Không thể đọc data/lcquad_train.json: {e}[/red]")
        return

    q_dict = {str(q.get("uid", "")): q for q in original_questions}

    # Lọc danh sách file JSON dựa trên tham số mode hoặc single_id
    valid_files = []
    for f in json_files:
        uid = f.stem
        if args.single_id and uid != args.single_id:
            continue

        if args.mode:
            q_info = q_dict.get(uid)
            if not q_info:
                continue
            subgraph_type = q_info.get("subgraph", "")
            if isinstance(subgraph_type, list):
                subgraph_type = " ".join(subgraph_type)

            if args.mode.lower() not in subgraph_type.lower():
                continue

        valid_files.append(f)

    if not valid_files:
        console.print(
            f"[red]Không tìm thấy file nào thỏa mãn điều kiện (Mode: {args.mode}, ID: {args.single_id})[/red]"
        )
        return

    sample_file = random.choice(valid_files)
    with open(sample_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "error" in data:
        console.print(
            "[red]File được chọn đang trúng file báo lỗi (thiếu Answer). Hãy chạy lại lệnh để lấy file khác.[/red]"
        )
        return

    clean = data.get("clean", {})
    q_id = str(clean.get("question_id", sample_file.stem))

    orig_q = q_dict.get(q_id)
    question_text = orig_q.get("question", "Unknown") if orig_q else "Unknown"

    answers = clean.get("answers", [])

    # 1. Thu thập TẤT CẢ các URI có trong các đồ thị để tra cứu label
    all_uris = set(answers)
    for variant, v_data in data.items():
        if variant == "error":
            continue
        for t in v_data.get("triples", []):
            all_uris.add(t.get("subject", ""))
            all_uris.add(t.get("predicate", ""))
            all_uris.add(t.get("object", ""))

    all_uris = {u for u in all_uris if u}  # Loại bỏ chuỗi rỗng

    console.print(f"Đang gọi Wikidata API để lấy {len(all_uris)} nhãn (labels)...")
    labels_dict = get_wikidata_labels(list(all_uris))

    answer_texts = [
        f"`{ans}` ({labels_dict.get(ans, ans.split('/')[-1])})" for ans in answers
    ]

    md_content = f"# Trực quan hóa Biểu đồ: `{q_id}`\n\n"
    md_content += f"**Câu hỏi:** {question_text}\n\n"
    md_content += "**Đáp án đích (tô <span style='color:green'>màu Xanh</span>):**\n"
    for at in answer_texts:
        md_content += f"- {at}\n"
    md_content += "\n---\n\n"

    # 2. Duyệt qua các biến thể (Clean, Broken, ...)
    for variant, v_data in data.items():
        if variant == "error":
            continue
        md_content += f"## Bản thể `{variant.upper()}`\n\n"
        md_content += "```mermaid\n"
        md_content += generate_mermaid_diagram(v_data, answers, labels_dict)
        md_content += "\n```\n\n"

    out_md_file = out_dir / f"{sample_file.stem}.md"
    with open(out_md_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    console.print(
        "[bold green]Thành công![/bold green] Đã xuất sơ đồ biểu diễn các đồ thị sang định dạng Markdown."
    )
    console.print(f"File lưu tại: [bold cyan]{out_md_file}[/bold cyan]")
    console.print(
        "=> [yellow]Trong VS Code, bạn hãy mở file trên và bấm tổ hợp phím `Ctrl + Shift + V` (Mở thanh Preview Markdown) để xem sơ đồ mạng lưới cực kì đẹp mắt nhé![/yellow]"
    )


if __name__ == "__main__":
    main()
