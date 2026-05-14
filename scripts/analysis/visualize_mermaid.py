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
    # Fetch labels in batches of 50 to avoid URL length limits
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
            "User-Agent": "GraphRAG-Causal-Benchmark/2.0 (https://github.com/MinhPV/GRACE-A-GraphRAG-Causal-Benchmark)"
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
    """Format Wikidata URI to valid Mermaid node ID."""
    node_id = uri.split("/")[-1]
    node_id = re.sub(r"[^a-zA-Z0-9_]", "_", node_id)
    return node_id

def sanitize_label(label):
    """Escape quotes for Mermaid text."""
    return str(label).replace('"', "&quot;")

def generate_mermaid(triples, labels_dict):
    lines = ["graph TD"]
    for idx, t in enumerate(triples):
        s = t.get("subject", "")
        p = t.get("predicate", "")
        o = t.get("object", "")

        s_id = clean_id(s)
        o_id = clean_id(o)
        
        s_label = sanitize_label(labels_dict.get(s, s.split("/")[-1]))
        o_label = sanitize_label(labels_dict.get(o, o.split("/")[-1]))
        p_label = sanitize_label(labels_dict.get(p, p.split("/")[-1]))

        # Format: NodeID["Label"] -->|Predicate| NodeID["Label"]
        lines.append(f'    {s_id}["{s_label}"] -->|"{p_label}"| {o_id}["{o_label}"]')
    
    # Add click events so nodes link to actual Wikidata pages
    added_links = set()
    for t in triples:
        for entity in [t.get("subject", ""), t.get("object", "")]:
            e_id = clean_id(entity)
            if e_id not in added_links and entity.startswith("http://www.wikidata.org/entity/Q"):
                lines.append(f'    click {e_id} "{entity}" "Go to Wikidata"')
                added_links.add(e_id)

    return "\n".join(lines)

def search_files(data_dir, mode=None, single_id=None):
    import csv
    if not data_dir.exists():
        return []
    
    files = list(data_dir.glob("*.json"))
    
    if single_id:
        files = [f for f in files if f.stem == str(single_id)]
        return files
        
    if mode:
        mapping_file = Path("data/question_type_mapping.csv")
        if not mapping_file.exists():
             Console().print("[red]Missing mapping file. Please run extract_question_types.py first.[/red]")
             return []
        
        valid_ids = set()
        with open(mapping_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("subgraph_type", "").lower() == mode.lower():
                    valid_ids.add(row["question_id"])
                    
        files = [f for f in files if f.stem in valid_ids]
        
    return files

def main():
    parser = argparse.ArgumentParser(description="Visualize Reasonings Subgraphs using Mermaid.js")
    parser.add_argument("--mode", type=str, help="Filter by subgraph type (e.g. 'right-subgraph', 'center')")
    parser.add_argument("--single_id", type=str, help="Render specific question ID directly")
    args = parser.parse_args()

    console = Console()
    data_dir = Path("data/test_perturbed_subgraphs")
    
    files = search_files(data_dir, mode=args.mode, single_id=args.single_id)
    
    if not files:
        console.print("[red]No matching files found based on the provided filters.[/red]")
        return
        
    # Select exactly 1 file to print if single_id is not specified (random sample)
    if args.single_id:
        selected_file = files[0]
    else:
        selected_file = random.choice(files)
        
    console.print(f"[bold cyan]Selected file for visualization:[/bold cyan] {selected_file.name}")
    
    with open(selected_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if "error" in data:
        console.print("[red]File contains an error tag, skipping.[/red]")
        return
        
    # Collect unique URIs to fetch labels at once
    all_uris = set()
    for variant, v_data in data.items():
        if not isinstance(v_data, dict): continue
        triples = v_data.get("triples", [])
        for t in triples:
            all_uris.add(t.get("subject", ""))
            all_uris.add(t.get("predicate", ""))
            all_uris.add(t.get("object", ""))
            
    console.print("Pulling human readable labels from Wikidata API...")
    labels_dict = get_wikidata_labels(list(all_uris))
    
    for variant, v_data in data.items():
        if not isinstance(v_data, dict): continue
        triples = v_data.get("triples", [])
        if not triples:
            continue
            
        console.print(f"\n[bold green]=== MERMAID DIAGRAM: {variant.upper()} ===[/bold green]")
        console.print("[dim]Copy paste the text below into: https://mermaid.live/[/dim]\n")
        
        mermaid_text = generate_mermaid(triples, labels_dict)
        # Using Rich to print standard raw text inside a block
        console.print(mermaid_text, style="white")
        console.print("\n" + "-"*50)

if __name__ == "__main__":
    main()
