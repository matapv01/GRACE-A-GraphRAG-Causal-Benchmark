import os
import json
import time
from pathlib import Path
from rich.console import Console
from tqdm import tqdm
import requests

console = Console()

def get_wikidata_labels(ids: set) -> dict:
    labels_dict = {}
    missing_ids = list(ids)
    chunk_size = 50
    
    url = "https://www.wikidata.org/w/api.php"
    headers = {"User-Agent": "GraphRAG_Causal_Benchmark_Gen/1.0"}
    
    for i in tqdm(range(0, len(missing_ids), chunk_size), desc="Fetching from Wikidata"):
        chunk = missing_ids[i:i+chunk_size]
        ids_str = "|".join(chunk)
        params = {
            "action": "wbgetentities",
            "ids": ids_str,
            "format": "json",
            "props": "labels",
            "languages": "en"
        }
        
        # Hàm fallback lấy từng ID
        def fetch_single(single_id):
            try:
                r = requests.get(url, params={"action": "wbgetentities", "ids": single_id, "format": "json", "props": "labels", "languages": "en"}, headers=headers, timeout=5)
                data = r.json()
                if "entities" in data and single_id in data["entities"]:
                    entity_data = data["entities"][single_id]
                    if "labels" in entity_data and "en" in entity_data["labels"]:
                        return entity_data["labels"]["en"]["value"]
            except Exception:
                pass
            return single_id

        try:
            r = requests.get(url, params=params, headers=headers, timeout=10)
            data = r.json()
            if "entities" in data:
                for eid in chunk:
                    entity_data = data["entities"].get(eid, {})
                    if "labels" in entity_data and "en" in entity_data["labels"]:
                        labels_dict[eid] = entity_data["labels"]["en"]["value"]
                    else:
                        labels_dict[eid] = eid
            else:
                for eid in chunk:
                    labels_dict[eid] = fetch_single(eid)
        except Exception as e:
            console.print(f"\n[yellow]Batch failed, falling back to single extraction for this chunk...[/yellow]")
            for eid in chunk:
                labels_dict[eid] = fetch_single(eid)
                
        time.sleep(0.1) # Dãn cách request
        
    return labels_dict

def main():
    console.print("[bold cyan]Bắt đầu quét dữ liệu để tìm Wikidata IDs...[/bold cyan]")
    
    all_ids = set()
    
    # Quét qua các thư mục chứa JSON
    folders_to_scan = ["data/clean_subgraphs", "data/test_perturbed_subgraphs"]
    for folder in folders_to_scan:
        if not os.path.exists(folder):
            continue
        for filepath in Path(folder).glob("*.json"):
            with open(filepath, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    
                    def extract_from_triples(triples):
                        for t in triples:
                            for key in ["subject", "predicate", "object"]:
                                val = t.get(key, "")
                                if isinstance(val, str) and ("wikidata.org/entity/" in val or "wikidata.org/prop/direct/" in val):
                                    all_ids.add(val.split("/")[-1])
                                    
                    if isinstance(data, list):
                        extract_from_triples(data)
                    elif isinstance(data, dict):
                        if "triples" in data:
                            extract_from_triples(data["triples"])
                        else:
                            for k, v in data.items():
                                if isinstance(v, dict) and "triples" in v:
                                    extract_from_triples(v["triples"])
                except Exception as e:
                    console.print(f"[red]Error reading {filepath}: {e}[/red]")
                    
    console.print(f"Tổng số Wikidata IDs được tìm thấy: [bold yellow]{len(all_ids)}[/bold yellow]")
    
    out_file = "data/wikidata_labels.json"
    existing_labels = {}
    if os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            existing_labels = json.load(f)
            
    missing_ids = all_ids - set(existing_labels.keys())
    console.print(f"Số IDs chưa có labels: [bold yellow]{len(missing_ids)}[/bold yellow]")
    
    if missing_ids:
        new_labels = get_wikidata_labels(missing_ids)
        existing_labels.update(new_labels)
        
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(existing_labels, f, indent=4, ensure_ascii=False)
            
    console.print(f"[bold green]OK! Toàn bộ ánh xạ đã được lưu tại {out_file}[/bold green]")

if __name__ == "__main__":
    main()