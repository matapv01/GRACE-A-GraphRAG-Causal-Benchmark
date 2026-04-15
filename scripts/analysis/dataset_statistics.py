import os
import json
from pathlib import Path
from collections import Counter

def main():
    stat_file = Path("data/dataset_statistics.md")
    
    clean_dir = Path("data/clean_subgraphs")
    perturb_dir = Path("data/perturbed_subgraphs")
    test_perturb_dir = Path("data/test_perturbed_subgraphs")
    lcquad_test_file = Path("data/lcquad_test.json")
    lcquad_train_file = Path("data/lcquad_train.json")
    
    # 1. Total samples in perturbed_subgraphs (Chính là tập data đầy đủ)
    files_to_scan = list(perturb_dir.glob("*.json")) if perturb_dir.exists() else []
    # Nếu test_perturbed_subgraphs có data thì quét thêm
    if test_perturb_dir.exists():
        files_to_scan.extend(list(test_perturb_dir.glob("*.json")))
        
    # Loại bỏ file trùng lặp nếu có ở cả 2 thư mục
    unique_files = {f.name: f for f in files_to_scan}
    test_files = list(unique_files.values())
    total_test_samples = len(test_files)
    
    # 2. Variants distribution
    variant_counts = Counter()
    for f in test_files:
        try:
            with open(f, "r", encoding="utf-8") as file:
                data = json.load(file)
                for variant in data.keys():
                    variant_counts[variant] += 1
        except:
            pass
            
    # 3. Query Types (SELECT vs ASK vs COUNT) from lcquad test
    query_types = Counter()
    test_ids = {f.stem for f in test_files}
    
    lcquad_data = []
    if lcquad_test_file.exists():
        with open(lcquad_test_file, "r", encoding="utf-8") as f:
            lcquad_data.extend(json.load(f))
    if lcquad_train_file.exists():
        with open(lcquad_train_file, "r", encoding="utf-8") as f:
            lcquad_data.extend(json.load(f))
            
    for item in lcquad_data:
        uid = str(item.get("uid"))
        if uid in test_ids:
            query = item.get("sparql_wikidata", item.get("sparql_query", "")).strip().upper()
            if query.startswith("SELECT"):
                # Check if it's a COUNT query
                if "COUNT(" in query:
                    query_types["COUNT"] += 1
                else:
                    query_types["SELECT"] += 1
            elif query.startswith("ASK"):
                query_types["ASK"] += 1
            else:
                query_types["OTHER"] += 1
                
    # Generate Markdown Report
    with open(stat_file, "w", encoding="utf-8") as f:
        f.write("# GRACE Benchmark: Dataset Statistics\n\n")
        f.write(f"**Total Valid Test Samples:** {total_test_samples}\n\n")
        
        f.write("## 1. Graph Variant Distribution\n")
        f.write("| Variant Name | Count | Percentage |\n")
        f.write("|--------------|-------|------------|\n")
        for variant, count in variant_counts.most_common():
            percentage = (count / total_test_samples) * 100 if total_test_samples else 0
            f.write(f"| `{variant}` | {count} | {percentage:.2f}% |\n")
            
        f.write("\n## 2. SPARQL Query Type Distribution (for Test Samples)\n")
        f.write("| Query Type | Count | Percentage |\n")
        f.write("|------------|-------|------------|\n")
        total_queries = sum(query_types.values())
        for qtype, count in query_types.most_common():
            percentage = (count / total_queries) * 100 if total_queries else 0
            f.write(f"| `{qtype}` | {count} | {percentage:.2f}% |\n")
            
    print(f"Successfully generated statistical report at: {stat_file}")

if __name__ == "__main__":
    main()
