import json
from pathlib import Path
from collections import Counter

def main():
    stat_file = Path("data/dataset_statistics.md")

    # Current benchmark data after convert_to_mcq.py
    perturb_dir = Path("data/test_perturbed_subgraphs")
    lcquad_test_file = Path("data/lcquad_test.json")

    # 1. Total samples in perturbed_subgraphs (already converted to MCQ)
    files_to_scan = list(perturb_dir.glob("*.json")) if perturb_dir.exists() else []

    # Filter out duplicates if any
    unique_files = {f.name: f for f in files_to_scan}
    test_files = list(unique_files.values())
    total_test_samples = len(test_files)

    # 2. Variants distribution
    variant_counts = Counter()
    question_time_group_counts = Counter()
    
    # Store example query IDs for each group
    group_examples = {"timestamp": [], "non_timestamp": [], "unknown": []}
    for f in test_files:
        try:
            with open(f, "r", encoding="utf-8") as file:
                data = json.load(file)

                # Distribution of question types after MCQ conversion
                q_group = data.get("question_time_group", "unknown")
                if q_group not in {"timestamp", "non_timestamp"}:
                    q_group = "unknown"
                question_time_group_counts[q_group] += 1
                
                # Store sample query ID
                if len(group_examples[q_group]) < 5:
                    group_examples[q_group].append(f.stem)

                for variant in data.keys():
                    if isinstance(data.get(variant), dict):
                        variant_counts[variant] += 1
        except Exception:
            pass

    # 3. Query Types (SELECT vs ASK vs COUNT) from lcquad test
    query_types = Counter()
    test_ids = {f.stem for f in test_files}

    if lcquad_test_file.exists():
        with open(lcquad_test_file, "r", encoding="utf-8") as f:
            lcquad_data = json.load(f)
    else:
        lcquad_data = []

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
        f.write("# GRACE-A Benchmark: Dataset Statistics\n\n")
        f.write(f"**Total Valid Test Samples:** {total_test_samples}\n\n")

        f.write("## 1. Question Time Group Distribution\n")
        f.write("| Question Group | Count | Percentage | Example Query IDs |\n")
        f.write("|----------------|-------|------------|-------------------|\n")
        for group in ["timestamp", "non_timestamp", "unknown"]:
            count = question_time_group_counts.get(group, 0)
            if count == 0 and group == "unknown":
                continue
            percentage = (count / total_test_samples) * 100 if total_test_samples else 0
            examples = ", ".join(group_examples[group]) if group_examples[group] else "-"
            f.write(f"| `{group}` | {count} | {percentage:.2f}% | {examples} |\n")

        f.write("\n## 2. Graph Variant Distribution\n")
        f.write("| Variant Name | Count | Percentage |\n")
        f.write("|--------------|-------|------------|\n")
        for variant, count in variant_counts.most_common():
            percentage = (count / total_test_samples) * 100 if total_test_samples else 0
            f.write(f"| `{variant}` | {count} | {percentage:.2f}% |\n")

        f.write("\n## 3. SPARQL Query Type Distribution (for Test Samples)\n")
        f.write("| Query Type | Count | Percentage |\n")
        f.write("|------------|-------|------------|\n")
        total_queries = sum(query_types.values())
        for qtype, count in query_types.most_common():
            percentage = (count / total_queries) * 100 if total_queries else 0
            f.write(f"| `{qtype}` | {count} | {percentage:.2f}% |\n")

    print(f"Successfully generated statistical report at: {stat_file}")

if __name__ == "__main__":
    main()
