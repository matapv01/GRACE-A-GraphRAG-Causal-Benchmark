import json
import csv
from pathlib import Path

def main():
    input_file = Path("data/lcquad_test.json")
    output_file = Path("data/question_type_mapping.csv")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["question_id", "subgraph_type", "template_id", "template_structure"]
        )

        for item in data:
            q_id = str(item.get("uid", ""))

            subgraph = item.get("subgraph", "")
            if isinstance(subgraph, list):
                subgraph = ", ".join(subgraph)

            template_id = str(item.get("template_id", ""))
            
            template = item.get("template", "")
            if isinstance(template, list):
                template = ", ".join(str(t) for t in template)
            template = str(template).strip()

            writer.writerow([q_id, subgraph, template_id, template])

    print(f"Successfully generated mapping file at: {output_file}")
    print(f"Total samples: {len(data)}")

if __name__ == "__main__":
    main()
