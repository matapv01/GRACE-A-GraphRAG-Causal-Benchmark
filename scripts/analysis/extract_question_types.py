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
            template = item.get("template", "").strip()

            writer.writerow([q_id, subgraph, template_id, template])

    print(f"Đã tạo file ánh xạ thành công tại: {output_file}")
    print(f"Tổng số mẫu: {len(data)}")


if __name__ == "__main__":
    main()
