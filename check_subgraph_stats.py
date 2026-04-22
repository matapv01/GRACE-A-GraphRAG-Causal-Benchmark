import os
import json
import glob

def find_outlier_questions(directory="data/test_perturbed_subgraphs", threshold=1000):
    files = glob.glob(os.path.join(directory, "*.json"))
    
    if not files:
        print(f"Không tìm thấy file .json nào trong thư mục: {directory}")
        return

    outlier_files = [] # Lưu danh sách các file bị loại và lý do

    for filepath in files:
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                file_is_outlier = False
                reasons = [] # Ghi chú lại variant nào vi phạm
                
                for variant, content in data.items():
                    triples = content.get("triples", [])
                    num_edges = len(triples)
                    
                    unique_nodes = set()
                    for t in triples:
                        s = t.get("subject")
                        o = t.get("object")
                        if s: unique_nodes.add(s)
                        if o: unique_nodes.add(o)
                        
                    num_nodes = len(unique_nodes)
                    
                    # KIỂM TRA ĐIỀU KIỆN OUTLIER
                    if num_nodes > threshold or num_edges > threshold:
                        file_is_outlier = True
                        reasons.append(f"{variant} (Nodes: {num_nodes}, Edges: {num_edges})")
                
                # Nếu file có ít nhất 1 variant vượt ngưỡng -> Cho vào danh sách loại
                if file_is_outlier:
                    q_id = os.path.basename(filepath)
                    outlier_files.append({
                        "id": q_id,
                        "reasons": reasons
                    })
                    
            except Exception as e:
                print(f"Lỗi đọc file {filepath}: {e}")

    # --- IN KẾT QUẢ TỔNG HỢP ---
    total_files = len(files)
    total_outliers = len(outlier_files)
    percent_removed = (total_outliers / total_files) * 100 if total_files > 0 else 0
    
    print("=" * 60)
    print(" BÁO CÁO TỔNG HỢP (UNION OUTLIERS)".center(60))
    print("=" * 60)
    print(f"Ngưỡng an toàn (Threshold): > {threshold} Nodes hoặc Edges")
    print(f"Tổng số câu hỏi (files) gốc:   {total_files}")
    print(f"Tổng số câu cần LOẠI BỎ:        {total_outliers}")
    print(f"Tỉ lệ dữ liệu bị cắt giảm:     {percent_removed:.2f}%\n")
    
    # In chi tiết một vài file vi phạm để xem thử
    print("--- Danh sách các file bị loại ---")
    for item in outlier_files:
        print(f"- {item['id']}:")
        for reason in item['reasons']:
            print(f"    + {reason}")

if __name__ == "__main__":
    # Đặt ngưỡng 1000 theo yêu cầu của bạn
    find_outlier_questions("data/test_perturbed_subgraphs", threshold=1000)