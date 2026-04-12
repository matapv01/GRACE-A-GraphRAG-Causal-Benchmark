# GraphRAG-Benchmark

A Universal Diagnostic Subgraph Framework for evaluating Knowledge Graph Reasoning and Faithfulness of Large Language Models (LLMs).

## 🚀 Hướng dẫn chạy Pipeline cào dữ liệu (Background / Ngầm)

Dự án sử dụng cơ chế chạy ngầm (`nohup`) để xử lý toàn bộ >24,000 dữ liệu của LC-QuAD 2.0 từ Wikidata mà không bị gián đoạn khi tắt terminal.
Code đã được tích hợp cơ chế tự động resume (chạy bù các file chưa có) và **retry vô hạn** cho lỗi rate-limit `429 Too Many Requests`.

### 1. Khởi động trích xuất đồ thị sạch (Extract Clean Subgraphs)
Lệnh này sẽ cào dữ liệu từ Wikidata, đối chiếu câu hỏi để tìm `Gold Path` thật, sau đó dùng `SemanticRetriever` lấy thêm dữ liệu nhiễu xung quanh thông qua Vector Embeddings GPU.

```bash
nohup uv run python scripts/extract_clean_subgraphs.py > logs_extract.txt 2>&1 &
```
* Tiến độ được ghi vào: `logs_extract.txt`
* Dữ liệu JSON sinh ra tại: `data/clean_subgraphs/`

### 2. Khởi động sinh đồ thị nhiễu (Generate Perturbations)
Lệnh này có thể chạy song song với lệnh 1. Nó liên tục quét thư mục clean ở trên, nếu thấy file mới sẽ tiến hành sinh ra bản JSON lồng 5 trạng thái Causal: `Clean`, `Broken`, `Type Matching`, `Topological`, `Swapping`.

```bash
nohup uv run python scripts/generate_perturbations.py > logs_perturb.txt 2>&1 &
```
* Tiến độ được ghi vào: `logs_perturb.txt`
* Dữ liệu JSON sinh ra tại: `data/perturbed_subgraphs/`

### 3. Theo dõi tiến độ & Thống kê lỗi (Statistics)
Công cụ này quét toàn bộ thư mục data và báo cáo chi tiết: tiến độ %, số lượng câu đã tải thành công, và tổng hợp nhóm các thể loại lỗi ở bước tạo Perturb (ví dụ: không có cạnh answer, mảng rỗng,...).

```bash
uv run python scripts/statistics.py
```

### 4. Cách dừng hệ thống khẩn cấp (Kill Process)
Nếu bạn lỡ chạy nhầm hoặc muốn sửa code dở dang, hãy copy lệnh này để kết liễu mọi tiến trình cào dữ liệu ngầm đang chạy:

```bash
pkill -f extract_clean_subgraphs.py
pkill -f generate_perturbations.py
```
