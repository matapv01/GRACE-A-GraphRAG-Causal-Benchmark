# GraphRAG-Benchmark

A Universal Diagnostic Subgraph Framework for evaluating Knowledge Graph Reasoning and Faithfulness of Large Language Models (LLMs).

## 🚀 Hướng dẫn chạy Pipeline cào dữ liệu (Background / Ngầm)

Dự án sử dụng cơ chế chạy ngầm (`nohup`) để xử lý toàn bộ dữ liệu của LC-QuAD 2.0 Test set từ Wikidata mà không bị gián đoạn khi tắt terminal.
Code đã được tích hợp cơ chế tự động resume (chạy bù các file chưa có) và **retry vô hạn** cho lỗi rate-limit `429 Too Many Requests`.

### 1. Khởi động trích xuất đồ thị sạch (Extract Clean Subgraphs)
Lệnh này sẽ cào dữ liệu từ Wikidata, đối chiếu câu hỏi để tìm `Gold Path` thật, sau đó dùng `SemanticRetriever` lấy thêm dữ liệu nhiễu xung quanh thông qua Vector Embeddings GPU.

**Lệnh chạy:**
```bash
nohup env PYTHONPATH=src uv run python scripts/data_generation/extract_clean_subgraphs.py > logs/logs_extract.txt 2>&1 &
```
* Dữ liệu JSON sinh ra tại: `data/test_clean_subgraphs/`

**Chạy lại các mẫu bị lỗi mạng/API:**
Thêm cờ `--retry_errors` để hệ thống bỏ qua các file thành công, tự động quét và cào lại các câu hỏi từng bị lỗi (trừ lỗi logic như Empty Gold Path).
```bash
nohup env PYTHONPATH=src uv run python scripts/data_generation/extract_clean_subgraphs.py --retry_errors > logs/logs_extract_retry.txt 2>&1 &
```

### 2. Khởi động sinh đồ thị nhiễu (Generate Perturbations)
Lệnh này có thể chạy song song với lệnh 1. Nó liên tục quét thư mục clean ở trên, nếu thấy file mới sẽ tiến hành sinh ra bản JSON lồng 5 trạng thái Causal: `Clean`, `Broken`, `Type Matching`, `Topological`, `Swapping`.

**Lệnh chạy:**
```bash
nohup env PYTHONPATH=src uv run python scripts/data_generation/generate_perturbations.py > logs/logs_perturb.txt 2>&1 &
```
* Dữ liệu JSON sinh ra tại: `data/test_perturbed_subgraphs/`

**Sinh lại đồ thị cho các mẫu bị lỗi sinh nhiễu:**
Thêm cờ `--retry_errors` để hệ thống tự động ghi đè, xử lý lại cho các JSON trước đó sinh ra dính cảnh báo `"error"`.
```bash
nohup env PYTHONPATH=src uv run python scripts/data_generation/generate_perturbations.py --retry_errors > logs/logs_perturb_retry.txt 2>&1 &
```

### 3. Theo dõi tiến độ & Thống kê lỗi (Statistics)
Công cụ này quét toàn bộ thư mục data và báo cáo chi tiết: tiến độ %, số lượng câu đã tải thành công, và tổng hợp nhóm các thể loại lỗi ở bước tạo Perturb (ví dụ: không có cạnh answer, mảng rỗng,...).

**Lệnh chạy:**
```bash
PYTHONPATH=src env PYTHONPATH=src uv run python scripts/analysis/statistics.py
```

### 4. Xem trước mẫu dữ liệu (Preview Samples)
Lệnh này hiển thị chi tiết thông tin của các mẫu báo cáo/đồ thị đã được trích xuất (thực thể, quan hệ, nhãn từ Wikidata) giúp bạn kiểm tra nhanh dữ liệu đầu ra và tính chính xác của tiến trình.

```bash
PYTHONPATH=src env PYTHONPATH=src uv run python scripts/analysis/preview_samples.py
```

### 5. Trực quan hóa cấu trúc đồ thị (Visualize Mermaid)
Sử dụng script này để in ra cấu trúc dạng Markdown Mermaid phục vụ dán lên các trình duyệt vẽ sơ đồ, giúp biểu diễn trực quan đồ thị Knowledge Graph cho các trạng thái Causal khác nhau đã sinh ra phục vụ mục đích Debug.

Chạy ngẫu nhiên một mẫu bất kỳ:
```bash
PYTHONPATH=src env PYTHONPATH=src uv run python scripts/analysis/visualize_mermaid.py
```

Chạy vẽ biểu đồ cho một tập truy vấn subgraph cụ thể (`--mode`) hoặc truyền trực tiếp mã câu hỏi (`--single_id`):

Các giá trị `--mode` (loại đồ thị) tiêu biểu trong tập dữ liệu bạn có thể chọn:
- `"simple question right"` / `"simple question left"`: Truy vấn đơn giản 1 hop.
- `"right-subgraph"` / `"left-subgraph"`: Truy vấn đa bước (Multi-hop) đi tiếp hoặc đi lùi.
- `"center"`: Đồ thị hội tụ/phân mảnh từ trung tâm.
- `"statement_property"`: Đồ thị rẽ nhánh bằng các thuộc tính phụ (Qualifiers).
- `"two intentions"`: Câu hỏi kết hợp (Conjunction) nhiều điều kiện.
- `"boolean"`: Câu hỏi dạng Yes/No.

```bash
# Vẽ mẫu một câu multi-hop ngẫu nhiên
PYTHONPATH=src env PYTHONPATH=src uv run python scripts/analysis/visualize_mermaid.py --mode "right-subgraph"

# Vẽ trực tiếp câu hỏi cụ thể đã biết ID
PYTHONPATH=src env PYTHONPATH=src uv run python scripts/analysis/visualize_mermaid.py --single_id 10856
```

### 6. Cách dừng hệ thống khẩn cấp (Kill Process)
Nếu bạn lỡ chạy nhầm hoặc muốn sửa code dở dang, hãy copy lệnh này để kết liễu mọi tiến trình cào dữ liệu ngầm đang chạy:

```bash
pkill -f extract_clean_subgraphs.py
pkill -f generate_perturbations.py
```

## Data Fixes & Backups

- Problem: Some entries in `data/lcquad_test.json` may have a null `question` field. This causes `scripts/benchmark/lightrag/run_all_lightrag.py` to raise a TypeError when concatenating the question with options during benchmark runs.

- Fix script: Use the provided backfill utility to populate missing `question` values from `NNQT_question` where available:

```bash
python scripts/analysis/backfill_lcquad_questions.py
```

- What it does: Creates a backup `data/lcquad_test.json.bak` and replaces null `question` fields with `NNQT_question` when present.

- Restore original: If you want to restore the original file:

```bash
mv data/lcquad_test.json.bak data/lcquad_test.json
```

- Remove backup: When you're confident, you can delete the backup:

```bash
rm data/lcquad_test.json.bak
```

- Design note: The benchmark runner intentionally surfaces data issues (it does not silently fall back). Use the backfill script to repair dataset entries before large-scale runs.
