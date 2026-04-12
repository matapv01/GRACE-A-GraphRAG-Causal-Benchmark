
# 📋 Plan: Universal Diagnostic Subgraph Framework for KG-Reasoning

Bản kế hoạch này mô tả quy trình xây dựng một hệ thống đánh giá tính **Faithfulness** (Độ trung thực) và **Causal Reasoning** (Suy luận nhân quả) của các pipeline RAG dựa trên Đồ thị tri thức (Knowledge Graph).

## 🎯 Mục tiêu cốt lõi
1.  [cite_start]Vượt qua giới hạn của các Benchmark tĩnh (vốn coi Gold Path là duy nhất)[cite: 1, 2].
2.  [cite_start]Xác định chính xác LLM đang thực sự suy luận hay chỉ sử dụng các "Lối tắt" (Shortcuts) như học vẹt tham số hoặc thiên kiến cấu trúc[cite: 5, 20].
3.  [cite_start]Tạo ra một môi trường Sandbox cô lập để đánh giá công bằng mọi framework[cite: 23].

---

## 🏗 Giai đoạn 1: Sơ chế dữ liệu & Trích xuất "Sự thật" (Ground Truth)

Thay vì các task Verification đơn giản, tập trung vào **Entity Prediction** để theo dõi luồng suy luận.

* **Lựa chọn Dataset:** Sử dụng **WebQSP** hoặc **Mintaka** vì có sẵn các truy vấn SPARQL chuẩn xác trên Wikidata.
* **Trích xuất Gold Path:**
    * Thực thi SPARQL query để lấy ra danh sách các triples $(s, r, o)$ tạo nên đường đi đúng từ thực thể câu hỏi đến đáp án.
    * Đây là "Khung xương" (Backbone) cho mọi biến thể Subgraph sau này.

---

## 🔍 Giai đoạn 2: Xây dựng Independent Retrieval Module (The Exam Generator)

[cite_start]Xây dựng một module tìm kiếm độc lập để định nghĩa "Không gian suy luận" (Candidate Reasoning Space)[cite: 22, 23].

* **Cơ chế:** **Semantic-driven Beam Search**.
    * Sử dụng Sentence-Transformer (ví dụ: `all-MiniLM-L6-v2`) để tính toán độ tương đồng ngữ nghĩa giữa câu hỏi và các nhãn quan hệ trên Wikidata.
    * [cite_start]Thực hiện mở rộng $k$-hop quanh thực thể câu hỏi, chỉ giữ lại Top-$K$ quan hệ có điểm số cao nhất để tránh bùng nổ tổ hợp[cite: 27, 28].
* **Output:** Một **Reasoning Subgraph** bao gồm Gold Path và các đường đi tiềm năng xung quanh.

---

## ☣️ Giai đoạn 3: Adversarial Factory (Tự động hóa "Cài bẫy")

Tự động tạo ra 4 biến thể của Subgraph để thử thách hệ thống qua các can thiệp nhân quả (Causal Interventions).

1.  [cite_start]**Biến thể Broken (Xóa cạnh then chốt):** Tự động xóa một cạnh quyết định trên Gold Path[cite: 3, 4].
2.  **Biến thể Type-matching (Bẫy cùng kiểu):** * Truy vấn Wikidata lấy các thực thể cùng loại (`instance of`) với đáp án đúng.
    * Bơm chúng vào Subgraph với các quan hệ nhiễu.
3.  **Biến thể Topological (Bẫy ngôi sao):** Bơm các thực thể có độ phổ biến cao (High-degree nodes) để kiểm tra thiên kiến cấu trúc.
4.  **Biến thể Swapping (Tráo đổi thực thể):** Thay đổi thực thể đáp án đúng trong đồ thị thành một thực thể sai nhưng hợp lệ về logic (Kiểm tra xem LLM có ưu tiên Context hơn Pre-train không).

---

## 🛡 Giai đoạn 4: Sandbox Isolation (Cô lập Framework)

Đảm bảo các framework bị đánh giá không thể truy cập dữ liệu bên ngoài Subgraph được cung cấp.

* **Kỹ thuật:** **Mocking Local Triplestore**.
    * Sử dụng thư viện **RDFLib** để nạp từng biến thể Subgraph vào một kho chứa tri thức ảo (In-memory DB) cho mỗi câu hỏi.
* **Kết nối:** Cấu hình các framework (như GNN-RAG, ToG) trỏ API truy vấn vào Sandbox này thay vì Wikidata Endpoint thật.
    * Điều này ép buộc bước **Internal Retrieval** của framework phải đối mặt trực tiếp với các bẫy bạn đã cài cắm.

---

## 📊 Giai đoạn 5: Đánh giá & Chẩn đoán Nhân quả (Causal Diagnosis)

Sử dụng chỉ số **Causal Effect Score (CES)** để định lượng khả năng của hệ thống.

### 1. Công thức tính CES
$$CES = P_{norm}(Y \mid do(G_{clean})) - P_{norm}(Y \mid do(G_{perturbed}))$$
*Trong đó $P_{norm}$ là xác suất dự đoán đúng đã được chuẩn hóa qua hàm $\exp$ và trung bình cộng Logprobs.*

### 2. Bảng phân loại lỗi (Shortcuts Taxonomy)
Dựa trên phản ứng của hệ thống, Agent thực hiện phân loại:

| Triệu chứng | Kết luận lỗi (Shortcuts) |
| :--- | :--- |
| Hiệu năng không đổi khi xóa Gold Path ($CES \approx 0$). | **Parametric Memorization** (Học vẹt từ lúc pre-train). |
| Hệ thống chọn nhầm thực thể "ngôi sao" (Hub node). | **Topological Bias** (Thiên kiến độ phổ biến). |
| Hệ thống chọn nhầm thực thể cùng Type. | **Type-matching Shortcut** (Suy luận lười biếng). |
| Hiệu năng tụt sâu khi bị can thiệp ($CES$ cao). | **Faithful Reasoning** (Suy luận trung thực trên đồ thị). |

---

## 🛠 Hướng dẫn triển khai kỹ thuật (Instruction cho Agent)

> **Bước 1:** Khởi tạo `WikidataExtractor` để lấy Triples từ SPARQL của WebQSP.
> 
> **Bước 2:** Chạy `SemanticRetriever` để tạo Subgraph mở rộng (Top-K relations).
> 
> **Bước 3:** Chạy `PerturbationModule` để tạo ra 4 tệp `.nt` (Clean, Broken, Decoy, Swap).
> 
> **Bước 4:** Khởi động Framework mục tiêu trong môi trường `SandboxWrapper`, chuyển hướng mọi lệnh `execute_query` vào tệp `.nt` tương ứng.
> 
> **Bước 5:** Trích xuất Logprobs từ đầu ra của LLM, tính toán CES và xuất báo cáo chẩn đoán lỗi dưới dạng bảng.

---