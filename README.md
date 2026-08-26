# AI Challenge HCMC 2026 - Multimodal Video Retrieval System
## Branch: `ket-hop-database` (Accuracy-First Hybrid LanceDB Architecture)

Hệ thống truy vấn video đa phương thức hiệu năng cao, tối ưu hóa độ chính xác và quản lý tài nguyên độc lập trên môi trường máy chủ **Kaggle 2x Tesla T4 GPU (15GB + 15GB VRAM)**.

---

### I. TỔNG QUAN VÀ CHIẾN LƯỢC TỐI THƯỢNG (ACCURACY-FIRST)

Hệ thống được thiết kế chuyên biệt để giải quyết 3 dạng truy vấn chính thức của cuộc thi:
1. **Textual Known Item Search (KIS)**: Định vị chính xác video và frame tương ứng với mô tả văn bản tự nhiên.
2. **Visual Question Answering (Q&A)**: Tìm video sự kiện và trích xuất câu trả lời thị giác (số cân, biển báo, tên đèo, số lượng).
3. **Temporal Retrieval and Alignment of Key Events (TRAKE)**: Tìm kiếm video và căn chỉnh chuỗi sự kiện thời gian theo thứ tự tăng dần nghiêm ngặt ($t_1 < t_2 < \dots < t_N$).

#### Mục tiêu tối ưu hóa điểm số:
Thang điểm của Ban Giám Khảo đánh giá theo công thức:
$$Final\ Score = \frac{1}{5} \sum_{k \in \{1, 5, 20, 50, 100\}} R@k$$
Hệ thống áp dụng chiến lược **Đẩy kết quả đúng lên TOP 1 ($R@1 = 1.0$)** thông qua cơ chế thẩm định trực quan sâu bằng VLM, nhằm giành trọn vẹn điểm số **$Final\ Score = 1.00$**.

---

### II. KIẾN TRÚC HỆ THỐNG 4 GIAI ĐOẠN

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              SƠ ĐỒ HOẠT ĐỘNG TOÀN DIỆN CỦA HỆ THỐNG                                    │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘

 [GIAI ĐOẠN 0: KHỞI TẠO LANCEDB MULTIMODAL STORE] (Chạy 1 lần duy nhất ~2–3 phút)
  • Nạp ~200,000 vector CLIP ViT-L/14 (768 chiều, L2-Normalized) từ clip-features-32/*.npy.
  • Đọc map-keyframes/*.csv -> Ánh xạ vĩnh viễn frame_idx sang Frame ID thật của video.
  • Nạp media-info/*.json (Title, Description, Keywords) + objects/*.json (OpenImages score >= 0.10).
  • Tạo chỉ mục: Flat Vector Index + Tantivy BM25 Full-Text Search đa trường.
                                    │
                                    ▼
 [GIAI ĐOẠN 1: MULTI-ASPECT NLP QUERY PARSING] (Qwen2.5-7B 4-bit trên GPU 0)
  • Phân rã câu hỏi thành 4 Prompts thị giác độc lập (Bối cảnh, Nhân vật, Đồ vật/Màu sắc, Hành động).
  • Trích xuất Cụm danh từ tiếng Việt có nghĩa (2–4 từ) cho BM25 (loại bỏ 100% stop words).
  • Trích xuất nhãn thực thể OpenImages & Câu hỏi thị giác (VQA).
  • [TỰ ĐỘNG UNLOAD GPU 0 VỀ 0 MB VRAM].
                                    │
                                    ▼
 [GIAI ĐOẠN 2: LANCEDB NATIVE HYBRID RETRIEVAL & GAUSSIAN SMOOTHING] (CLIP ViT-L trên GPU 0)
  • Max-Sim Multi-Vector Retrieval: Tìm kiếm song song 4 Prompts thị giác qua LanceDB.
  • Tantivy BM25 Đa Trường: Title (x3.0) + Keywords (x2.0) + OCR (x2.0) + Objects (x1.5).
  • Lọc cộng hưởng thời gian Gauss 1D (Gaussian Temporal Smoothing): Lọc bỏ nhiễu frame ngẫu nhiên.
  • Trích xuất Top 10 Video ứng viên sáng giá nhất + Danh sách Top 100 Candidates.
  • [TỰ ĐỘNG UNLOAD GPU 0 VỀ 0 MB VRAM].
                                    │
                                    ▼
 [GIAI ĐOẠN 3: HEAVY VLM DEEP VISUAL VERIFICATION & TOP-1 PROMOTION] (Qwen2.5-VL-7B trên GPU 1)
  • Textual KIS: VLM trực tiếp xem ảnh Top 5 video -> Thẩm định độ khớp chi tiết -> ĐẨY LÊN TOP 1.
  • Visual Q&A: Lấy 3 frame thumbnail ($448 \times 448$) -> Đọc số cân / biển báo -> ĐẨY LÊN TOP 1.
  • TRAKE: Thuật toán Viterbi DP căn chỉnh chuỗi sự kiện strictly monotonic t_1 < t_2 < ... < t_N.
  • [TỰ ĐỘNG UNLOAD GPU 1 VỀ 0 MB VRAM].
                                    │
                                    ▼
 [GIAI ĐOẠN 4: SUBMISSION VALIDATION & CODABENCH PACKAGING]
  • Kiểm tra 100% số dòng (đúng 100 dòng/câu), đúng số lượng mốc TRAKE, escape chuỗi Q&A đúng chuẩn.
  • Đóng gói tự động file submission.zip đạt chuẩn nộp bài.
```

---

### III. CẤU TRÚC BẢNG DỮ LIỆU LANCEDB (12 TRƯỜNG DỮ LIỆU)

Toàn bộ dữ liệu của 873 video (~200,000 frames) được lưu trữ theo định dạng nhị phân Apache Arrow trong thư mục `data/aic_lancedb`:

| Trường Dữ Liệu | Kiểu Dữ Liệu | Mô Tả Kỹ Thuật |
| :--- | :--- | :--- |
| `vector` | `Float32[768]` | Vector đặc trưng CLIP ViT-L/14 chuẩn hóa L2 ($\|\mathbf{v}\| = 1$) |
| `video_id` | `String` | Tên video (ví dụ: `L21_V001`) |
| `frame_idx` | `Int32` | Thứ tự index trích xuất: `0, 1, 2...` (tương ứng `001.jpg`) |
| `frame_id` | `Int64` | **FRAME ID THẬT CỦA VIDEO TỪ CSV** (ví dụ: `0, 90, 261, 1234...`) |
| `pts_time` | `Float32` | Mốc thời gian thực tế (giây): `0.0, 3.0, 8.7...` |
| `image_path` | `String` | Đường dẫn tuyệt đối/tương đối đến file ảnh `.jpg` |
| `detected_objects` | `String` | Nhãn OpenImages ($\ge 0.10$): `lantern, car, balloon, boat...` |
| `ocr_text` | `String` | Chữ/số trích xuất từ khung hình |
| `video_title` | `String` | Tiêu đề YouTube từ `media-info.json` |
| `video_description` | `String` | Mô tả tóm tắt nội dung video |
| `video_keywords` | `String` | Danh sách 31 tags từ khóa |
| `all_text_weighted` | `String` | Chuỗi văn bản đa trọng số phục vụ tìm kiếm Tantivy BM25 |

---

### IV. CƠ CHẾ PHÂN BỔ TÀI NGUYÊN TRÊN DUAL T4 GPU (0% CUDA OOM)

- **GPU 0 (`cuda:0`, 15GB VRAM)**:
  - Chạy Giai đoạn 1 (NLP LLM Qwen2.5-7B, ~5.2 GB) $\rightarrow$ Unload sạch về 0 MB.
  - Chạy Giai đoạn 2 (CLIP ViT-L/14, ~1.8 GB) $\rightarrow$ Unload sạch về 0 MB.
- **GPU 1 (`cuda:1`, 15GB VRAM)**:
  - Dành trọn vẹn 15GB VRAM cho Giai đoạn 3 (Qwen2.5-VL-7B 4-bit NF4, ~5.5 GB).
  - Tối ưu hóa kích thước ảnh thumbnail ($448 \times 448$) và giới hạn token `max_pixels = 384 * 28 * 28` $\rightarrow$ Bộ nhớ tính toán activation luôn dưới 100 MB, triệt tiêu 100% nguy cơ tràn VRAM.

---

### V. CẤU TRÚC THƯ MỤC MÃ NGUỒN

```
AI-Challenge-HCMC-2026/
├── configs/
│   └── lancedb_config.yaml          <-- File cấu hình tập trung quản lý toàn bộ hệ thống
│
├── src/
│   ├── database/
│   │   ├── schema.py                <-- Định nghĩa PyArrow Schema chuẩn 12 trường dữ liệu
│   │   ├── ingest_pipeline.py       <-- Script tự động quét thô và đóng gói vào LanceDB
│   │   └── lancedb_manager.py       <-- Interface truy vấn Vector, BM25, lọc và trích xuất ảnh
│   │
│   ├── preprocessing/
│   │   ├── llm_query_parser.py      <-- Phân rã 4 khía cạnh thị giác và cụm từ BM25 (Qwen2.5-7B)
│   │   ├── extract_features.py      <-- Trích xuất đặc trưng CLIP (nếu cần tạo thêm .npy)
│   │   └── run_ocr.py               <-- Module EasyOCR trích xuất chữ
│   │
│   ├── search/
│   │   ├── lancedb_hybrid_search.py <-- Bộ tìm kiếm kết hợp Max-Sim Vector + Tantivy BM25 + Objects
│   │   └── temporal_smoother.py     <-- Thuật toán 1D Gaussian Temporal Smoothing
│   │
│   ├── tasks/
│   │   ├── task_kis.py              <-- Solver KIS tích hợp VLM Deep Visual Verification
│   │   ├── task_vqa.py              <-- Solver Q&A kèm chiến thuật đẩy lên Top 1 (R@1 = 1.0)
│   │   └── task_trake.py            <-- Solver TRAKE căn chỉnh Viterbi DP strictly monotonic
│   │
│   ├── service/
│   │   └── api.py                   <-- FastAPI Web Service cung cấp REST API cho toàn hệ thống
│   │
│   ├── export_submission.py         <-- Điều phối toàn bộ 4 giai đoạn & đóng gói submission.zip
│   └── utils.py                     <-- Các hàm tiện ích hỗ trợ đọc ghi file
│
└── requirements.txt                 <-- Khai báo đầy đủ các thư viện
```

---

### VI. HƯỚNG DẪN CHẠY TRÊN KAGGLE NOTEBOOK (4 CELLS TỰ ĐỘNG HÓA)

#### CELL 1: Cài đặt thư viện & Kéo mã nguồn
```python
import os
import torch

print("GPU khả dụng:", torch.cuda.device_count(), "-", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

!rm -rf AI-Challenge-HCMC-2026
!git clone -b ket-hop-database https://github.com/Quandinh030106/AI-Challenge-HCMC-2026.git

os.chdir("/kaggle/working/AI-Challenge-HCMC-2026")
!pip install -q lancedb pyarrow tantivy-py qwen-vl-utils rank-bm25 pyyaml scipy transformers accelerate bitsandbytes fastapi uvicorn
```

#### CELL 2: Khởi động lại Kernel (Nếu cần dọn sạch VRAM về 0 MB)
```python
import os
os._exit(0)
```

#### CELL 3: Cấu hình đường dẫn & Xây dựng LanceDB (Chạy 1 lần duy nhất ~2–3 phút)
```python
import os
import yaml
from src.database.ingest_pipeline import MultimodalIngestPipeline

os.chdir("/kaggle/working/AI-Challenge-HCMC-2026")

with open("configs/lancedb_config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Tự động quét và build LanceDB nếu chưa có
ingest_pipeline = MultimodalIngestPipeline(config)
ingest_pipeline.build_database(overwrite=False)

# Nén lưu trữ dự phòng
!zip -q -r /kaggle/working/aic_lancedb.zip /kaggle/working/aic_lancedb
print("[INFO] LanceDB Database is ready!")
```

#### CELL 4: Chạy toàn bộ Master Pipeline & Xuất Submission
```python
import os
from src.export_submission import run_master_pipeline

os.chdir("/kaggle/working/AI-Challenge-HCMC-2026")
run_master_pipeline(config_path="configs/lancedb_config.yaml", output_zip="/kaggle/working/submission.zip")
```

---

### VII. CHẠY FASTAPI WEB SERVICE (TRUY VẤN TỪ XA / LOCAL)

Để khởi động máy chủ API tìm kiếm:
```bash
python -m src.service.api
```
Truy cập Swagger UI tại: `http://localhost:8000/docs` để thử nghiệm trực quan các API tìm kiếm đa phương thức.
