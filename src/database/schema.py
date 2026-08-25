# ==============================================================================
# AIC 2026 - PYARROW SCHEMA DEFINITION FOR LANCEDB MULTIMODAL STORE
# ==============================================================================
import pyarrow as pa

def get_aic_master_schema(vector_dim: int = 768) -> pa.Schema:
    """
    Returns the standardized PyArrow Schema for the LanceDB master keyframe table.
    Contains all 12 core multimodal fields required for high-precision video retrieval.
    """
    return pa.schema([
        # 1. Tầng Vector Đặc Trưng
        pa.field("vector", pa.list_(pa.float32(), vector_dim)),  # CLIP ViT-L/14 embedding 768-dim (L2-Normalized)

        # 2. Tầng Định Danh & Ánh Xạ Khung Hình Chuẩn Xác
        pa.field("video_id", pa.string()),                        # Tên video: 'L21_V001'
        pa.field("frame_idx", pa.int32()),                        # Thứ tự index: 0, 1, 2... (tương ứng 001.jpg)
        pa.field("frame_id", pa.int64()),                         # FRAME ID THẬT CỦA VIDEO TỪ CSV: 0, 90, 261, 1234...
        pa.field("pts_time", pa.float32()),                       # Mốc thời gian thực tế (giây): 0.0, 3.0, 8.7...
        pa.field("image_path", pa.string()),                      # Đường dẫn tuyệt đối hoặc tương đối đến file ảnh .jpg

        # 3. Tầng Nhãn Vật Thể & OCR
        pa.field("detected_objects", pa.string()),                # Nhãn OpenImages (>=0.10): 'lantern, car, balloon...'
        pa.field("ocr_text", pa.string()),                        # Toàn bộ ký tự OCR trích xuất được trên frame

        # 4. Tầng Metadata & Văn Bản Đa Trường
        pa.field("video_title", pa.string()),                     # Tiêu đề từ media-info.json
        pa.field("video_description", pa.string()),               # Mô tả tóm tắt nội dung từ media-info.json
        pa.field("video_keywords", pa.string()),                  # Tags từ khóa từ media-info.json
        pa.field("all_text_weighted", pa.string())                # Văn bản tổng hợp đa trọng số phục vụ Tantivy BM25
    ])
