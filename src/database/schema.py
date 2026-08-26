# ==============================================================================
# AIC 2026 - PYARROW SCHEMA DEFINITIONS FOR NORMALIZED 2-TABLE LANCEDB STORE
# ==============================================================================
import pyarrow as pa

def get_videos_schema() -> pa.Schema:
    """
    Returns the standardized PyArrow Schema for the Video Metadata table (`videos`).
    Contains video-level attributes without redundancy (1 row per video).
    """
    return pa.schema([
        pa.field("video_id", pa.string()),           # Primary Key: 'L21_V001'
        pa.field("video_title", pa.string()),        # YouTube title from media-info.json
        pa.field("video_description", pa.string()),  # Summary description
        pa.field("video_keywords", pa.string()),     # 31 tags list
        pa.field("all_video_text", pa.string())      # Consolidated text for Video-Level Search
    ])

def get_keyframes_schema(vector_dim: int = 768) -> pa.Schema:
    """
    Returns the standardized PyArrow Schema for the Keyframe Visual table (`keyframes`).
    Contains frame-level visual features, exact frame IDs, OCR, objects, and captions.
    """
    return pa.schema([
        # 1. Tầng Vector Đặc Trưng
        pa.field("vector", pa.list_(pa.float32(), vector_dim)),  # CLIP ViT-L/14 embedding 768-dim (L2-Normalized)

        # 2. Tầng Định Danh & Ánh Xạ Khung Hình Chuẩn Xác
        pa.field("video_id", pa.string()),                        # Foreign Key: 'L21_V001'
        pa.field("frame_idx", pa.int32()),                        # 0-based sequence index: 0, 1, 2...
        pa.field("frame_id", pa.int64()),                         # FRAME ID THẬT TỪ CSV: 0, 90, 261, 1234...
        pa.field("pts_time", pa.float32()),                       # Mốc thời gian thực tế (giây): 0.0, 3.0, 8.7...
        pa.field("image_path", pa.string()),                      # Đường dẫn tuyệt đối đến file ảnh .jpg

        # 3. Tầng Nhãn Thị Giác, OCR & Captions Độc Bản Khung Hình
        pa.field("keyframe_caption", pa.string()),                # Caption tự động do VLM/Captioner sinh ra
        pa.field("detected_objects", pa.string()),                # Nhãn OpenImages (>=0.10): 'lantern, car, balloon...'
        pa.field("ocr_text", pa.string()),                        # Chữ/số OCR trích xuất trên khung hình
        pa.field("text_genre", pa.string()),                      # Thể loại bố cục ('POEM', 'MAP', 'TRAFFIC_SIGN', etc.)
        pa.field("frame_text_weighted", pa.string())              # Văn bản độc bản khung hình cho Tantivy FTS
    ])

