# ==============================================================================
# AIC 2026 - PYARROW SCHEMA DEFINITIONS FOR PRODUCTION WEB SERVICE 2-TABLE STORE
# ==============================================================================
import pyarrow as pa

def get_videos_schema() -> pa.Schema:
    """
    Returns the standardized PyArrow Schema for the Video Metadata table (`videos`).
    Contains video-level attributes, streaming video paths, and duration without redundancy.
    """
    return pa.schema([
        pa.field("video_id", pa.string()),           # Primary Key: 'L21_V001'
        pa.field("video_title", pa.string()),        # Video title from media-info.json
        pa.field("video_description", pa.string()),  # Summary description
        pa.field("video_keywords", pa.string()),     # Tags list
        pa.field("all_video_text", pa.string()),     # Consolidated text for Video-Level Search
        pa.field("video_path", pa.string()),         # Streaming MP4 video path or web URL
        pa.field("duration_seconds", pa.float32()),  # Total video duration in seconds
        pa.field("fps", pa.float32()),               # Video frames per second
        pa.field("total_keyframes", pa.int32())      # Total keyframes in video
    ])

def get_keyframes_schema(vector_dim: int = 512) -> pa.Schema:
    """
    Returns the standardized PyArrow Schema for the Keyframe Visual table (`keyframes`).
    Contains frame-level visual features, exact frame IDs, formatted timestamps, OCR, objects, and captions.
    """
    return pa.schema([
        # 1. Tầng Vector Đặc Trưng
        pa.field("vector", pa.list_(pa.float32(), vector_dim)),  # L2-Normalized CLIP embedding (512 or 768-dim)

        # 2. Tầng Định Danh & Ánh Xạ Khung Hình Chuẩn Xác
        pa.field("video_id", pa.string()),                        # Foreign Key: 'L21_V001'
        pa.field("frame_idx", pa.int32()),                        # 0-based sequence index: 0, 1, 2...
        pa.field("frame_id", pa.int64()),                         # Real CSV Frame ID: 0, 90, 261, 1234...
        pa.field("pts_time", pa.float32()),                       # Timestamp in seconds: 202.1
        pa.field("timestamp_formatted", pa.string()),             # Formatted timestamp: '03:22'
        pa.field("image_path", pa.string()),                      # Absolute path / URL to keyframe image

        # 3. Tầng Nhãn Thị Giác, OCR & Captions Độc Bản Khung Hình
        pa.field("keyframe_caption", pa.string()),                # Multi-sentence Qwen2.5-VL caption
        pa.field("detected_objects", pa.string()),                # OpenImages objects: 'person, hat, glasses...'
        pa.field("ocr_text", pa.string()),                        # Extracted OCR text / scale numbers
        pa.field("text_genre", pa.string()),                      # Layout genre ('GENERAL', 'MAP', 'SIGN', etc.)
        pa.field("frame_text_weighted", pa.string())              # Consolidated frame text for Tantivy FTS
    ])

