import os
import json
import yaml
import numpy as np
from psycopg2.extras import execute_values
from .config import get_connection
from .schema import init_db_schema

def run_indexing(config_path="configs/default.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    db_conf = config.get("database", {})
    conn = get_connection(db_conf)
    init_db_schema(conn)
    cur = conn.cursor()

    data_conf = config.get("data", {})
    raw_dir = data_conf.get("raw_dir", "")
    keyframes_dir = data_conf.get("keyframes_dir", "")
    features_dir = data_conf.get("features_dir", "")
    metadata_dir = data_conf.get("metadata_dir", "")

    # 1. Quét Metadata
    video_rows = {}
    if metadata_dir and os.path.exists(metadata_dir):
        for root, _, files in os.walk(metadata_dir):
            for file in files:
                if file.endswith(".json"):
                    vid = os.path.splitext(file)[0]
                    vpath = os.path.join(raw_dir, f"{vid}.mp4")
                    video_rows[str(vid)] = (str(vid), f"Video {vid}", "", "general", "[]", vpath)

    # 2. Quét Keyframes và tự động lấy video_id chuẩn từ folder cha
    keyframe_items = []
    if keyframes_dir and os.path.exists(keyframes_dir):
        for root, _, files in os.walk(keyframes_dir):
            for file in files:
                if file.endswith((".jpg", ".png", ".webp")):
                    # Lấy tên folder cha (ví dụ: L21_V008) làm video_id
                    vid = os.path.basename(root)
                    if not vid or vid == "keyframes":
                        parts = os.path.splitext(file)[0].split("_")
                        vid = "_".join(parts[:-1]) if len(parts) >= 2 else parts[0]

                    vid_str = str(vid)
                    if vid_str not in video_rows:
                        vpath = os.path.join(raw_dir, f"{vid_str}.mp4")
                        video_rows[vid_str] = (vid_str, f"Video {vid_str}", "", "general", "[]", vpath)
                    
                    frame_id = os.path.splitext(file)[0]
                    keyframe_items.append((frame_id, vid_str, file))

    # 3. Chèn tất cả video_id vào bảng videos trước (để không bao giờ bị lỗi Foreign Key)
    if video_rows:
        insert_vid_sql = """
        INSERT INTO videos (video_id, title, description, category, main_keywords, video_path)
        VALUES %s
        ON CONFLICT (video_id) DO NOTHING;
        """
        execute_values(cur, insert_vid_sql, list(video_rows.values()))
        conn.commit()
        print(f"✅ Đã nạp thành công {len(video_rows)} videos vào CSDL!")

    # 4. Chèn danh sách Keyframes
    keyframe_rows = []
    for frame_id, vid_str, file in keyframe_items:
        try:
            f_idx = int(os.path.splitext(file)[0].split("_")[-1])
        except Exception:
            f_idx = 0

        vec_str = str(np.zeros(512).tolist())
        keyframe_rows.append((
            frame_id, vid_str, f_idx, vec_str, 
            f"Keyframe {frame_id}", "[]", "[]", "[]", ""
        ))

        if len(keyframe_rows) >= 10000:
            _insert_keyframes_batch(cur, keyframe_rows)
            conn.commit()
            keyframe_rows = []

    if keyframe_rows:
        _insert_keyframes_batch(cur, keyframe_rows)
        conn.commit()
        print(f"✅ Đã nạp thành công toàn bộ keyframes!")

    cur.close()
    conn.close()

def _insert_keyframes_batch(cur, rows):
    sql = """
    INSERT INTO keyframes (
        frame_id, video_id, frame_idx, clip_embedding, 
        frame_prompt, objects_tier1, objects_tier2, objects_tier3, ocr_text
    ) VALUES %s
    ON CONFLICT (frame_id) DO NOTHING;
    """
    execute_values(cur, sql, rows)