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

    print("🚀 Bắt đầu nạp dữ liệu vào CSDL 2 bảng theo pipeline chuẩn của nhóm...")

    data_conf = config.get("data", {})
    raw_dir = data_conf.get("raw_dir", "")            # Dataset video
    keyframes_dir = data_conf.get("keyframes_dir", "")  # Dataset keyframes
    features_dir = data_conf.get("features_dir", "")   # Tập numpy feature chuẩn của nhóm
    metadata_dir = data_conf.get("metadata_dir", "")   # Dataset metadata media-info
    objects_dir = data_conf.get("objects_dir", "")     # Visual features/objects

    # ==========================================
    # 1. NẠP BẢNG 1: VIDEOS
    # ==========================================
    video_rows = {}
    print("📦 Quét dữ liệu Video & Metadata...")

    if metadata_dir and os.path.exists(metadata_dir):
        for root, _, files in os.walk(metadata_dir):
            for file in files:
                if file.endswith(".json"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as jf:
                            meta = json.load(jf)
                            vid = str(meta.get("video_id", os.path.splitext(file)[0]))
                            title = meta.get("title", f"Video {vid}")
                            desc = meta.get("description", "")
                            cat = meta.get("category", "tin_tuc")
                            keywords = json.dumps(meta.get("keywords", []))
                            vpath = os.path.join(raw_dir, f"{vid}.mp4")
                            video_rows[vid] = (vid, title, desc, cat, keywords, vpath)
                    except Exception:
                        continue

    if raw_dir and os.path.exists(raw_dir):
        for root, _, files in os.walk(raw_dir):
            for file in files:
                if file.endswith((".mp4", ".mkv", ".avi")):
                    vid = str(os.path.splitext(file)[0])
                    if vid not in video_rows:
                        vpath = os.path.join(root, file)
                        video_rows[vid] = (vid, f"Video {vid}", "No description", "tin_tuc", "[]", vpath)

    if keyframes_dir and os.path.exists(keyframes_dir):
        for root, _, files in os.walk(keyframes_dir):
            for file in files:
                if file.endswith((".jpg", ".png", ".webp")):
                    frame_name = os.path.splitext(file)[0]
                    parts = frame_name.split("_")
                    vid = "_".join(parts[:-1]) if len(parts) >= 2 else parts[0]
                    vid = str(vid)
                    if vid not in video_rows:
                        vpath = os.path.join(raw_dir, f"{vid}.mp4")
                        video_rows[vid] = (vid, f"Video {vid}", "Auto-created", "tin_tuc", "[]", vpath)

    if video_rows:
        insert_vid_sql = """
        INSERT INTO videos (video_id, title, description, category, main_keywords, video_path)
        VALUES %s
        ON CONFLICT (video_id) DO UPDATE SET
            title = EXCLUDED.title,
            description = EXCLUDED.description,
            category = EXCLUDED.category,
            main_keywords = EXCLUDED.main_keywords,
            video_path = EXCLUDED.video_path;
        """
        execute_values(cur, insert_vid_sql, list(video_rows.values()))
        conn.commit()
        print(f"✅ Đã nạp {len(video_rows)} video vào bảng 'videos'!")

    # ==========================================
    # 2. NẠP BẢNG 2: KEYFRAMES (Load Features chuẩn của nhóm)
    # ==========================================
    keyframe_rows = []
    print("📦 Quét và nạp dữ liệu Keyframes + CLIP Embeddings...")

    # Đọc file feature npy/npz nếu có trong features_dir của nhóm
    feat_dict = {}
    if features_dir and os.path.exists(features_dir):
        print(f"🔄 Đang đọc file feature từ: {features_dir}")
        for root, _, files in os.walk(features_dir):
            for file in files:
                if file.endswith((".npy", ".npz")):
                    try:
                        feat_path = os.path.join(root, file)
                        data = np.load(feat_path)
                        if isinstance(data, np.ndarray):
                            # Nếu file npy lưu mảng
                            fname = os.path.splitext(file)[0]
                            feat_dict[fname] = data
                        elif hasattr(data, 'files'):
                            # Nếu file npz chứa dict
                            for k in data.files:
                                feat_dict[k] = data[k]
                    except Exception:
                        continue

    if keyframes_dir and os.path.exists(keyframes_dir):
        for root, _, files in os.walk(keyframes_dir):
            for file in files:
                if file.endswith((".jpg", ".png", ".webp")):
                    frame_name = os.path.splitext(file)[0]
                    parts = frame_name.split("_")
                    if len(parts) >= 2:
                        vid = "_".join(parts[:-1])
                        try:
                            f_idx = int(parts[-1])
                        except ValueError:
                            f_idx = 0
                    else:
                        vid = parts[0]
                        f_idx = 0

                    # Lấy vector thật của nhóm nếu có, nếu chưa có mới dùng vector 0
                    if frame_name in feat_dict:
                        vec = feat_dict[frame_name].flatten().astype(np.float32)
                        vec_str = str(vec.tolist())
                    else:
                        vec_str = str(np.zeros(512).tolist())

                    frame_prompt = f"Keyframe {frame_name} of video {vid}"
                    objs_t1 = json.dumps([])
                    objs_t2 = json.dumps([])
                    objs_t3 = json.dumps([])
                    ocr = ""

                    keyframe_rows.append((
                        frame_name, str(vid), f_idx, vec_str, 
                        frame_prompt, objs_t1, objs_t2, objs_t3, ocr
                    ))

                    if len(keyframe_rows) >= 10000:
                        _insert_keyframes_batch(cur, keyframe_rows)
                        conn.commit()
                        keyframe_rows = []

        if keyframe_rows:
            _insert_keyframes_batch(cur, keyframe_rows)
            conn.commit()
            print("✅ Đã nạp hoàn tất toàn bộ Keyframes vào bảng 'keyframes'!")

    cur.close()
    conn.close()
    print("🎉 Hoàn tất nạp CSDL 2 bảng giữ nguyên cấu trúc nhóm!")

def _insert_keyframes_batch(cur, rows):
    sql = """
    INSERT INTO keyframes (
        frame_id, video_id, frame_idx, clip_embedding, 
        frame_prompt, objects_tier1, objects_tier2, objects_tier3, ocr_text
    ) VALUES %s
    ON CONFLICT (frame_id) DO UPDATE SET
        clip_embedding = EXCLUDED.clip_embedding,
        frame_prompt = EXCLUDED.frame_prompt,
        objects_tier1 = EXCLUDED.objects_tier1;
    """
    execute_values(cur, sql, rows)

if __name__ == "__main__":
    run_indexing()
