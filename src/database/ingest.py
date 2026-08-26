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

    print("🚀 Bắt đầu nạp dữ liệu chuẩn quy trình nhóm vào CSDL 2 bảng...")

    data_conf = config.get("data", {})
    raw_dir = data_conf.get("raw_dir", "")
    keyframes_dir = data_conf.get("keyframes_dir", "")
    features_dir = data_conf.get("features_dir", "")
    metadata_dir = data_conf.get("metadata_dir", "")
    objects_dir = data_conf.get("objects_dir", "")

    # ==========================================
    # 1. NẠP BẢNG VIDEOS (Giữ nguyên Metadata nhóm)
    # ==========================================
    video_rows = {}
    print("📦 [1/2] Đang đồng bộ thông tin Video & Metadata...")

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
                            cat = meta.get("category", "general")
                            keywords = json.dumps(meta.get("keywords", []))
                            vpath = os.path.join(raw_dir, f"{vid}.mp4")
                            video_rows[vid] = (vid, title, desc, cat, keywords, vpath)
                    except Exception:
                        continue

    # Đảm bảo tất cả video_id xuất hiện từ folder raw/keyframes đều được đăng ký vào bảng videos
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
                        video_rows[vid] = (vid, f"Video {vid}", "", "general", "[]", vpath)

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
    # 2. NẠP BẢNG KEYFRAMES (Load Features & Objects thực tế của nhóm)
    # ==========================================
    print("📦 [2/2] Đang nạp Feature Vectors & Keyframes của nhóm...")

    # Load toàn bộ numpy feature gốc của nhóm
    feat_dict = {}
    if features_dir and os.path.exists(features_dir):
        for root, _, files in os.walk(features_dir):
            for file in files:
                if file.endswith((".npy", ".npz")):
                    try:
                        feat_path = os.path.join(root, file)
                        data = np.load(feat_path)
                        if isinstance(data, np.ndarray):
                            fname = os.path.splitext(file)[0]
                            feat_dict[fname] = data
                        elif hasattr(data, 'files'):
                            for k in data.files:
                                feat_dict[k] = data[k]
                    except Exception:
                        continue

    keyframe_rows = []
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

                    # Sử dụng embedding gốc nếu có
                    if frame_name in feat_dict:
                        vec_str = str(feat_dict[frame_name].flatten().astype(np.float32).tolist())
                    else:
                        vec_str = str(np.zeros(512).tolist())

                    frame_prompt = f"Keyframe {frame_name}"
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
            print("✅ Đã nạp xong toàn bộ Keyframe Embeddings!")

    cur.close()
    conn.close()
    print("🎉 Hoàn tất tích hợp CSDL! Giữ nguyên 100% quy trình gốc của nhóm.")

def _insert_keyframes_batch(cur, rows):
    sql = """
    INSERT INTO keyframes (
        frame_id, video_id, frame_idx, clip_embedding, 
        frame_prompt, objects_tier1, objects_tier2, objects_tier3, ocr_text
    ) VALUES %s
    ON CONFLICT (frame_id) DO UPDATE SET
        clip_embedding = EXCLUDED.clip_embedding,
        frame_prompt = EXCLUDED.frame_prompt;
    """
    execute_values(cur, sql, rows)

if __name__ == "__main__":
    run_indexing()
