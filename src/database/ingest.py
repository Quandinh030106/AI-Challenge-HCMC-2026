import os, json, yaml
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
    print("🚀 Bắt đầu nạp dữ liệu vào CSDL 2 bảng...")
    data_conf = config.get("data", {})
    metadata_dir = data_conf.get("metadata_dir", "")
    if metadata_dir and os.path.exists(metadata_dir):
        video_rows = []
        for root, _, files in os.walk(metadata_dir):
            for file in files:
                if file.endswith(".json"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as jf:
                            meta = json.load(jf)
                            vid = meta.get("video_id", os.path.splitext(file)[0])
                            title = meta.get("title", f"Video {vid}")
                            desc = meta.get("description", "")
                            cat = meta.get("category", "tin_tuc")
                            keywords = json.dumps(meta.get("keywords", []))
                            vpath = meta.get("video_path", f"videos/{vid}.mp4")
                            video_rows.append((vid, title, desc, cat, keywords, vpath))
                    except Exception:
                        continue
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
            execute_values(cur, insert_vid_sql, video_rows)
            conn.commit()
            print(f"✅ Đã nạp {len(video_rows)} video vào bảng 'videos'!")
    cur.close()
    conn.close()
    print("🎉 Hoàn tất quá trình Index!")

if __name__ == "__main__":
    run_indexing()
