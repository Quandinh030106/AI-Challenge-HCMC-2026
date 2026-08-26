import os
import json
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
import yaml

def run_indexing(config_path="configs/default.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    db_conf = config.get("database", {})
    conn = psycopg2.connect(**db_conf)
    cur = conn.cursor()

    print("🚀 Đang khởi tạo bảng và nạp dữ liệu vào Database...")

    cur.execute("""
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS keyframe_features (
        frame_id VARCHAR(128) PRIMARY KEY,
        video_id VARCHAR(64) NOT NULL,
        frame_idx INT NOT NULL,
        clip_embedding vector(512), 
        objects_tier1 JSONB DEFAULT '[]',
        objects_tier2 JSONB DEFAULT '[]',
        objects_tier3 JSONB DEFAULT '[]',
        ocr_text TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_clip_embedding 
    ON keyframe_features USING hnsw (clip_embedding vector_cosine_ops);

    CREATE INDEX IF NOT EXISTS idx_objects_t1 
    ON keyframe_features USING gin(objects_tier1);
    """)
    conn.commit()

    print("✅ Hoàn tất thiết lập bảng & Index!")
    cur.close()
    conn.close()

if __name__ == "__main__":
    run_indexing()
