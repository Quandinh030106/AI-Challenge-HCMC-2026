def init_db_schema(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                video_id VARCHAR(64) PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                category VARCHAR(50),
                main_keywords JSONB DEFAULT '[]',
                video_path TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS keyframes (
                frame_id VARCHAR(128) PRIMARY KEY,
                video_id VARCHAR(64) NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
                frame_idx INT NOT NULL,
                clip_embedding vector(512),
                frame_prompt TEXT,
                objects_tier1 JSONB DEFAULT '[]',
                objects_tier2 JSONB DEFAULT '[]',
                objects_tier3 JSONB DEFAULT '[]',
                ocr_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_videos_category ON videos(category);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_videos_keywords ON videos USING gin(main_keywords);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_keyframes_clip ON keyframes USING hnsw (clip_embedding vector_cosine_ops);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_keyframes_objects ON keyframes USING gin(objects_tier1);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_keyframes_video_id ON keyframes(video_id);")
    conn.commit()
