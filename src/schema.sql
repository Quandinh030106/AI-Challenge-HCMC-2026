CREATE EXTENSION IF NOT EXISTS vector;

-- Bảng 1: Quản lý thông tin Video (Metadata)
CREATE TABLE IF NOT EXISTS videos (
    video_id VARCHAR(50) PRIMARY KEY,
    title TEXT,
    description TEXT,
    category VARCHAR(50), -- Tin tức, Nấu ăn, Thời sự,...
    main_keywords TEXT[], -- Mảng keyword chính
    video_path TEXT NOT NULL
);

-- Bảng 2: Quản lý Keyframes & Vector Embedding
CREATE TABLE IF NOT EXISTS keyframes (
    frame_id VARCHAR(100) PRIMARY KEY,
    video_id VARCHAR(50) REFERENCES videos(video_id) ON DELETE CASCADE,
    frame_idx INT NOT NULL,
    frame_path TEXT,
    embedding vector(512), -- Ma trận vector CLIP/Qwen (chỉnh lại số chiều nếu dùng model khác)
    prompt_description TEXT, -- 1 dòng prompt/description miêu tả chi tiết keyframe
    detected_objects TEXT[] -- Thông tin objects dùng để filter
);

-- Tạo Index để tăng tốc truy vấn Vector & Keyword
CREATE INDEX IF NOT EXISTS idx_keyframes_embedding ON keyframes USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_keyframes_video_id ON keyframes(video_id);