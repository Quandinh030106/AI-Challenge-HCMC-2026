import argparse
import os
import yaml
import numpy as np
from database_searcher import DatabaseSearcher

# Giả lập mô hình nạp Embeddings (Nếu chưa kết nối mô hình CLIP thực tế)
def get_dummy_embedding(dim=512):
    vec = np.random.rand(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)

def test_single_query(db_searcher, query_text, tier1_objs=None, top_k=10):
    print(f"\n📝 Câu query đầu vào: '{query_text}'")
    print(f"📦 Tier 1 Objects để Hard Filter: {tier1_objs if tier1_objs else 'Không áp dụng'}")
    
    # 1. Tạo Query Vector (Ở pipeline thực tế: lấy từ CLIP Model)
    query_vector = get_dummy_embedding(512)
    
    # 2. Truy vấn Database
    results = db_searcher.search_candidates(
        query_embedding=query_vector,
        tier1_objects=tier1_objs,
        top_k=top_k
    )
    
    # 3. Hiển thị kết quả
    print(f"📊 Kết quả Top {top_k} ứng viên lọc được từ Database:")
    if not results:
        print("⚠️ Không tìm thấy khung ảnh nào phù hợp với điều kiện Hard Filter.")
    else:
        for rank, (frame_id, video_id, score) in enumerate(results, 1):
            print(f"  [{rank:02d}] Video: {video_id:<12} | Frame ID: {frame_id:<20} | Similarity Score: {score:.4f}")

def run_test_from_file(db_searcher, file_path, top_k=10):
    if not os.path.exists(file_path):
        print(f"❌ Không tìm thấy file test tại đường dẫn: {file_path}")
        return
        
    print(f"\n📁 Đang đọc danh sách queries từ file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
        
    for idx, query in enumerate(lines, 1):
        print(f"\n--- [Test Case {idx}/{len(lines)}] ---")
        test_single_query(db_searcher, query_text=query, top_k=top_k)

def main():
    parser = argparse.ArgumentParser(description="Test Bench mở cho Database Searcher")
    parser.add_argument("--file", type=str, help="Đường dẫn tới file .txt chứa danh sách câu hỏi test")
    parser.add_argument("--query", type=str, help="Nhập trực tiếp 1 câu truy vấn text để test")
    parser.add_argument("--objects", type=str, nargs="+", help="Danh sách Tier 1 Objects lọc cứng (ví dụ: --objects hat car)")
    parser.add_argument("--top_k", type=int, default=5, help="Số lượng ứng viên muốn lấy ra")
    
    args = parser.parse_args()

    # Read config
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    db_searcher = DatabaseSearcher(config.get("database", {}))

    if args.file:
        run_test_from_file(db_searcher, args.file, top_k=args.top_k)
    elif args.query:
        test_single_query(db_searcher, query_text=args.query, tier1_objs=args.objects, top_k=args.top_k)
    else:
        print("�� Gợi ý sử dụng:")
        print(" 1. Test bằng câu hỏi trực tiếp:")
        print("    python src/test_db_flexible.py --query 'Người đội nón đỏ' --objects hat")
        print(" 2. Test bằng file danh sách câu hỏi:")
        print("    python src/test_db_flexible.py --file path/to/queries.txt --top_k 10")

    db_searcher.close()

if __name__ == "__main__":
    main()
