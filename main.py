import argparse
import os
import json
import numpy as np
from src.utils import load_config
from src.search.dense_search import DenseSearcher
from src.search.sparse_search import SparseSearcher
from src.search.fusion import reciprocal_rank_fusion
from src.preprocessing.query_processor import QueryProcessor
from src.tasks.task1_kis import solve_task1, get_frame_id_from_idx
from src.tasks.task2_vqa import solve_task2
from src.tasks.task3_trake import solve_task3
from src.evaluation.evaluator import Evaluator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    print("--- KÍCH HOẠT PIPELINE TÌM KIẾM VIDEO ---")
    
    # 1. Khởi tạo các module tìm kiếm & xử lý query
    dense_searcher = DenseSearcher(config)
    sparse_searcher = SparseSearcher(config)
    query_processor = QueryProcessor()
    
    keyframes_dir = config["data"]["keyframes_dir"]
    metadata_dir = config["data"]["metadata_dir"]
    
    # 2. Tìm kiếm file Ground Truth của tập Validation cục bộ
    gt_path = os.path.join(metadata_dir, "local_val_gt.json")
    if not os.path.exists(gt_path):
        gt_path = "data/metadata/local_val_gt.json"
        
    if os.path.exists(gt_path):
        print(f"\nTìm thấy file Ground Truth tại: {gt_path}. Bắt đầu chạy đánh giá...")
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
            
        predictions_dict = {"task1": {}, "task2": {}, "task3": {}}
        
        # --- CHẠY VÀ ĐÁNH GIÁ TASK 1 (KIS) ---
        if "task1" in gt_data:
            print("\nĐang xử lý Task 1 (Textual KIS)...")
            for q in gt_data["task1"]:
                query_id = q["query_id"]
                query_text = q["query"]
                
                # Tiền xử lý & Dịch thuật Query
                q_info = query_processor.process(query_text)
                query_en = q_info["query_en"]
                intent = q_info["intent_info"]
                
                # Thực hiện tìm kiếm kết hợp Dense (English) + Sparse (Vietnamese)
                dense_res = dense_searcher.search(query_en, top_k_videos=20)
                sparse_res = sparse_searcher.search(query_text, top_k_videos=20)
                fused = reciprocal_rank_fusion(
                    dense_res, sparse_res, 
                    dense_weight=intent["dense_weight"], 
                    sparse_weight=intent["sparse_weight"]
                )
                
                # Sinh danh sách 100 câu trả lời để Evaluator tính R@k
                preds = []
                for cand in fused:
                    video_id = cand["video_id"]
                    dense_info = cand["dense_info"]
                    if dense_info is not None and "all_scores" in dense_info:
                        scores = dense_info["all_scores"]
                        top_frame_idxs = np.argsort(scores)[::-1][:10]
                        for idx in top_frame_idxs:
                            fid = get_frame_id_from_idx(keyframes_dir, video_id, idx, metadata_dir=metadata_dir)
                            preds.append({"video_id": video_id, "frame_id": fid})
                    else:
                        preds.append({"video_id": video_id, "frame_id": "0000"})
                predictions_dict["task1"][query_id] = preds[:100]
                
        # --- CHẠY VÀ ĐÁNH GIÁ TASK 2 (Q&A) ---
        if "task2" in gt_data:
            print("\nĐang xử lý Task 2 (Visual Q&A)...")
            for q in gt_data["task2"]:
                query_id = q["query_id"]
                query_text = q["query"]
                question = q["question"]
                
                q_info = query_processor.process(query_text)
                query_en = q_info["query_en"]
                intent = q_info["intent_info"]
                
                dense_res = dense_searcher.search(query_en, top_k_videos=20)
                sparse_res = sparse_searcher.search(query_text, top_k_videos=20)
                fused = reciprocal_rank_fusion(
                    dense_res, sparse_res, 
                    dense_weight=intent["dense_weight"], 
                    sparse_weight=intent["sparse_weight"]
                )
                
                # Sinh câu trả lời bằng VLM Qwen2-VL
                ans_res = solve_task2(query_text, question, fused, keyframes_dir, model_id=config["models"]["vlm_model"])
                
                preds = [{"video_id": ans_res["video_id"], "frame_id": ans_res["frame_id"], "answer": ans_res["answer"]}]
                for cand in fused[1:]:
                    video_id = cand["video_id"]
                    preds.append({"video_id": video_id, "frame_id": "0000", "answer": "none"})
                predictions_dict["task2"][query_id] = preds[:100]
                
        # --- CHẠY VÀ ĐÁNH GIÁ TASK 3 (TRAKE) ---
        if "task3" in gt_data:
            print("\nĐang xử lý Task 3 (TRAKE)...")
            for q in gt_data["task3"]:
                query_id = q["query_id"]
                query_text = q["query"]
                
                events = [ev["name"] for ev in q["events"]]
                
                q_info = query_processor.process(query_text)
                query_en = q_info["query_en"]
                intent = q_info["intent_info"]
                
                dense_res = dense_searcher.search(query_en, top_k_videos=20)
                sparse_res = sparse_searcher.search(query_text, top_k_videos=20)
                fused = reciprocal_rank_fusion(
                    dense_res, sparse_res, 
                    dense_weight=intent["dense_weight"], 
                    sparse_weight=intent["sparse_weight"]
                )
                
                # Căn chỉnh chuỗi sự kiện bằng Dynamic Programming
                align_res = solve_task3(events, fused, keyframes_dir, dense_searcher)
                
                preds = [{"video_id": align_res["video_id"], "frame_ids": align_res["frame_ids"]}]
                for cand in fused[1:]:
                    video_id = cand["video_id"]
                    preds.append({"video_id": video_id, "frame_ids": ["0000"] * len(events)})
                predictions_dict["task3"][query_id] = preds[:100]
                
        # 3. Tính toán và hiển thị điểm số
        evaluator = Evaluator(gt_data)
        metrics = evaluator.evaluate_all(predictions_dict)
        
        print("\n====================================")
        print("📊 KẾT QUẢ ĐÁNH GIÁ CỤC BỘ (LOCAL EVALUATION):")
        print(f" * Điểm Final Score: {metrics['Final_Score']:.4f}")
        print(f" * R@1  : {metrics['R@1']:.4f}")
        print(f" * R@5  : {metrics['R@5']:.4f}")
        print(f" * R@20 : {metrics['R@20']:.4f}")
        print(f" * R@50 : {metrics['R@50']:.4f}")
        print(f" * R@100: {metrics['R@100']:.4f}")
        print("====================================")
        
    else:
        print(f"\nKhông tìm thấy file Ground Truth tại {gt_path}.")
        print("Chạy thử 1 câu query mẫu để kiểm tra:")
        test_query = "một diễn giả đang phát biểu trước máy quay"
        q_info = query_processor.process(test_query)
        dense_res = dense_searcher.search(q_info["query_en"], top_k_videos=3)
        print(f"Kết quả tìm kiếm cho query '{test_query}' (English: '{q_info['query_en']}'):")
        for idx, res in enumerate(dense_res):
            print(f"Top {idx+1}: Video={res['video_id']}, Score={res['max_score']:.4f}, Best Frame Index={res['best_frame_idx']}")

if __name__ == "__main__":
    main()
