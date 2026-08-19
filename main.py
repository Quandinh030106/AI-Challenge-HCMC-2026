import argparse
import os
import json
import numpy as np
from src.utils import load_config, normalize_query_item
from src.search.dense_search import DenseSearcher
from src.search.sparse_search import SparseSearcher
from src.search.fusion import reciprocal_rank_fusion
from src.preprocessing.query_processor import QueryProcessor
from src.tasks.task1_kis import solve_task1, get_frame_id_from_idx, generate_diversity_top100_kis
from src.tasks.task2_vqa import solve_task2
from src.tasks.task3_trake import solve_task3
from src.evaluation.evaluator import Evaluator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    print("--- KHOI CHAY PIPELINE TIM KIEM VIDEO ---")
    
    # 1. Khoi tao cac module tim kiem & xu ly query
    dense_searcher = DenseSearcher(config)
    sparse_searcher = SparseSearcher(config)
    query_processor = QueryProcessor()
    
    keyframes_dir = config["data"]["keyframes_dir"]
    metadata_dir = config["data"]["metadata_dir"]
    
    # 2. Tim kiem file Ground Truth tap Validation cuc bo
    gt_path = os.path.join(metadata_dir, "local_val_gt.json")
    if not os.path.exists(gt_path):
        gt_path = "data/metadata/local_val_gt.json"
        
    if os.path.exists(gt_path):
        print(f"\nTim thay file Ground Truth tai: {gt_path}. Bat dau danh gia...")
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
            
        predictions_dict = {"task1": {}, "task2": {}, "task3": {}}
        
        # Task 1 (Textual KIS)
        if "task1" in gt_data:
            print("\nDang xu ly Task 1 (Textual KIS)...")
            task1_items = gt_data["task1"] if isinstance(gt_data["task1"], list) else gt_data["task1"].values()
            for raw_q in task1_items:
                norm_q = normalize_query_item(raw_q)
                query_id = norm_q["query_id"]
                query_text = norm_q["query"]
                
                q_info = query_processor.process(query_text)
                intent = q_info["intent_info"]
                
                dense_res = dense_searcher.search(q_info["prompt_ensemble"], top_k_videos=20)
                sparse_res = sparse_searcher.search(query_text, top_k_videos=20)
                fused = reciprocal_rank_fusion(
                    dense_res, sparse_res, 
                    dense_weight=intent["dense_weight"], 
                    sparse_weight=intent["sparse_weight"]
                )
                
                preds = generate_diversity_top100_kis(
                    fused, keyframes_dir, metadata_dir=metadata_dir, total_preds=100
                )
                predictions_dict["task1"][query_id] = preds
                
        # Task 2 (Visual Q&A)
        if "task2" in gt_data:
            print("\nDang xu ly Task 2 (Visual Q&A)...")
            task2_items = gt_data["task2"] if isinstance(gt_data["task2"], list) else gt_data["task2"].values()
            for raw_q in task2_items:
                norm_q = normalize_query_item(raw_q)
                query_id = norm_q["query_id"]
                query_text = norm_q["query"]
                question = norm_q["question"]
                
                q_info = query_processor.process(query_text)
                intent = q_info["intent_info"]
                
                dense_res = dense_searcher.search(q_info["prompt_ensemble"], top_k_videos=20)
                sparse_res = sparse_searcher.search(query_text, top_k_videos=20)
                fused = reciprocal_rank_fusion(
                    dense_res, sparse_res, 
                    dense_weight=intent["dense_weight"], 
                    sparse_weight=intent["sparse_weight"]
                )
                
                ans_res = solve_task2(query_text, question, fused, keyframes_dir, model_id=config["models"]["vlm_model"])
                
                preds = [{"video_id": ans_res["video_id"], "frame_id": ans_res["frame_id"], "answer": ans_res["answer"]}]
                for cand in fused[1:]:
                    preds.append({"video_id": cand["video_id"], "frame_id": "0000", "answer": "none"})
                predictions_dict["task2"][query_id] = preds[:100]
                
        # Task 3 (TRAKE)
        if "task3" in gt_data:
            print("\nDang xu ly Task 3 (TRAKE)...")
            task3_items = gt_data["task3"] if isinstance(gt_data["task3"], list) else gt_data["task3"].values()
            for raw_q in task3_items:
                norm_q = normalize_query_item(raw_q)
                query_id = norm_q["query_id"]
                query_text = norm_q["query"]
                events = norm_q["events"]
                
                q_info = query_processor.process(query_text)
                intent = q_info["intent_info"]
                
                dense_res = dense_searcher.search(q_info["prompt_ensemble"], top_k_videos=20)
                sparse_res = sparse_searcher.search(query_text, top_k_videos=20)
                fused = reciprocal_rank_fusion(
                    dense_res, sparse_res, 
                    dense_weight=intent["dense_weight"], 
                    sparse_weight=intent["sparse_weight"]
                )
                
                align_res = solve_task3(events, fused, keyframes_dir, dense_searcher)
                
                preds = [{"video_id": align_res["video_id"], "frame_ids": align_res["frame_ids"]}]
                for cand in fused[1:]:
                    preds.append({"video_id": cand["video_id"], "frame_ids": ["0000"] * len(events)})
                predictions_dict["task3"][query_id] = preds[:100]
                
        # 3. Tinh toan diem so
        evaluator = Evaluator(gt_data)
        metrics = evaluator.evaluate_all(predictions_dict)
        
        print("\n====================================")
        print("KET QUA DANH GIA CUC BO (LOCAL EVALUATION):")
        print(f" * Final Score: {metrics['Final_Score']:.4f}")
        print(f" * R@1        : {metrics['R@1']:.4f}")
        print(f" * R@5        : {metrics['R@5']:.4f}")
        print(f" * R@20       : {metrics['R@20']:.4f}")
        print(f" * R@50       : {metrics['R@50']:.4f}")
        print(f" * R@100      : {metrics['R@100']:.4f}")
        print("====================================")
        
    else:
        print(f"\nKhong tim thay file Ground Truth tai {gt_path}.")
        print("Chay thu 1 cau query mau de kiem tra:")
        test_query = "một diễn giả đang phát biểu trước máy quay"
        q_info = query_processor.process(test_query)
        dense_res = dense_searcher.search(q_info["prompt_ensemble"], top_k_videos=3)
        print(f"Ket qua tim kiem cho query '{test_query}' (English: '{q_info['query_en']}'):")
        for idx, res in enumerate(dense_res):
            print(f"Top {idx+1}: Video={res['video_id']}, Score={res['max_score']:.4f}, Best Frame Index={res['best_frame_idx']}")

if __name__ == "__main__":
    main()
