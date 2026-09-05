import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import argparse
import json
import csv
import time
from tqdm import tqdm
from src.utils import load_config, normalize_query_item
from src.search.dense_search import DenseSearcher
from src.search.sparse_search import SparseSearcher
from src.search.fusion import reciprocal_rank_fusion
from src.search.sequence_search import rerank_sequence_aware_kis
from src.search.temporal_refiner import TemporalRefiner
from src.preprocessing.query_processor import QueryProcessor
from src.tasks.task1_kis import solve_task1, get_frame_id_from_idx, generate_diversity_top100_kis
from src.tasks.task2_vqa import (
    build_task2_top100_predictions,
    solve_task2,
)
from src.tasks.task3_trake import solve_task3


def export_submissions(input_file, config_path="configs/default.yaml", output_dir="submissions"):
    """
    Script tu dong doc de thi chinh thuc cua BTC, chay pipeline va xuat file nop bai dat chuan.
    Ho tro ca dinh dang JSON va CSV dau vao.
    """
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    config = load_config(config_path)
    
    print("--- BAT DAU TIEN TRINH XUAT FILE NOP BAI ---")
    print(f"File de thi dau vao : {input_file}")
    print(f"Thu muc luu ket qua : {output_dir}")
    
    # 1. Khoi tao cac module tim kiem
    dense_searcher = DenseSearcher(config)
    sparse_searcher = SparseSearcher(config)
    query_processor = QueryProcessor(config)
    temporal_refiner = TemporalRefiner(config, dense_searcher)
    
    keyframes_dir = config["data"].get("keyframes_dir")
    metadata_dir = config["data"].get("metadata_dir")
    map_keyframes_dir = config["data"].get("map_keyframes_dir", metadata_dir)
    
    # 2. Doc file de thi dau vao
    queries_data = {"task1": [], "task2": [], "task3": []}
    
    if input_file.endswith(".json"):
        with open(input_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            
        if isinstance(raw_data, dict):
            for t_name in ["task1", "task2", "task3"]:
                if t_name in raw_data:
                    items = raw_data[t_name]
                    if isinstance(items, list):
                        queries_data[t_name] = items
                    elif isinstance(items, dict):
                        queries_data[t_name] = list(items.values())
        elif isinstance(raw_data, list):
            # Neu BTC dua 1 danh sach chung, mac dinh phan loai vao task1 hoac theo truong task
            for item in raw_data:
                t_type = item.get("task", "task1").lower()
                if t_type in queries_data:
                    queries_data[t_type].append(item)
                else:
                    queries_data["task1"].append(item)
                    
    elif input_file.endswith(".csv"):
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t_type = row.get("task", "task1").lower()
                if t_type in queries_data:
                    queries_data[t_type].append(row)
                else:
                    queries_data["task1"].append(row)

    submission_json = {}
    
    # --- XU LY TASK 1 (Textual KIS) ---
    if queries_data["task1"]:
        print(f"\nDang xu ly {len(queries_data['task1'])} cau hoi Task 1 (KIS)...")
        t1_rows = []
        submission_json["task1"] = {}
        
        for raw_q in tqdm(queries_data["task1"], desc="Task 1"):
            norm_q = normalize_query_item(raw_q)
            query_id = norm_q["query_id"]
            query_text = norm_q["query"]
            
            q_info = query_processor.process(query_text)
            intent = q_info["intent_info"]
            
            dense_res = dense_searcher.search(q_info["semantic_views"], top_k_videos=30)
            sparse_res = sparse_searcher.search(query_text, top_k_videos=30)
            fused = reciprocal_rank_fusion(
                dense_res, sparse_res, 
                dense_weight=intent["dense_weight"], 
                sparse_weight=intent["sparse_weight"]
            )
            fused, _ = rerank_sequence_aware_kis(
                query_text=query_text,
                fused_candidates=fused,
                dense_searcher=dense_searcher,
                sparse_searcher=sparse_searcher,
                query_processor=query_processor,
                config=config,
                pre_object_candidates=fused,
                query_id=query_id,
            )
            
            coarse_top100 = generate_diversity_top100_kis(
                fused, keyframes_dir, metadata_dir=map_keyframes_dir, total_preds=100
            )
            top100_preds, _ = temporal_refiner.refine_kis_predictions(
                query_id=query_id,
                query_text=query_text,
                prompt_ensemble=q_info["prompt_ensemble"],
                coarse_predictions=coarse_top100,
                fused_candidates=fused,
                query_processor=query_processor,
            )

            submission_json["task1"][query_id] = top100_preds
            
            # Format CSV: query_id, rank, video_id, frame_id
            for rank, pred in enumerate(top100_preds):
                t1_rows.append({
                    "query_id": query_id,
                    "rank": rank + 1,
                    "video_id": pred["video_id"],
                    "frame_id": pred["frame_id"]
                })
                
        # Xuat CSV Task 1
        t1_csv_path = os.path.join(output_dir, "submission_task1.csv")
        with open(t1_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["query_id", "rank", "video_id", "frame_id"])
            writer.writeheader()
            writer.writerows(t1_rows)
        print(f"-> Da xuat: {t1_csv_path}")

    # --- XU LY TASK 2 (Visual Q&A) ---
    if queries_data["task2"]:
        print(f"\nDang xu ly {len(queries_data['task2'])} cau hoi Task 2 (Q&A)...")
        t2_rows = []
        submission_json["task2"] = {}
        
        for raw_q in tqdm(queries_data["task2"], desc="Task 2"):
            norm_q = normalize_query_item(raw_q)
            query_id = norm_q["query_id"]
            query_text = norm_q["query"]
            question = norm_q["question"]
            
            q_info = query_processor.process(query_text)
            intent = q_info["intent_info"]
            
            dense_res = dense_searcher.search(q_info["semantic_views"], top_k_videos=30)
            sparse_res = sparse_searcher.search(query_text, top_k_videos=30)
            fused = reciprocal_rank_fusion(
                dense_res, sparse_res, 
                dense_weight=intent["dense_weight"], 
                sparse_weight=intent["sparse_weight"]
            )
            
            ans_res = solve_task2(
                query_text,
                question,
                fused,
                keyframes_dir,
                model_id=config["models"]["vlm_model"],
                metadata_dir=map_keyframes_dir,
                ocr_dir=config["data"].get("metadata_dir"),
                qa_config=config.get("search", {}).get("qa_evidence", {}),
                temporal_refiner=temporal_refiner,
                query_processor=query_processor,
                query_id=query_id,
            )

            promoted_idx = int(ans_res.get("promoted_idx", 0) or 0)
            if 0 < promoted_idx < len(fused):
                promoted_candidate = fused.pop(promoted_idx)
                fused.insert(0, promoted_candidate)

            top100_preds = build_task2_top100_predictions(
                fused_candidates=fused,
                answer_result=ans_res,
                keyframes_dir=keyframes_dir,
                metadata_dir=map_keyframes_dir,
                total_preds=100,
                qa_config=config.get("search", {}).get("qa_evidence", {}),
            )
            submission_json["task2"][query_id] = top100_preds
            
            for rank, pred in enumerate(top100_preds):
                t2_rows.append({
                    "query_id": query_id,
                    "rank": rank + 1,
                    "video_id": pred["video_id"],
                    "frame_id": pred["frame_id"],
                    "answer": pred["answer"]
                })
                
        t2_csv_path = os.path.join(output_dir, "submission_task2.csv")
        with open(t2_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["query_id", "rank", "video_id", "frame_id", "answer"])
            writer.writeheader()
            writer.writerows(t2_rows)
        print(f"-> Da xuat: {t2_csv_path}")

    # --- XU LY TASK 3 (TRAKE) ---
    if queries_data["task3"]:
        print(f"\nDang xu ly {len(queries_data['task3'])} cau hoi Task 3 (TRAKE)...")
        t3_rows = []
        submission_json["task3"] = {}
        
        for raw_q in tqdm(queries_data["task3"], desc="Task 3"):
            norm_q = normalize_query_item(raw_q)
            query_id = norm_q["query_id"]
            query_text = norm_q["query"]
            events = norm_q["events"]
            
            q_info = query_processor.process(query_text)
            intent = q_info["intent_info"]
            
            dense_res = dense_searcher.search(q_info["semantic_views"], top_k_videos=30)
            sparse_res = sparse_searcher.search(query_text, top_k_videos=30)
            fused = reciprocal_rank_fusion(
                dense_res, sparse_res, 
                dense_weight=intent["dense_weight"], 
                sparse_weight=intent["sparse_weight"]
            )
            
            align_res = solve_task3(
                events, fused, keyframes_dir, dense_searcher,
                metadata_dir=map_keyframes_dir, query_processor=query_processor,
                config=config, temporal_refiner=temporal_refiner, query_id=query_id,
            )

            top100_preds = []

            if (
                align_res.get("video_id") not in {None, "none"}
                and len(align_res.get("frame_ids", [])) == len(events)
            ):
                top100_preds.append({
                    "video_id": align_res["video_id"],
                    "frame_ids": align_res["frame_ids"],
                })
            submission_json["task3"][query_id] = top100_preds
            
            for rank, pred in enumerate(top100_preds):
                t3_rows.append({
                    "query_id": query_id,
                    "rank": rank + 1,
                    "video_id": pred["video_id"],
                    "frame_ids": ";".join(pred["frame_ids"])
                })
                
        t3_csv_path = os.path.join(output_dir, "submission_task3.csv")
        with open(t3_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["query_id", "rank", "video_id", "frame_ids"])
            writer.writeheader()
            writer.writerows(t3_rows)
        print(f"-> Da xuat: {t3_csv_path}")

    # 3. Xuat file tong hop JSON
    json_path = os.path.join(output_dir, "submission.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(submission_json, f, ensure_ascii=False, indent=2)
    print(f"-> Da xuat: {json_path}")
    
    elapsed = time.time() - start_time
    print(f"\n--- HOAN THANH XUAT FILE NOP BAI TRONG {elapsed:.2f} GIAY ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Duong dan file de thi cua BTC (.json hoac .csv)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output_dir", default="submissions")
    args = parser.parse_args()
    
    export_submissions(args.input, args.config, args.output_dir)
