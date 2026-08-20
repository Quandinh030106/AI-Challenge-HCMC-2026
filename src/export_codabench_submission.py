import os
import sys

# Dam bao thu muc goc cua du an luon nam trong sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import glob
import json
import zipfile
import argparse
import time
import re
import numpy as np
import yaml
from tqdm import tqdm

from src.preprocessing.query_processor import QueryProcessor

from src.search.dense_search import DenseSearcher
from src.search.sparse_search import SparseSearcher
from src.search.fusion import reciprocal_rank_fusion
from src.tasks.task1_kis import get_frame_id_from_idx, generate_diversity_top100_kis, gaussian_smooth_scores
from src.tasks.task2_vqa import solve_task2
from src.tasks.task3_trake import solve_task3, align_events_dynamic_programming
from src.search.object_search import ObjectSearcher
from src.search.visual_reranker import VisualReRanker

def load_config(config_path="configs/default.yaml"):
    """Doc file cau hinh he thong."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def parse_query_file(file_path):
    """Phan tich noi dung file cau hoi (.txt) thanh cau truc du lieu chuan."""
    filename = os.path.basename(file_path)
    query_id = os.path.splitext(filename)[0]
    
    with open(file_path, "r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f.readlines() if line.strip()]
        
    full_content = "\n".join(raw_lines)
    
    # 1. Kiem tra Task 2: Visual Q&A
    has_qa_flag = any(k in full_content.lower() for k in ["câu hỏi:", "cau hoi:", "q&a", "question:"])
    if has_qa_flag:
        visual_lines = []
        question_lines = []
        is_question = False
        for line in raw_lines:
            lower = line.lower()
            if any(k in lower for k in ["câu hỏi:", "cau hoi:", "question:"]):
                is_question = True
                cleaned = re.sub(r'^(câu hỏi|cau hoi|question)\s*:\s*', '', line, flags=re.IGNORECASE).strip()
                if cleaned:
                    question_lines.append(cleaned)
            elif is_question:
                question_lines.append(line)
            else:
                visual_lines.append(line)
                
        return {
            "query_id": query_id,
            "task_type": "qa",
            "query": " ".join(visual_lines).strip() if visual_lines else " ".join(question_lines).strip(),
            "question": " ".join(question_lines).strip()
        }
        
    # 2. Kiem tra Task 3: TRAKE
    has_trake_flag = any(re.match(r'^(sự kiện|su kien|event|bước|buoc|e\d+)\s*\d*\s*[:\.]', line, re.IGNORECASE) for line in raw_lines)
    if has_trake_flag or len(raw_lines) >= 3:
        events = []
        for line in raw_lines:
            cleaned = re.sub(r'^(sự kiện|su kien|event|bước|buoc|e\d+)\s*\d*\s*[:\.]\s*', '', line, flags=re.IGNORECASE).strip()
            if cleaned:
                events.append(cleaned)
        if len(events) >= 2:
            return {
                "query_id": query_id,
                "task_type": "trake",
                "events": events,
                "query": " ".join(events)
            }

    # 3. Mac dinh la Task 1: Textual KIS
    return {
        "query_id": query_id,
        "task_type": "kis",
        "query": " ".join(raw_lines).strip()
    }

def format_answer_for_csv(ans_text):
    """Format cau tra loi VQA cho file CSV."""
    if not ans_text:
        return '""'
    ans_cleaned = str(ans_text).strip().strip('"').strip("'")
    ans_escaped = ans_cleaned.replace('"', '""')
    return f'"{ans_escaped}"'

def run_codabench_pipeline(input_dir, config_path="configs/default.yaml", output_zip="submission.zip"):
    """Chay toan bo pipeline tren bo de thi va tao file submission.zip."""
    start_time = time.time()
    config = load_config(config_path)
    
    submission_dir = "submission"
    os.makedirs(submission_dir, exist_ok=True)
    
    print("=====================================================")
    print("KHOI CHAY HE THONG TAO SUBMISSION AIC 2026")
    print(f"Thu muc de thi : {input_dir}")
    print(f"File zip xuat  : {output_zip}")
    print("=====================================================")
    
    dense_searcher = DenseSearcher(config)
    sparse_searcher = SparseSearcher(config)
    query_processor = QueryProcessor()
    object_searcher = ObjectSearcher(config)
    visual_reranker = VisualReRanker(config["models"].get("vlm_model", "Qwen/Qwen2-VL-7B-Instruct"))
    
    keyframes_dir = config["data"].get("keyframes_dir")
    map_keyframes_dir = config["data"].get("map_keyframes_dir") or config["data"].get("metadata_dir")
    
    # Tu dong xac dinh thu muc map-keyframes tren Kaggle
    map_csv_count = 0
    if map_keyframes_dir and os.path.exists(map_keyframes_dir):
        map_csv_count = len(glob.glob(os.path.join(map_keyframes_dir, "*.csv")) + glob.glob(os.path.join(map_keyframes_dir, "**", "*.csv"), recursive=True))
    if map_csv_count == 0 and os.path.exists("/kaggle/input"):
        csvs = glob.glob("/kaggle/input/**/L*.csv", recursive=True)
        if csvs:
            map_keyframes_dir = os.path.dirname(csvs[0])
            map_csv_count = len(csvs)
            
    print(f"Map-Keyframes Directory: {map_keyframes_dir} ({map_csv_count} CSV files found)")
    if map_csv_count == 0:
        print("Canh bao: Chua tim thay thu muc map-keyframes CSV!")
    else:
        print("Ket noi thanh cong thu muc map-keyframes cua BTC.")
    print("-----------------------------------------------------")
    
    txt_files = []
    if os.path.isfile(input_dir) and input_dir.lower().endswith(".zip"):
        unzip_tmp = "/kaggle/working/bo_de_thi_extracted"
        os.makedirs(unzip_tmp, exist_ok=True)
        with zipfile.ZipFile(input_dir, "r") as zf:
            zf.extractall(unzip_tmp)
        input_dir = unzip_tmp
        
    if os.path.isdir(input_dir):
        zips = glob.glob(os.path.join(input_dir, "*.zip")) + glob.glob(os.path.join(input_dir, "**", "*.zip"), recursive=True)
        for z in zips:
            try:
                with zipfile.ZipFile(z, "r") as zf:
                    zf.extractall(input_dir)
            except Exception:
                pass
                
    if os.path.exists(input_dir):
        txt_files = sorted(glob.glob(os.path.join(input_dir, "*.txt")))
        if not txt_files:
            txt_files = sorted(glob.glob(os.path.join(input_dir, "**", "*.txt"), recursive=True))
            
    if not txt_files and os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if any(k in root.lower() for k in ["thu-nghiem", "bo-de-thi", "query", "queries"]):
                for file in files:
                    if file.lower().endswith(".txt"):
                        txt_files.append(os.path.join(root, file))
                    elif file.lower().endswith(".zip"):
                        try:
                            unzip_dir = "/kaggle/working/bo_de_thi_auto"
                            os.makedirs(unzip_dir, exist_ok=True)
                            with zipfile.ZipFile(os.path.join(root, file), "r") as zf:
                                zf.extractall(unzip_dir)
                            txt_files = sorted(glob.glob(os.path.join(unzip_dir, "**", "*.txt"), recursive=True))
                        except Exception:
                            pass
                if txt_files:
                    break
                    
    txt_files = sorted(list(set(txt_files)))
    if not txt_files:
        print(f"Khong tim thay file .txt nao trong {input_dir}!")
        return

    print(f"Tim thay {len(txt_files)} file cau hoi can xu ly.")
    print("-----------------------------------------------------")
    
    for file_path in tqdm(txt_files, desc="Xu ly cau hoi"):
        parsed = parse_query_file(file_path)
        task_type = parsed["task_type"]
        query_id = parsed["query_id"]
        query_text = parsed["query"]
        
        csv_filename = f"{query_id}.csv"
        csv_filepath = os.path.join(submission_dir, csv_filename)
        
        q_info = query_processor.process(query_text)
        intent = q_info["intent_info"]
        
        dense_res = dense_searcher.search(q_info["prompt_ensemble"], top_k_videos=100)
        sparse_res = sparse_searcher.search(query_text, top_k_videos=50)
        fused = reciprocal_rank_fusion(
            dense_res, sparse_res,
            dense_weight=intent["dense_weight"],
            sparse_weight=intent["sparse_weight"],
            dense_dict=getattr(dense_searcher, "last_dense_dict", None)
        )
        
        fused = object_searcher.boost_candidates(fused, q_info.get("query_en", query_text))
        
        try:
            fused = visual_reranker.rerank_candidates(fused, query_text, keyframes_dir, top_n_verify=5)
        except Exception as e:
            pass

        # TASK 1: TEXTUAL KIS
        if task_type == "kis":
            top100_preds = generate_diversity_top100_kis(
                fused, keyframes_dir, metadata_dir=map_keyframes_dir, total_preds=100
            )
            
            with open(csv_filepath, "w", encoding="utf-8", newline="") as f_out:
                for pred in top100_preds:
                    vid = pred["video_id"]
                    fid = int(pred["frame_id"]) if str(pred["frame_id"]).isdigit() else pred["frame_id"]
                    f_out.write(f"{vid}, {fid}\n")
                    
        # TASK 2: VISUAL Q&A
        elif task_type == "qa":
            question = parsed["question"]
            ans_res = solve_task2(
                query_text, question, fused, keyframes_dir, 
                model_id=config["models"]["vlm_model"],
                metadata_dir=map_keyframes_dir,
                object_searcher=object_searcher
            )
            vlm_answer = format_answer_for_csv(ans_res["answer"])
            
            top100_preds = generate_diversity_top100_kis(
                fused, keyframes_dir, metadata_dir=map_keyframes_dir, total_preds=100
            )
            
            with open(csv_filepath, "w", encoding="utf-8", newline="") as f_out:
                for pred in top100_preds:
                    vid = pred["video_id"]
                    fid = int(pred["frame_id"]) if str(pred["frame_id"]).isdigit() else pred["frame_id"]
                    f_out.write(f"{vid}, {fid}, {vlm_answer}\n")
                    
        # TASK 3: TRAKE
        elif task_type == "trake":
            events = parsed["events"]
            align_res = solve_task3(
                events, fused, keyframes_dir, dense_searcher, 
                metadata_dir=map_keyframes_dir, query_processor=query_processor
            )
            best_vid = align_res["video_id"]
            best_frame_ids = align_res["frame_ids"]
            
            with open(csv_filepath, "w", encoding="utf-8", newline="") as f_out:
                clean_fids = [str(int(f)) if str(f).isdigit() else str(f) for f in best_frame_ids]
                f_out.write(f"{best_vid}, " + ", ".join(clean_fids) + "\n")
                
                count = 1
                for cand in fused:
                    vid = cand["video_id"]
                    if vid == best_vid:
                        continue
                    sub_align = solve_task3(
                        events, [cand], keyframes_dir, dense_searcher, 
                        metadata_dir=map_keyframes_dir, query_processor=query_processor
                    )
                    sub_fids = [str(int(f)) if str(f).isdigit() else str(f) for f in sub_align["frame_ids"]]
                    if len(sub_fids) == len(events):
                        f_out.write(f"{vid}, " + ", ".join(sub_fids) + "\n")
                        count += 1

                    if count >= 100:
                        break
                        
                while count < 100:
                    dummy_fids = clean_fids if clean_fids else ["0"] * len(events)
                    f_out.write(f"{best_vid}, " + ", ".join(dummy_fids) + "\n")
                    count += 1

    print("-----------------------------------------------------")
    print("Dong goi thu muc submission vao file zip...")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(submission_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.join("submission", file)
                zipf.write(file_path, arcname=arcname)
                
    elapsed = time.time() - start_time
    print(f"Hoan tat: File nop bai tai {os.path.abspath(output_zip)}")
    print(f"Thoi gian thuc hien: {elapsed:.2f} giay")
    print("=====================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Thu muc chua cac file .txt truy van cua BTC")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output_zip", default="submission.zip")
    args = parser.parse_args()
    
    run_codabench_pipeline(args.input_dir, args.config, args.output_zip)
