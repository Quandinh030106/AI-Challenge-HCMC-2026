import sys
import os

# Tu dong them thu muc goc vao sys.path de khong bao gio loi import src
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import argparse
import glob
import re
import csv
import zipfile
import time
from tqdm import tqdm
from src.utils import load_config
from src.search.dense_search import DenseSearcher
from src.search.sparse_search import SparseSearcher
from src.search.fusion import reciprocal_rank_fusion
from src.preprocessing.query_processor import QueryProcessor
from src.tasks.task1_kis import get_frame_id_from_idx, generate_diversity_top100_kis, gaussian_smooth_scores
from src.tasks.task2_vqa import solve_task2
from src.tasks.task3_trake import solve_task3, align_events_dynamic_programming


def parse_query_file(file_path):
    """
    Doc va phan tich noi dung file .txt truy van cua BTC dua vao hau to ten file (kis, qa, trake).
    """
    filename = os.path.basename(file_path).lower()
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
        
    full_text = " ".join(lines)
    
    # 1. KIS Query
    if "kis" in filename:
        return {
            "task_type": "kis",
            "query_id": os.path.splitext(os.path.basename(file_path))[0],
            "query": full_text
        }
        
    # 2. Q&A Query
    elif "qa" in filename or "q&a" in filename:
        query_text = ""
        question = ""
        
        # Neu co nhieu dong (Dong 1: Boi canh, Dong 2: Cau hoi)
        if len(lines) >= 2:
            query_text = lines[0]
            question = lines[1]
        else:
            # Neu chi co 1 dong, tach theo dau ? hoac tu khoa
            parts = re.split(r'(?<=\?)\s*|(?=câu hỏi:)|(?=question:)', full_text, flags=re.IGNORECASE)
            if len(parts) >= 2:
                query_text = parts[0].strip()
                question = parts[1].strip()
            else:
                query_text = full_text
                question = full_text
                
        # Loai bo tien to neu co
        question = re.sub(r'^(câu hỏi|question)\s*[:\-]\s*', '', question, flags=re.IGNORECASE)
        query_text = re.sub(r'^(bối cảnh|mô tả|context|query)\s*[:\-]\s*', '', query_text, flags=re.IGNORECASE)
        
        return {
            "task_type": "qa",
            "query_id": os.path.splitext(os.path.basename(file_path))[0],
            "query": query_text if query_text else question,
            "question": question if question else query_text
        }
        
    # 3. TRAKE Query
    elif "trake" in filename:
        events = []
        main_query = lines[0] if lines else ""
        
        # Neu co danh sach dong su kien con
        if len(lines) > 1:
            for l in lines[1:]:
                clean_l = re.sub(r'^(\d+[\.\)\-:]|\-|\*)\s*', '', l).strip()
                if clean_l:
                    events.append(clean_l)
        else:
            # Neu chi co 1 dong chua dau phay / cham phay
            split_events = re.split(r'[,;]|\s+sau đó\s+|\s+tiếp theo\s+|\s+then\s+', full_text, flags=re.IGNORECASE)
            events = [e.strip() for e in split_events if e.strip()]
            
        if not events:
            events = [main_query]
            
        return {
            "task_type": "trake",
            "query_id": os.path.splitext(os.path.basename(file_path))[0],
            "query": main_query,
            "events": events
        }
        
    # Mac dinh neu khong ro hau to -> KIS
    return {
        "task_type": "kis",
        "query_id": os.path.splitext(os.path.basename(file_path))[0],
        "query": full_text
    }

def format_answer_for_csv(ans_text):
    """
    Format answer cho Q&A theo dung quy chuan BTC:
    - Bo dau xuong dong thua
    - Escape dau ngoac kep neu can
    - Giu do dai duoi 100 ky tu
    """
    if not ans_text:
        return "không rõ"
    ans_clean = str(ans_text).replace("\r", "").replace("\n", " ").strip()
    ans_clean = ans_clean[:95]
    
    # Neu chua dau phay hoac ngoac kep -> bao quanh bang ngoac kep
    if ',' in ans_clean or '"' in ans_clean:
        ans_escaped = ans_clean.replace('"', '""')
        return f'"{ans_escaped}"'
    return ans_clean

def run_codabench_pipeline(input_dir, config_path="configs/default.yaml", output_zip="submission.zip"):
    """
    Chay toan bo pipeline tren goi cau hoi cua BTC va tao file submission.zip nop Codabench.
    """
    start_time = time.time()
    config = load_config(config_path)
    
    # 1. Thu muc tam luu cac file csv
    submission_dir = "submission"
    os.makedirs(submission_dir, exist_ok=True)
    
    print("=====================================================")
    print("🚀 KHOI CHAY HE THONG TAO FILE NOP BAI CODABENCH AIC 2026")
    print(f"Thu muc chua goi cau hoi : {input_dir}")
    print(f"File zip dau ra          : {output_zip}")
    print("=====================================================")
    
    # 2. Khoi tao cac module tim kiem
    dense_searcher = DenseSearcher(config)
    sparse_searcher = SparseSearcher(config)
    query_processor = QueryProcessor()
    
    keyframes_dir = config["data"].get("keyframes_dir")
    map_keyframes_dir = config["data"].get("map_keyframes_dir") or config["data"].get("metadata_dir")
    
    # 3. Tim tat ca cac file cau hoi (.txt)
    txt_files = sorted(glob.glob(os.path.join(input_dir, "*.txt")))
    if not txt_files:
        txt_files = sorted(glob.glob(os.path.join(input_dir, "**", "*.txt"), recursive=True))
        
    if not txt_files:
        print(f"❌ Khong tim thay file .txt nao trong {input_dir}!")
        return
        
    print(f"Tim thay {len(txt_files)} file cau hoi can xu ly:")
    for f in txt_files:
        print(f" - {os.path.basename(f)}")
    print("-----------------------------------------------------")
    
    # 4. Xu ly tung file cau hoi va xuat ra tung file .csv tuong ung
    for file_path in tqdm(txt_files, desc="Xu ly cau hoi"):
        parsed = parse_query_file(file_path)
        task_type = parsed["task_type"]
        query_id = parsed["query_id"]
        query_text = parsed["query"]
        
        csv_filename = f"{query_id}.csv"
        csv_filepath = os.path.join(submission_dir, csv_filename)
        
        # Buoc chung: Tien xu ly query + Dense + Sparse + Fusion
        q_info = query_processor.process(query_text)
        intent = q_info["intent_info"]
        
        dense_res = dense_searcher.search(q_info["prompt_ensemble"], top_k_videos=35)
        sparse_res = sparse_searcher.search(query_text, top_k_videos=35)
        fused = reciprocal_rank_fusion(
            dense_res, sparse_res,
            dense_weight=intent["dense_weight"],
            sparse_weight=intent["sparse_weight"]
        )
        
        # --- TASK 1: TEXTUAL KIS ---
        if task_type == "kis":
            # Sinh 100 dong theo format: <Tên file video>, <Frame Idx> (KHONG HEADER)
            top100_preds = generate_diversity_top100_kis(
                fused, keyframes_dir, metadata_dir=map_keyframes_dir, total_preds=100
            )
            
            with open(csv_filepath, "w", encoding="utf-8", newline="") as f_out:
                for pred in top100_preds:
                    vid = pred["video_id"]
                    fid = int(pred["frame_id"]) if str(pred["frame_id"]).isdigit() else pred["frame_id"]
                    f_out.write(f"{vid}, {fid}\n")
                    
        # --- TASK 2: VISUAL Q&A ---
        elif task_type == "qa":
            question = parsed["question"]
            ans_res = solve_task2(query_text, question, fused, keyframes_dir, model_id=config["models"]["vlm_model"])
            vlm_answer = format_answer_for_csv(ans_res["answer"])
            
            top100_preds = generate_diversity_top100_kis(
                fused, keyframes_dir, metadata_dir=map_keyframes_dir, total_preds=100
            )
            
            with open(csv_filepath, "w", encoding="utf-8", newline="") as f_out:
                for pred in top100_preds:
                    vid = pred["video_id"]
                    fid = int(pred["frame_id"]) if str(pred["frame_id"]).isdigit() else pred["frame_id"]
                    f_out.write(f"{vid}, {fid}, {vlm_answer}\n")
                    
        # --- TASK 3: TRAKE ---
        elif task_type == "trake":
            events = parsed["events"]
            align_res = solve_task3(events, fused, keyframes_dir, dense_searcher, metadata_dir=map_keyframes_dir)
            best_vid = align_res["video_id"]
            best_frame_ids = align_res["frame_ids"]
            
            with open(csv_filepath, "w", encoding="utf-8", newline="") as f_out:
                # Dong 1: Ket qua tot nhat tu Dynamic Programming
                clean_fids = [str(int(f)) if str(f).isdigit() else str(f) for f in best_frame_ids]
                f_out.write(f"{best_vid}, " + ", ".join(clean_fids) + "\n")
                
                # Cac dong tiep theo tu cac video ung vien khac
                count = 1
                for cand in fused:
                    vid = cand["video_id"]
                    if vid == best_vid:
                        continue
                    # Lay chuoi frame dai dien
                    sub_align = solve_task3(events, [cand], keyframes_dir, dense_searcher, metadata_dir=map_keyframes_dir)
                    sub_fids = [str(int(f)) if str(f).isdigit() else str(f) for f in sub_align["frame_ids"]]
                    if len(sub_fids) == len(events):
                        f_out.write(f"{vid}, " + ", ".join(sub_fids) + "\n")
                        count += 1

                    if count >= 100:
                        break
                        
                while count < 100:
                    dummy_fids = ["0"] * len(events)
                    f_out.write(f"{best_vid}, " + ", ".join(dummy_fids) + "\n")
                    count += 1

    # 5. Dong goi thu muc submission thanh file submission.zip chuan Codabench
    print("\n-----------------------------------------------------")
    print("📦 Dang dong goi thu muc submission vao file zip...")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(submission_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Dat duong dan ben trong zip la submission/filename.csv
                arcname = os.path.join("submission", file)
                zipf.write(file_path, arcname=arcname)
                
    elapsed = time.time() - start_time
    print(f"✅ HOAN TAT! File nop bai da san sang tai: {os.path.abspath(output_zip)}")
    print(f"⏱️ Tong thoi gian thuc hien: {elapsed:.2f} giay")
    print("=====================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Thu muc chua cac file .txt truy van cua BTC")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output_zip", default="submission.zip")
    args = parser.parse_args()
    
    run_codabench_pipeline(args.input_dir, args.config, args.output_zip)
