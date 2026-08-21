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
import shutil
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

def extract_trake_events(raw_lines):
    """Boc tach cac su kien TRAKE: Uu tien tuyet doi cac dong co tien to (E1:, E2:, E3:, Su kien 1:, Buoc 1:...)."""
    prefix_pattern = r'^(e\d+|sự kiện|su kien|event|bước|buoc|\d+[\.\:\)]|\(\d+\))\s*\d*\s*[:\.]?\s*'
    
    # 1. Kiem tra xem file co dong nao chua tien to su kien ro rang hay khong
    has_explicit_prefix = any(
        re.match(r'^(e\d+|sự kiện|su kien|event|bước|buoc|\d+[\.\:\)]|\(\d+\))', l.strip(), re.IGNORECASE)
        for l in raw_lines if l.strip()
    )
    
    events = []
    if has_explicit_prefix:
        for l in raw_lines:
            l_strip = l.strip()
            if not l_strip:
                continue
            if re.match(r'^(e\d+|sự kiện|su kien|event|bước|buoc|\d+[\.\:\)]|\(\d+\))', l_strip, re.IGNORECASE):
                cleaned = re.sub(prefix_pattern, '', l_strip, flags=re.IGNORECASE).strip()
                if cleaned:
                    events.append(cleaned)
    else:
        # Fallback neu file khong ghi ro E1, E2: Bo qua cac dong dan nhap
        intro_keywords = [
            "tìm các sự kiện", "tìm sự kiện", "gồm các khoảnh khắc", "gồm các sự kiện",
            "khoảnh khắc sơ chế", "các sự kiện sau", "các khoảnh khắc", "sự kiện sau", "các sự kiện"
        ]
        for l in raw_lines:
            l_strip = l.strip()
            if not l_strip:
                continue
            l_lower = l_strip.lower()
            is_intro = any(kw in l_lower for kw in intro_keywords) or (
                l_lower.endswith(":") and not any(
                    l_lower.startswith(p) for p in ["sự kiện", "su kien", "event", "bước", "buoc", "khoảnh khắc", "e1", "e2", "1.", "2.", "(1)", "(2)"]
                )
            )
            if is_intro:
                continue
            cleaned = re.sub(prefix_pattern, '', l_strip, flags=re.IGNORECASE).strip()
            if cleaned:
                events.append(cleaned)
                
    if not events:
        events = [re.sub(prefix_pattern, '', l, flags=re.IGNORECASE).strip() for l in raw_lines if l.strip()]
        
    return events


def parse_query_file(file_path):
    """Phan tich noi dung file cau hoi (.txt) uu tien tuyet doi theo ten duoi file."""
    filename = os.path.basename(file_path)
    query_id = os.path.splitext(filename)[0]
    q_id_lower = query_id.lower()
    
    with open(file_path, "r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f.readlines() if line.strip()]
        
    full_content = "\n".join(raw_lines)
    full_content_lower = full_content.lower()
    
    # 1. UU TIEN SO 1: NHAN DIEN THEO TEN DUOI FILE (-kis, -qa, -trake)
    is_explicit_kis = any(q_id_lower.endswith(k) or f"-{k}-" in q_id_lower or f"_{k}_" in q_id_lower for k in ["kis", "-kis", "_kis"])
    is_explicit_qa = any(q_id_lower.endswith(k) or f"-{k}-" in q_id_lower or f"_{k}_" in q_id_lower for k in ["qa", "-qa", "_qa", "vqa", "-vqa", "_vqa"])
    is_explicit_trake = any(q_id_lower.endswith(k) or f"-{k}-" in q_id_lower or f"_{k}_" in q_id_lower for k in ["trake", "-trake", "_trake", "event", "-event", "_event"])
    
    if is_explicit_kis:
        print(f"[{query_id}] -> Xac dinh theo ten file: TASK 1 (Textual KIS)")
        return {
            "query_id": query_id,
            "task_type": "kis",
            "query": " ".join(raw_lines).strip()
        }
        
    if is_explicit_qa:
        visual_lines = []
        question_lines = []
        is_q = False
        for line in raw_lines:
            line_l = line.lower()
            if "?" in line or any(k in line_l for k in ["câu hỏi", "cau hoi", "question", "hỏi:"]):
                is_q = True
                cleaned = re.sub(r'^(câu hỏi|cau hoi|question)\s*[:\.]?\s*', '', line, flags=re.IGNORECASE).strip()
                if cleaned:
                    question_lines.append(cleaned)
            elif is_q:
                question_lines.append(line)
            else:
                visual_lines.append(line)
                
        # Phan tach ro rang phan mo ta boi canh va cau hoi cu the
        visual_parts = []
        question_parts = []
        for line in raw_lines:
            # Tach cac cau trong dong
            sents = re.split(r'(?<=[.!?])\s+', line)
            for s in sents:
                s_strip = s.strip()
                if not s_strip:
                    continue
                if "?" in s_strip or re.search(r'^(câu hỏi|hỏi|cho biết|tìm xem)\b', s_strip, re.IGNORECASE):
                    cleaned_q = re.sub(r'^(câu hỏi|cau hoi|question)\s*[:\.]?\s*', '', s_strip, flags=re.IGNORECASE).strip()
                    if cleaned_q:
                        question_parts.append(cleaned_q)
                else:
                    visual_parts.append(s_strip)
                    
        query = " ".join(visual_parts).strip() or full_content
        question = " ".join(question_parts).strip() or full_content
        print(f"[{query_id}] -> Xac dinh theo ten file: TASK 2 (Visual Q&A) | Visual: '{query[:60]}...' | Question: '{question}'")
        return {
            "query_id": query_id,
            "task_type": "qa",
            "query": query,
            "question": question
        }

        
    if is_explicit_trake:
        events = extract_trake_events(raw_lines)
        print(f"[{query_id}] -> Xac dinh theo ten file: TASK 3 (TRAKE) | {len(events)} su kien")
        return {
            "query_id": query_id,
            "task_type": "trake",
            "events": events,
            "query": " ".join(events)
        }

    # 2. FALLBACK KHI TEN FILE KHONG CO HAU TO: PHAN TICH THEO NOI DUNG
    qa_indicators = [
        "?", "câu hỏi", "cau hoi", "question", "q&a", "hỏi:", "là gì", "ở đâu", 
        "thế nào", "màu gì", "bao nhiêu", "tên của", "ai là", "mấy câu thơ", "tiêu đề"
    ]
    if any(k in full_content_lower for k in qa_indicators):
        visual_lines = []
        question_lines = []
        is_q = False
        for line in raw_lines:
            line_l = line.lower()
            if "?" in line or any(k in line_l for k in ["câu hỏi", "cau hoi", "question", "hỏi:"]):
                is_q = True
                cleaned = re.sub(r'^(câu hỏi|cau hoi|question)\s*[:\.]?\s*', '', line, flags=re.IGNORECASE).strip()
                if cleaned:
                    question_lines.append(cleaned)
            elif is_q:
                question_lines.append(line)
            else:
                visual_lines.append(line)
        query = " ".join(visual_lines).strip() or full_content
        question = " ".join(question_lines).strip() or full_content
        print(f"[{query_id}] -> Nhan dien: TASK 2 (Visual Q&A) | Question: '{question}'")
        return {"query_id": query_id, "task_type": "qa", "query": query, "question": question}
        
    has_trake_flag = any(re.match(r'^(sự kiện|su kien|event|bước|buoc|e\d+|\d+[\.\:\)]|\(\d+\))\s*', line, re.IGNORECASE) for line in raw_lines)
    if has_trake_flag and len(raw_lines) >= 3:
        events = extract_trake_events(raw_lines)
        print(f"[{query_id}] -> Nhan dien: TASK 3 (TRAKE) | {len(events)} su kien")
        return {"query_id": query_id, "task_type": "trake", "events": events, "query": " ".join(events)}

    print(f"[{query_id}] -> Nhan dien: TASK 1 (Textual KIS)")
    return {"query_id": query_id, "task_type": "kis", "query": full_content}


def format_answer_for_csv(ans_text):
    """Format cau tra loi VQA cho file CSV tuan thu dung quy dinh toi da 100 ky tu cua BTC."""
    if not ans_text or str(ans_text).strip() in ["", '""', "''", "None"]:
        ans_text = "Không rõ"
    ans_cleaned = str(ans_text).strip().strip('"').strip("'").replace("\n", " ").strip()
    # Quy dinh cua BTC: Answer (Q&A) co do dai toi da 100 ky tu
    if len(ans_cleaned) > 100:
        ans_cleaned = ans_cleaned[:100].strip()
    if not ans_cleaned:
        ans_cleaned = "Không rõ"
    ans_escaped = ans_cleaned.replace('"', '""')
    return f'"{ans_escaped}"'



def run_codabench_pipeline(input_dir, config_path="configs/default.yaml", output_zip="submission.zip", query_filter=None):
    """Chay pipeline tren bo de thi (hoac 1 cau hoi cu the neu co query_filter)."""
    start_time = time.time()
    config = load_config(config_path)
    
    submission_dir = "submission"
    if os.path.exists(submission_dir):
        shutil.rmtree(submission_dir)
    os.makedirs(submission_dir, exist_ok=True)

    
    print("=====================================================")
    print("KHOI CHAY HE THONG TAO SUBMISSION AIC 2026")
    print(f"Thu muc de thi : {input_dir}")
    if query_filter:
        print(f"Bo loc cau hoi : CHI CHAY CAU TRUY VAN MANG TUKHOA '{query_filter}'")
    print(f"File zip xuat  : {output_zip}")
    print("=====================================================")
    
    dense_searcher = DenseSearcher(config)
    sparse_searcher = SparseSearcher(config)
    query_processor = QueryProcessor()
    object_searcher = ObjectSearcher(config)
    vlm_model_name = config.get("models", {}).get("vlm_model", "Qwen/Qwen2-VL-2B-Instruct")
    visual_reranker = VisualReRanker(vlm_model_name)

    
    keyframes_dir = config["data"].get("keyframes_dir")
    map_keyframes_dir = config["data"].get("map_keyframes_dir") or config["data"].get("metadata_dir")
    
    # Tu dong xac dinh thu muc map-keyframes tren Kaggle sieu toc (< 0.001s)
    map_csv_count = 0
    candidate_map_dirs = [
        map_keyframes_dir,
        "/kaggle/input/ai-challenge-hcmc-2026-metadata/map-keyframes-aic25-b1/map-keyframes",
        "/kaggle/input/datasets/quninhphmanh/ai-challenge-hcmc-2026-metadata/map-keyframes-aic25-b1/map-keyframes",
        "/kaggle/input/ai-challenge-hcmc-2026-metadata/map-keyframes",
        "/kaggle/input/datasets/quninhphmanh/ai-challenge-hcmc-2026-metadata/map-keyframes"
    ]
    for cmd in candidate_map_dirs:
        if cmd and os.path.exists(cmd):
            try:
                csv_files = [f for f in os.listdir(cmd) if f.lower().endswith(".csv")]
                if len(csv_files) > 10:
                    map_keyframes_dir = cmd
                    map_csv_count = len(csv_files)
                    break
            except Exception:
                pass

            
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
                
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
        
    all_found_txts = []
    if os.path.exists(input_dir):
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(".txt"):
                    all_found_txts.append(os.path.join(root, file))
                    
    if not all_found_txts and os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if any(k in root.lower() for k in ["thu-nghiem", "bo-de-thi", "query", "queries"]):
                for file in files:
                    if file.lower().endswith(".txt"):
                        all_found_txts.append(os.path.join(root, file))
                    elif file.lower().endswith(".zip"):
                        try:
                            unzip_dir = "/kaggle/working/bo_de_thi_auto"
                            os.makedirs(unzip_dir, exist_ok=True)
                            with zipfile.ZipFile(os.path.join(root, file), "r") as zf:
                                zf.extractall(unzip_dir)
                            for r_sub, _, f_sub in os.walk(unzip_dir):
                                for fs in f_sub:
                                    if fs.lower().endswith(".txt"):
                                        all_found_txts.append(os.path.join(r_sub, fs))
                        except Exception:
                            pass

    txt_files = sorted(list(set(all_found_txts)), key=natural_sort_key)
    
    # Loc rieng cau hoi theo query_filter neu nguoi dung yeu cau
    if query_filter:
        txt_files = [f for f in txt_files if str(query_filter).lower() in os.path.basename(f).lower()]
        
    if not txt_files:
        print(f"Khong tim thay file .txt nao phu hop voi bo loc '{query_filter}'!")
        return

    print(f"Tim thay {len(txt_files)} file cau hoi duoc chon de chay:")
    for f in txt_files:
        print(f"  -> {os.path.basename(f)}")
    print("-----------------------------------------------------")
    
    for file_path in tqdm(txt_files, desc="Xu ly cau hoi"):

        parsed = parse_query_file(file_path)
        task_type = parsed["task_type"]
        query_id = parsed["query_id"]
        query_text = parsed["query"]

        
        csv_filename = f"{query_id}.csv"
        csv_filepath = os.path.join(submission_dir, csv_filename)
        
        # q_info duoc tao tu query_text (mo ta thi giac sach, khong bi nhiem tu khoa cau hoi)
        q_info = query_processor.process(query_text)
        intent = q_info["intent_info"]
        
        search_text = f"{query_text} {parsed.get('question', '')}".strip() if task_type == "qa" else query_text
        dense_res = dense_searcher.search(q_info["prompt_ensemble"], top_k_videos=100)
        sparse_res = sparse_searcher.search(search_text, top_k_videos=50)
        fused = reciprocal_rank_fusion(
            dense_res, sparse_res,
            dense_weight=intent["dense_weight"],
            sparse_weight=intent["sparse_weight"],
            dense_dict=getattr(dense_searcher, "last_dense_dict", None)
        )
        
        fused = object_searcher.boost_candidates(fused, f"{query_text} {q_info.get('query_en', '')}")


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
                model_id=vlm_model_name,
                metadata_dir=map_keyframes_dir,
                object_searcher=object_searcher
            )
            
            promoted_idx = ans_res.get("promoted_idx", 0)
            if promoted_idx > 0 and promoted_idx < len(fused):
                promoted_cand = fused.pop(promoted_idx)
                fused.insert(0, promoted_cand)

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
        # TASK 3: TRAKE
        elif task_type == "trake":
            events = parsed["events"]
            from src.tasks.task3_trake import solve_task3_batch
            all_aligned_preds = solve_task3_batch(
                events, fused, keyframes_dir, dense_searcher, 
                metadata_dir=map_keyframes_dir, query_processor=query_processor,
                total_preds=100
            )
            
            with open(csv_filepath, "w", encoding="utf-8", newline="") as f_out:
                for item in all_aligned_preds:
                    vid = item["video_id"]
                    clean_fids = [str(int(f)) if str(f).isdigit() else str(f) for f in item["frame_ids"]]
                    f_out.write(f"{vid}, " + ", ".join(clean_fids) + "\n")


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
    parser.add_argument("--query_filter", default=None, help="Chi chay rieng mot cau hoi chua tu khoa nay (vd: 19 hoac p1-19-qa)")
    args = parser.parse_args()
    
    run_codabench_pipeline(args.input_dir, args.config, args.output_zip, query_filter=args.query_filter)

