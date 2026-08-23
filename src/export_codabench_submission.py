import os
import sys
import glob
import json
import zipfile
import shutil
import argparse
import time
import re

# Đảm bảo thư mục gốc dự án nằm trong sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import yaml
from tqdm import tqdm

from src.preprocessing.llm_query_parser import LLMQueryParser
from src.search.generic_hybrid_search import GenericHybridSearcher
from src.tasks.task_solvers import TaskSolvers, clean_vlm_answer

def load_config(config_path="configs/default.yaml"):
    """
    Đọc cấu hình hệ thống một cách linh hoạt:
    - Hỗ trợ truyền thẳng Config Dictionary trực tiếp từ Kaggle Notebook cell.
    - Hỗ trợ đọc từ file YAML.
    - Tự động phát hiện (Auto-Discovery) đường dẫn trên Kaggle nếu đường dẫn bị sai.
    """
    config = None
    if isinstance(config_path, dict):
        config = config_path
    elif isinstance(config_path, str) and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            
    if not config:
        config = {
            "models": {
                "nlp_llm": "Qwen/Qwen2.5-7B-Instruct",
                "clip_model": "openai/clip-vit-large-patch14",
                "vlm_model": "Qwen/Qwen2.5-VL-7B-Instruct"
            },
            "data": {
                "features_dir": "/kaggle/input/clip-features-32-aic25-b1",
                "metadata_dir": "/kaggle/input/ai-challenge-hcmc-2026-metadata/metadata",
                "ocr_dir": "/kaggle/input/ai-challenge-hcmc-2026-metadata/ocr",
                "objects_dir": "/kaggle/input/ai-challenge-hcmc-2026-objects/objects",
                "keyframes_dir": "/kaggle/input/ai-challenge-hcmc-2026-keyframes"
            }
        }

    # TỰ ĐỘNG PHÁT HIỆN ĐƯỜNG DẪN TRÊN KAGGLE (AUTO-DISCOVERY FALLBACK)
    if os.path.exists("/kaggle/input"):
        data_cfg = config.get("data", {})
        
        # 1. Quét đường dẫn features_dir (.npy)
        if not data_cfg.get("features_dir") or not os.path.exists(data_cfg.get("features_dir", "")):
            for root, _, files in os.walk("/kaggle/input"):
                if any(f.endswith(".npy") for f in files):
                    data_cfg["features_dir"] = root
                    print(f"Auto-Discovery: Đã tự phát hiện features_dir tại '{root}'")
                    break

        # 2. Quét đường dẫn keyframes_dir
        if not data_cfg.get("keyframes_dir") or not os.path.exists(data_cfg.get("keyframes_dir", "")):
            for root, dirs, _ in os.walk("/kaggle/input"):
                if "keyframe" in root.lower() or "keyframes" in root.lower():
                    data_cfg["keyframes_dir"] = root
                    print(f"Auto-Discovery: Đã tự phát hiện keyframes_dir tại '{root}'")
                    break

    return config


def parse_raw_query_file(file_path):
    """
    Phân tích định dạng file câu hỏi .txt của BTC hoàn toàn tổng quát (Zero-Bias).
    Phận loại loại bài toán (KIS, QA, TRAKE) dựa trên cấu trúc tên file hoặc nội dung.
    """
    filename = os.path.basename(file_path)
    query_id = os.path.splitext(filename)[0]
    q_id_lower = query_id.lower()

    with open(file_path, "r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f.readlines() if line.strip()]

    full_content = "\n".join(raw_lines)
    full_content_lower = full_content.lower()

    is_kis = any(k in q_id_lower for k in ["kis", "-kis", "_kis"])
    is_qa = any(k in q_id_lower for k in ["qa", "-qa", "_qa", "vqa", "-vqa", "_vqa"])
    is_trake = any(k in q_id_lower for k in ["trake", "-trake", "_trake", "event", "-event", "_event"])

    if is_kis:
        return {"query_id": query_id, "task_type": "kis", "query_vi": full_content, "raw_question": ""}

    if is_qa:
        visual_parts = []
        question_parts = []
        for line in raw_lines:
            if "?" in line or any(k in line.lower() for k in ["câu hỏi", "question", "hỏi:"]):
                cleaned_q = re.sub(r'^(câu hỏi|question)\s*[:\.]?\s*', '', line, flags=re.IGNORECASE).strip()
                if cleaned_q:
                    question_parts.append(cleaned_q)
            else:
                visual_parts.append(line)
        query = " ".join(visual_parts).strip() or full_content
        question = " ".join(question_parts).strip() or full_content
        return {"query_id": query_id, "task_type": "qa", "query_vi": query, "raw_question": question}

    if is_trake:
        prefix_pattern = r'^(e\d+|sự kiện|event|bước|\d+[\.\:\)])\s*[:\.]?\s*'
        events = [re.sub(prefix_pattern, '', l, flags=re.IGNORECASE).strip() for l in raw_lines if l.strip()]
        return {"query_id": query_id, "task_type": "trake", "query_vi": " ".join(events), "raw_question": "", "events": events}

    if "?" in full_content_lower or "câu hỏi" in full_content_lower:
        return {"query_id": query_id, "task_type": "qa", "query_vi": full_content, "raw_question": full_content}

    return {"query_id": query_id, "task_type": "kis", "query_vi": full_content, "raw_question": ""}


def format_vqa_answer_for_csv(ans_text):
    """Format đáp án VQA cho file CSV tuân thủ chuẩn bọc ngoặc kép cho cả văn bản nhiều dòng."""
    cleaned = clean_vlm_answer(ans_text)
    escaped = cleaned.replace('"', '""')
    return f'"{escaped}"'


def run_sequential_pipeline(input_dir, config_path="configs/default.yaml", output_zip="submission.zip", query_filter=None):
    """
    PIPELINE ĐIỀU KHIỂN VÒNG ĐỜI NỐI TIẾP NGUYÊN KHỐI (100% ZERO-BIAS):
    BƯỚC 1: NLP LLM (Qwen2.5-7B) Đọc hiểu đề thi -> Sinh JSON Cấu trúc Ngữ nghĩa -> Giải phóng khỏi VRAM.
    BƯỚC 2: Generic Hybrid Search (CLIP + BM25 + OpenImages) -> Tìm kiếm Top 100 Ứng viên.
    BƯỚC 3: Heavy VLM (Qwen2.5-VL-7B trên 2 GPU) -> Giải quyết KIS, VQA & TRAKE DP Alignment -> Đóng gói submission.zip.
    """
    start_time = time.time()
    config = load_config(config_path)

    submission_dir = "/kaggle/working/submission" if os.path.exists("/kaggle/working") else "scratch/submission"
    if os.path.exists(submission_dir):
        shutil.rmtree(submission_dir)
    os.makedirs(submission_dir, exist_ok=True)

    txt_files = []
    if os.path.isfile(input_dir) and input_dir.lower().endswith(".zip"):
        unzip_tmp = "/kaggle/working/bo_de_thi_extracted"
        os.makedirs(unzip_tmp, exist_ok=True)
        with zipfile.ZipFile(input_dir, "r") as zf:
            zf.extractall(unzip_tmp)
        input_dir = unzip_tmp

    if os.path.exists(input_dir):
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(".txt"):
                    txt_files.append(os.path.join(root, file))

    txt_files = sorted(list(set(txt_files)))
    if query_filter:
        txt_files = [f for f in txt_files if str(query_filter).lower() in os.path.basename(f).lower()]

    if not txt_files:
        print(f"Không tìm thấy file câu hỏi .txt nào trong '{input_dir}'!")
        return

    print("================================================================================")
    print(f"🚀 BẮT ĐẦU PIPELINE NỐI TIẾP 3 BƯỚC CHO {len(txt_files)} CÂU HỎI THI (100% ZERO-BIAS)")
    print("================================================================================")

    # BƯỚC 1: NLP LLM (Qwen2.5-7B) NẠP VÀO VRAM -> ĐỌC HIỂU ĐỀ ĐỘNG -> XUẤT JSON
    print("\n🔹 BƯỚC 1: Nạp NLP LLM (Qwen2.5-7B-Instruct) đọc hiểu ngữ nghĩa đề thi...")
    nlp_model_name = config.get("models", {}).get("nlp_llm", "Qwen/Qwen2.5-7B-Instruct")
    llm_parser = LLMQueryParser(model_id=nlp_model_name)
    llm_parser.load_model()

    parsed_queries = []
    for file_path in tqdm(txt_files, desc="BƯỚC 1: NLP LLM Phân tích Động"):
        raw_info = parse_raw_query_file(file_path)
        schema = llm_parser.parse_query_dynamically(
            query_vi=raw_info["query_vi"],
            task_type=raw_info["task_type"],
            raw_question=raw_info.get("raw_question", "")
        )
        schema["query_id"] = raw_info["query_id"]
        schema["task_type"] = raw_info["task_type"]
        if "events" in raw_info:
            schema["events"] = raw_info["events"]
        parsed_queries.append(schema)

    llm_parser.unload_model()
    print("✅ BƯỚC 1 HOÀN TẤT: Đã giải phóng hoàn toàn NLP LLM khỏi VRAM!\n")

    # BƯỚC 2: TÌM KIẾM ĐA PHƯƠNG THỨC HỖN HỢP TỔNG QUÁT (GENERIC HYBRID SEARCH)
    print("🔹 BƯỚC 2: Khởi tạo Bộ tìm kiếm Hybrid Searcher (CLIP + BM25 + OpenImages)...")
    searcher = GenericHybridSearcher(config=config)

    retrieved_candidates = {}
    for schema in tqdm(parsed_queries, desc="BƯỚC 2: Tìm kiếm Candidate"):
        qid = schema["query_id"]
        candidates = searcher.search_candidates(schema, top_k_videos=100)
        retrieved_candidates[qid] = candidates

    print("✅ BƯỚC 2 HOÀN TẤT: Đã tìm kiếm xong Candidate cho tất cả câu hỏi!\n")

    # BƯỚC 3: DỒN 2 GPU NẠP HEAVY VLM (Qwen2.5-VL-7B) -> GIẢI BÀI TOÁN -> XUẤT CSV
    print("🔹 BƯỚC 3: Dồn 2 GPU nạp Heavy VLM (Qwen2.5-VL-7B) giải VQA, KIS & TRAKE...")
    vlm_name = config.get("models", {}).get("vlm_model", "Qwen/Qwen2.5-VL-7B-Instruct")
    keyframes_dir = config.get("data", {}).get("keyframes_dir", None)
    metadata_dir = config.get("data", {}).get("metadata_dir", None)

    task_solvers = TaskSolvers(keyframes_dir=keyframes_dir, metadata_dir=metadata_dir, vlm_model_id=vlm_name)

    for schema in tqdm(parsed_queries, desc="BƯỚC 3: Thực thi Solvers"):
        qid = schema["query_id"]
        task_type = schema["task_type"]
        candidates = retrieved_candidates.get(qid, [])
        csv_filepath = os.path.join(submission_dir, f"{qid}.csv")

        if task_type == "kis":
            preds = task_solvers.solve_kis(candidates, total_preds=100)
            with open(csv_filepath, "w", encoding="utf-8", newline="") as f_out:
                for p in preds:
                    f_out.write(f"{p['video_id']}, {p['frame_id']}\n")

        elif task_type == "qa":
            vqa_res = task_solvers.solve_vqa(schema, candidates)
            promoted_idx = vqa_res.get("promoted_idx", 0)
            if promoted_idx > 0 and promoted_idx < len(candidates):
                promoted_cand = candidates.pop(promoted_idx)
                candidates.insert(0, promoted_cand)

            vlm_ans_formatted = format_vqa_answer_for_csv(vqa_res["answer"])
            preds = task_solvers.solve_kis(candidates, total_preds=100)

            with open(csv_filepath, "w", encoding="utf-8", newline="") as f_out:
                for p in preds:
                    f_out.write(f"{p['video_id']}, {p['frame_id']}, {vlm_ans_formatted}\n")

        elif task_type == "trake":
            aligned = task_solvers.solve_trake(schema, candidates, dense_engine=getattr(searcher, "dense_engine", None), total_preds=100)
            with open(csv_filepath, "w", encoding="utf-8", newline="") as f_out:
                for item in aligned:
                    vid = item["video_id"]
                    fids = ", ".join([str(f) for f in item["frame_ids"]])
                    f_out.write(f"{vid}, {fids}\n")

    task_solvers.unload_vlm()

    print("\n📦 Đóng gói thư mục submission vào file zip...")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(submission_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.join("submission", file)
                zipf.write(file_path, arcname=arcname)

    elapsed = time.time() - start_time
    print("================================================================================")
    print(f"✅ CHÚC MỪNG: THÀNH CÔNG NỘP BÀI! File tại: {os.path.abspath(output_zip)}")
    print(f"⏱️ Tổng thời gian thực thi: {elapsed:.2f} giây")
    print("================================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Thư mục chứa các file .txt câu hỏi của BTC")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output_zip", default="submission.zip")
    parser.add_argument("--query_filter", default=None)
    args = parser.parse_args()

    run_sequential_pipeline(args.input_dir, args.config, args.output_zip, query_filter=args.query_filter)
