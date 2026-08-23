import os
import sys
import glob
import json
import zipfile
import shutil
import argparse
import time
import re

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
    Loads system configuration:
    - Supports direct Kaggle notebook dictionary.
    - Supports YAML configuration files.
    - Performs fallback auto-discovery scan on Kaggle input directory.
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

    if os.path.exists("/kaggle/input"):
        data_cfg = config.get("data", {})
        
        if not data_cfg.get("features_dir") or not os.path.exists(data_cfg.get("features_dir", "")):
            for root, _, files in os.walk("/kaggle/input"):
                if any(f.endswith(".npy") for f in files):
                    data_cfg["features_dir"] = root
                    print(f"[INFO] Auto-Discovery: Features directory found at '{root}'")
                    break

        if not data_cfg.get("keyframes_dir") or not os.path.exists(data_cfg.get("keyframes_dir", "")):
            for root, dirs, _ in os.walk("/kaggle/input"):
                if "keyframe" in root.lower() or "keyframes" in root.lower():
                    data_cfg["keyframes_dir"] = root
                    print(f"[INFO] Auto-Discovery: Keyframes directory found at '{root}'")
                    break

    return config


def parse_raw_query_file(file_path):
    """
    Parses test query .txt files provided by BTC.
    Classifies task_type (KIS, QA, TRAKE) based on filename suffixes or text content.
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
    """Formats VQA answer string adhering to Codabench CSV quote escaping rules."""
    cleaned = clean_vlm_answer(ans_text)
    escaped = cleaned.replace('"', '""')
    return f'"{escaped}"'


def run_sequential_pipeline(input_dir, config_path="configs/default.yaml", output_zip="submission.zip", query_filter=None):
    """
    Sequential Multi-GPU Execution Pipeline:
    Step 1: NLP LLM (Qwen2.5-7B) parses query -> extracts schema -> unloads from VRAM.
    Step 2: Generic Hybrid Search (CLIP + BM25 + OpenImages) -> retrieves top candidates.
    Step 3: Heavy VLM (Qwen2.5-VL-7B) -> resolves KIS, VQA & TRAKE DP alignment -> packages submission.zip.
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
        print(f"[ERROR] No query text files found in '{input_dir}'")
        return

    print("================================================================================")
    print(f"[INFO] STARTING SEQUENTIAL PIPELINE FOR {len(txt_files)} QUERIES")
    print("================================================================================")

    # Step 1: NLP LLM Parsing
    print("\n[INFO] Step 1: Loading NLP LLM (Qwen2.5-7B-Instruct)...")
    nlp_model_name = config.get("models", {}).get("nlp_llm", "Qwen/Qwen2.5-7B-Instruct")
    llm_parser = LLMQueryParser(model_id=nlp_model_name)
    llm_parser.load_model()

    parsed_queries = []
    for file_path in tqdm(txt_files, desc="Step 1: LLM Parsing"):
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
    print("[INFO] Step 1 completed. NLP LLM unloaded from VRAM.\n")

    # Step 2: Hybrid Candidate Retrieval
    print("[INFO] Step 2: Initializing Hybrid Searcher (CLIP + BM25 + OpenImages)...")
    searcher = GenericHybridSearcher(config=config)

    retrieved_candidates = {}
    for schema in tqdm(parsed_queries, desc="Step 2: Candidate Retrieval"):
        qid = schema["query_id"]
        candidates = searcher.search_candidates(schema, top_k_videos=100)
        retrieved_candidates[qid] = candidates

    print("[INFO] Step 2 completed. Retrieval done.\n")

    # Step 3: Heavy VLM Task Solving
    print("[INFO] Step 3: Loading Heavy VLM (Qwen2.5-VL-7B)...")
    vlm_name = config.get("models", {}).get("vlm_model", "Qwen/Qwen2.5-VL-7B-Instruct")
    keyframes_dir = config.get("data", {}).get("keyframes_dir", None)
    metadata_dir = config.get("data", {}).get("metadata_dir", None)

    task_solvers = TaskSolvers(keyframes_dir=keyframes_dir, metadata_dir=metadata_dir, vlm_model_id=vlm_name)

    for schema in tqdm(parsed_queries, desc="Step 3: Task Solving"):
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

    print("\n[INFO] Packaging submission directory to zip file...")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(submission_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.join("submission", file)
                zipf.write(file_path, arcname=arcname)

    elapsed = time.time() - start_time
    print("================================================================================")
    print(f"[INFO] Pipeline completed successfully. Output: {os.path.abspath(output_zip)}")
    print(f"[INFO] Total execution time: {elapsed:.2f} seconds")
    print("================================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Path to directory containing query .txt files")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output_zip", default="submission.zip")
    parser.add_argument("--query_filter", default=None)
    args = parser.parse_args()

    run_sequential_pipeline(args.input_dir, args.config, args.output_zip, query_filter=args.query_filter)
