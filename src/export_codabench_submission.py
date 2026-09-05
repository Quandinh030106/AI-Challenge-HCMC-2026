import os
import sys

# Cau hinh bo nho GPU chong phan manh VRAM tren Kaggle
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

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
from src.search.sequence_search import rerank_sequence_aware_kis
from src.search.temporal_refiner import TemporalRefiner
from src.tasks.task1_kis import get_frame_id_from_idx, generate_diversity_top100_kis, gaussian_smooth_scores
from src.tasks.task2_vqa import (
    build_task2_top100_predictions,
    solve_task2,
)
from src.tasks.task3_trake import solve_task3, align_events_dynamic_programming
from src.search.object_search import ObjectSearcher
from src.search.visual_reranker import VisualReRanker

def load_config(config_path="configs/default.yaml"):
    """Doc file cau hinh he thong."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def parse_query_file(file_path):
    """
    Phan tich file query BTC.

    Output:
    - KIS:
        {
            "query_id": str,
            "task_type": "kis",
            "query": str
        }

    - Q&A:
        {
            "query_id": str,
            "task_type": "qa",
            "query": visual_context,
            "question": question
        }

    - TRAKE:
        {
            "query_id": str,
            "task_type": "trake",
            "query": retrieval_context,
            "context": context,
            "events": [E1, E2, E3, ...]
        }

    Quy tac TRAKE:
    - Context dung de retrieve video.
    - Chi cac dong E1/E2/E3... moi tao semantic event.
    """

    filename = os.path.basename(file_path)
    query_id = os.path.splitext(filename)[0]
    q_id_lower = query_id.lower()

    with open(file_path, "r", encoding="utf-8") as f:
        raw_lines = [
            line.strip()
            for line in f.readlines()
            if line.strip()
        ]

    full_content = "\n".join(raw_lines).strip()
    full_content_lower = full_content.lower()

    # ---------------------------------------------------------
    # Helper 1: Task type tu filename
    # ---------------------------------------------------------
    filename_tokens = [
        token
        for token in re.split(r"[-_]+", q_id_lower)
        if token
    ]

    is_explicit_trake = any(
        token in {"trake", "event"}
        for token in filename_tokens
    )

    is_explicit_qa = any(
        token in {"qa", "vqa"}
        for token in filename_tokens
    )

    is_explicit_kis = "kis" in filename_tokens

    # ---------------------------------------------------------
    # Helper 2: Nhan dien dong semantic event TRAKE
    # Ho tro:
    # E1 ...
    # E1: ...
    # Event 1 ...
    # Su kien 1 ...
    # Buoc 1 ...
    # 1. ...
    # ---------------------------------------------------------
    trake_event_pattern = re.compile(
        r"^(?:"
        r"e\s*(\d+)"
        r"|(?:sự\s*kiện|su\s*kien|event|bước|buoc)\s*(\d+)"
        r"|(\d+)\s*[\.\:\)]"
        r")"
        r"\s*[:\.\-\)]?\s*(.*)$",
        flags=re.IGNORECASE,
    )

    def parse_trake_lines(lines):
        context_lines = []
        events = []
        seen_first_event = False

        for line in lines:
            match = trake_event_pattern.match(line)

            if match:
                seen_first_event = True
                event_text = (match.group(4) or "").strip()

                if event_text:
                    events.append(event_text)

                continue

            # Dong khong co marker truoc E1 -> Context retrieval.
            if not seen_first_event:
                context_lines.append(line)
                continue

            # Dong khong co marker sau khi da gap E1:
            # coi la continuation cua event truoc, khong tao event moi.
            if events:
                events[-1] = f"{events[-1]} {line}".strip()

        context = " ".join(context_lines).strip()

        # Neu query TRAKE khong co context rieng,
        # dung event text lam retrieval fallback.
        retrieval_query = (
            context
            if context
            else " ".join(events).strip()
        )

        return context, events, retrieval_query

    # ---------------------------------------------------------
    # Helper 3: Q&A visual context / question split
    # ---------------------------------------------------------
    def looks_like_question(sentence):
        text = sentence.strip()
        text_lower = text.lower()

        if "?" in text:
            return True

        if re.match(
            r"^(câu hỏi|cau hoi|question|hỏi|hoi|cho biết|cho biet)\b",
            text_lower,
        ):
            return True

        question_phrases = [
            " bao nhiêu",
            " là gì",
            " ở đâu",
            " màu gì",
            " tên của ",
            " ai là",
        ]

        return any(
            phrase in f" {text_lower}"
            for phrase in question_phrases
        )

    def parse_qa_lines(lines):
        content = " ".join(lines).strip()

        sentences = [
            sentence.strip()
            for sentence in re.split(
                r"(?<=[.!?])\s+",
                content,
            )
            if sentence.strip()
        ]

        question_start = None

        for idx, sentence in enumerate(sentences):
            if looks_like_question(sentence):
                question_start = idx
                break

        if question_start is None:
            # File co suffix QA nhung khong tach duoc question:
            # khong tu gan full content thanh ca query lan question.
            return content, ""

        visual_context = " ".join(
            sentences[:question_start]
        ).strip()

        question = " ".join(
            sentences[question_start:]
        ).strip()

        question = re.sub(
            r"^(câu hỏi|cau hoi|question)\s*[:\.]?\s*",
            "",
            question,
            flags=re.IGNORECASE,
        ).strip()

        # Neu question nam ngay dau file va khong co visual_context rieng,
        # giu full content cho retrieval thay vi tao query rong.
        if not visual_context:
            visual_context = content

        return visual_context, question

    # =========================================================
    # 1. EXPLICIT TASK DETECTION TU FILENAME
    # =========================================================

    # TRAKE duoc uu tien truoc vi marker E1/E2/E3 co cau truc manh.
    if is_explicit_trake:
        context, events, retrieval_query = parse_trake_lines(
            raw_lines
        )

        if not events:
            print(
                f"[{query_id}] CANH BAO: "
                "File duoc nhan dien TRAKE nhung khong tim thay E1/E2/..."
            )

        print(
            f"[{query_id}] -> TASK 3 (TRAKE) | "
            f"Context: '{context[:80]}...' | "
            f"{len(events)} events"
        )

        return {
            "query_id": query_id,
            "task_type": "trake",
            "query": retrieval_query,
            "context": context,
            "events": events,
        }

    if is_explicit_qa:
        query, question = parse_qa_lines(raw_lines)

        if not question:
            print(
                f"[{query_id}] CANH BAO: "
                "File QA nhung parser chua tach duoc cau hoi."
            )

        print(
            f"[{query_id}] -> TASK 2 (Visual Q&A) | "
            f"Visual: '{query[:80]}...' | "
            f"Question: '{question}'"
        )

        return {
            "query_id": query_id,
            "task_type": "qa",
            "query": query,
            "question": question,
        }

    if is_explicit_kis:
        query = " ".join(raw_lines).strip()

        print(
            f"[{query_id}] -> TASK 1 (Textual KIS)"
        )

        return {
            "query_id": query_id,
            "task_type": "kis",
            "query": query,
        }

    # =========================================================
    # 2. FALLBACK TASK DETECTION KHI FILENAME KHONG CO SUFFIX
    # =========================================================

    # Structural TRAKE marker manh hon Q&A keyword.
    trake_event_count = sum(
        1
        for line in raw_lines
        if trake_event_pattern.match(line)
    )

    if trake_event_count >= 2:
        context, events, retrieval_query = parse_trake_lines(
            raw_lines
        )

        print(
            f"[{query_id}] -> Nhan dien noi dung: "
            f"TASK 3 (TRAKE) | {len(events)} events"
        )

        return {
            "query_id": query_id,
            "task_type": "trake",
            "query": retrieval_query,
            "context": context,
            "events": events,
        }

    qa_indicators = [
        "?",
        "câu hỏi",
        "cau hoi",
        "question",
        "q&a",
        "hỏi:",
        "là gì",
        "ở đâu",
        "thế nào",
        "màu gì",
        "bao nhiêu",
        "tên của",
        "ai là",
        "mấy câu thơ",
        "tiêu đề",
    ]

    if any(
        indicator in full_content_lower
        for indicator in qa_indicators
    ):
        query, question = parse_qa_lines(raw_lines)

        print(
            f"[{query_id}] -> Nhan dien noi dung: "
            f"TASK 2 (Visual Q&A) | "
            f"Question: '{question}'"
        )

        return {
            "query_id": query_id,
            "task_type": "qa",
            "query": query,
            "question": question,
        }

    # Mac dinh la KIS.
    print(
        f"[{query_id}] -> Nhan dien noi dung: "
        "TASK 1 (Textual KIS)"
    )

    return {
        "query_id": query_id,
        "task_type": "kis",
        "query": full_content,
    }



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



def run_codabench_pipeline(input_dir, config_path="configs/default.yaml", output_zip="submission.zip"):
    """Chay toan bo pipeline tren bo de thi va tao file submission.zip."""
    start_time = time.time()
    config = load_config(config_path)
    
    submission_dir = "submission"
    if os.path.exists(submission_dir):
        shutil.rmtree(submission_dir)
    os.makedirs(submission_dir, exist_ok=True)

    
    print("=====================================================")
    print("KHOI CHAY HE THONG TAO SUBMISSION AIC 2026")
    print(f"Thu muc de thi : {input_dir}")
    print(f"File zip xuat  : {output_zip}")
    print("=====================================================")
    
    dense_searcher = DenseSearcher(config)
    sparse_searcher = SparseSearcher(config)
    query_processor = QueryProcessor(config)
    object_searcher = ObjectSearcher(config)
    vlm_model_name = config.get("models", {}).get("vlm_model", "Qwen/Qwen2-VL-2B-Instruct")
    visual_reranker = VisualReRanker(vlm_model_name)
    temporal_refiner = TemporalRefiner(config, dense_searcher)

    
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
                if csv_files:
                    map_keyframes_dir = cmd
                    map_csv_count = len(csv_files)
                    break
            except Exception as exc:
                print(
                    "Canh bao: Khong doc duoc thu muc Map-Keyframes "
                    f"{cmd}: {exc}"
                )

            
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
            except Exception as exc:
                print(
                    f"Canh bao: Khong giai nen duoc {z}: {exc}"
                )
                
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
                        except Exception as exc:
                            print(
                                "Canh bao: Khong giai nen duoc "
                                f"{os.path.join(root, file)}: {exc}"
                            )

    txt_files = sorted(list(set(all_found_txts)), key=natural_sort_key)
    if not txt_files:
        print(f"Khong tim thay file .txt nao trong {input_dir}!")
        return

    print(f"Tim thay DAY DU {len(txt_files)} file cau hoi theo dung thu tu:")
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
        
        # Video retrieval chi dung visual/context description.
        # Question duoc giu rieng cho VQA sau retrieval.
        search_text = query_text
        dense_res = dense_searcher.search(q_info["semantic_views"], top_k_videos=100)
        sparse_res = sparse_searcher.search(search_text, top_k_videos=50)
        dense_w = 0.4 if task_type == "qa" else intent["dense_weight"]
        sparse_w = 0.6 if task_type == "qa" else intent["sparse_weight"]
        
        fused = reciprocal_rank_fusion(
            dense_res, sparse_res,
            dense_weight=dense_w,
            sparse_weight=sparse_w,
            dense_dict=getattr(dense_searcher, "last_dense_dict", None)
        )
        pre_object_fused = [dict(item) for item in fused]
        
        fused = object_searcher.boost_candidates(fused, f"{query_text} {q_info.get('query_en', '')}")



        # TASK 1: TEXTUAL KIS
        if task_type == "kis":
            fused, _ = rerank_sequence_aware_kis(
                query_text=query_text,
                fused_candidates=fused,
                dense_searcher=dense_searcher,
                sparse_searcher=sparse_searcher,
                query_processor=query_processor,
                config=config,
                pre_object_candidates=pre_object_fused,
                query_id=query_id,
            )
            if visual_reranker is not None:
                fused = visual_reranker.rerank_candidates(
                    fused, query_text, keyframes_dir, top_n_verify=5
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
                object_searcher=object_searcher,
                ocr_dir=config["data"].get("metadata_dir"),
                qa_config=config.get("search", {}).get("qa_evidence", {}),
                temporal_refiner=temporal_refiner,
                query_processor=query_processor,
                query_id=query_id,
            )
            
            promoted_idx = ans_res.get("promoted_idx", 0)
            if promoted_idx > 0 and promoted_idx < len(fused):
                promoted_cand = fused.pop(promoted_idx)
                fused.insert(0, promoted_cand)

            top100_preds = build_task2_top100_predictions(
                fused_candidates=fused,
                answer_result=ans_res,
                keyframes_dir=keyframes_dir,
                metadata_dir=map_keyframes_dir,
                total_preds=100,
                qa_config=config.get("search", {}).get("qa_evidence", {}),
            )
            
            with open(csv_filepath, "w", encoding="utf-8", newline="") as f_out:
                for pred in top100_preds:
                    vid = pred["video_id"]
                    fid = int(pred["frame_id"]) if str(pred["frame_id"]).isdigit() else pred["frame_id"]
                    answer_for_row = format_answer_for_csv(pred.get("answer"))
                    f_out.write(f"{vid}, {fid}, {answer_for_row}\n")

                    
        # TASK 3: TRAKE
        elif task_type == "trake":
            events = parsed["events"]
            from src.tasks.task3_trake import solve_task3_batch
            all_aligned_preds = solve_task3_batch(
                events, fused, keyframes_dir, dense_searcher,
                metadata_dir=map_keyframes_dir, query_processor=query_processor,
                total_preds=100, config=config, temporal_refiner=temporal_refiner,
                query_id=query_id,
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
    args = parser.parse_args()
    
    run_codabench_pipeline(args.input_dir, args.config, args.output_zip)
