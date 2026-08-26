# ==============================================================================
# AIC 2026 - MASTER PIPELINE COORDINATOR & CODABENCH SUBMISSION PACKAGER
# ==============================================================================
import os
import gc
import re
import glob
import yaml
import shutil
import zipfile
import torch
from tqdm import tqdm
from typing import List, Dict, Any

from src.database.ingest_pipeline import MultimodalIngestPipeline
from src.database.lancedb_manager import LanceDBManager
from src.preprocessing.llm_query_parser import LLMQueryParser
from src.search.lancedb_hybrid_search import LanceDBHybridSearcher
from src.tasks.task_kis import TextualKISSolver
from src.tasks.task_vqa import VisualVQASolver
from src.tasks.task_trake import TRAKESolver

def parse_raw_query_file(file_path: str) -> dict:
    """Parses raw Vietnamese query text file into standardized query representation."""
    qid = os.path.splitext(os.path.basename(file_path))[0]
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    task_type = "kis"
    lower_qid = qid.lower()
    if "qa" in lower_qid or "question" in lower_qid:
        task_type = "qa"
    elif "trake" in lower_qid or "temporal" in lower_qid or "event" in lower_qid:
        task_type = "trake"

    # Identify questions or event lists
    raw_question = ""
    events = []

    lines = [l.strip() for l in content.split("\n") if l.strip()]
    if task_type == "qa":
        for l in lines:
            if any(q_word in l.lower() for q_word in ["bao nhiêu", "là gì", "ở đâu", "màu gì", "như thế nào", "?"]):
                raw_question = l
                break
        if not raw_question and lines:
            raw_question = lines[-1]

    elif task_type == "trake":
        for l in lines:
            if re.match(r'^(e\d+|sự kiện \d+|\d+\.|\-|\*)\s*[:\.]?', l, re.IGNORECASE):
                events.append(l)
        if not events:
            events = lines[1:] if len(lines) > 1 else lines

    return {
        "query_id": qid,
        "query_vi": content,
        "task_type": task_type,
        "raw_question": raw_question,
        "events": events
    }

def run_master_pipeline(config_path: str = "configs/lancedb_config.yaml", output_zip: str = "submission.zip", output_dir: str = None):
    """Executes full 4-stage Accuracy-First pipeline and exports submission.zip."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if not output_zip.startswith(output_dir):
            output_zip = os.path.join(output_dir, "submission.zip")

    print("=" * 80)
    print("[INFO] AIC 2026 - STARTING ACCURACY-FIRST MULTIMODAL RETRIEVAL PIPELINE")
    print("=" * 80)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # --------------------------------------------------------------------------
    # GIAI ĐOẠN 0: KHỞI TẠO HOẶC KẾT NỐI LANCEDB
    # --------------------------------------------------------------------------
    lancedb_uri = config.get("data", {}).get("lancedb_uri", "data/aic_lancedb")
    db_manager = LanceDBManager(db_uri=lancedb_uri)
    
    if not db_manager.is_ready():
        print("[INFO] LanceDB master table not found. Running Ingest Pipeline...")
        ingest_pipeline = MultimodalIngestPipeline(config)
        ingest_pipeline.build_database(overwrite=True)
        db_manager = LanceDBManager(db_uri=lancedb_uri)

    # --------------------------------------------------------------------------
    # ĐỌC DANH SÁCH CÂU HỎI ĐỀ THI
    # --------------------------------------------------------------------------
    queries_dir = config.get("data", {}).get("queries_dir", "")
    query_files = sorted(glob.glob(os.path.join(queries_dir, "*.txt")) if os.path.exists(queries_dir) else [])
    
    if not query_files:
        # Search Kaggle input
        for root, _, files in os.walk("/kaggle/input"):
            for f in files:
                if f.endswith(".txt") and "query" in f.lower():
                    query_files.append(os.path.join(root, f))
        query_files = sorted(list(set(query_files)))

    print(f"[INFO] Found {len(query_files)} query files to process.")
    raw_queries = [parse_raw_query_file(qf) for qf in query_files]

    # --------------------------------------------------------------------------
    # GIAI ĐOẠN 1: NLP MULTI-ASPECT QUERY PARSING (GPU 0)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[INFO] GIAI ĐOẠN 1: NLP QUERY PARSING (Qwen2.5-7B on GPU 0)")
    print("=" * 80)
    
    nlp_model_id = config.get("models", {}).get("nlp_llm", "Qwen/Qwen2.5-7B-Instruct")
    llm_parser = LLMQueryParser(model_id=nlp_model_id)

    parsed_schemas = []
    for q_item in tqdm(raw_queries, desc="Phase 1: NLP Parsing"):
        schema = llm_parser.parse_query_dynamically(
            query_vi=q_item["query_vi"],
            task_type=q_item["task_type"],
            raw_question=q_item["raw_question"]
        )
        schema["query_id"] = q_item["query_id"]
        schema["task_type"] = q_item["task_type"]
        if q_item.get("events"):
            schema["events"] = q_item["events"]
        parsed_schemas.append(schema)

    # Unload NLP LLM from GPU 0
    llm_parser.unload_model()
    del llm_parser
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------------------------------------------------
    # GIAI ĐOẠN 2: LANCEDB HYBRID CANDIDATE RETRIEVAL (GPU 0)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[INFO] GIAI ĐOẠN 2: LANCEDB HYBRID SEARCH & GAUSSIAN SMOOTHING (GPU 0)")
    print("=" * 80)

    searcher = LanceDBHybridSearcher(config=config)
    retrieved_candidates = {}
    trake_precomputed_preds = {}

    for schema in tqdm(parsed_schemas, desc="Phase 2: Hybrid Retrieval"):
        qid = schema["query_id"]
        cands = searcher.search_candidates(schema, top_k_videos=100)
        retrieved_candidates[qid] = cands

        # If TRAKE query, pre-compute Viterbi DP alignment while CLIP is active
        if schema.get("task_type") == "trake":
            trake_phase2 = TRAKESolver(db_manager=db_manager, clip_encoder=searcher)
            trake_precomputed_preds[qid] = trake_phase2.solve(schema, cands, total_preds=100)

    # Unload CLIP from GPU 0
    searcher.unload()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------------------------------------------------
    # GIAI ĐOẠN 3: TASK SOLVERS & DEEP VLM RE-RANKING (GPU 1)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[INFO] GIAI ĐOẠN 3: TASK SOLVING & DEEP VISUAL RE-RANKING (GPU 1)")
    print("=" * 80)

    vlm_model_id = config.get("models", {}).get("vlm_model", "Qwen/Qwen2.5-VL-7B-Instruct")
    target_vlm_device = "cuda:1" if torch.cuda.device_count() > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load VLM on GPU 1
    print(f"[INFO] Loading Heavy VLM ({vlm_model_id}) on {target_vlm_device}...")
    vlm_model = None
    vlm_processor = None
    try:
        from transformers import AutoProcessor, BitsAndBytesConfig
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as VLMClass
        except ImportError:
            from transformers import AutoModelForCausalLM as VLMClass

        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        vlm_model = VLMClass.from_pretrained(
            vlm_model_id,
            quantization_config=bnb_config if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=target_vlm_device if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        min_pixels = int(config.get("vlm_verification", {}).get("min_pixels", 100352))
        max_pixels = int(config.get("vlm_verification", {}).get("max_pixels", 301056))
        vlm_processor = AutoProcessor.from_pretrained(vlm_model_id, min_pixels=min_pixels, max_pixels=max_pixels, trust_remote_code=True)
        vlm_model.eval()
        print("[INFO] Loaded Heavy VLM successfully.")
    except Exception as e:
        print(f"[WARNING] VLM Loading warning ({e}). Proceeding in lightweight mode...")

    kis_solver = TextualKISSolver(vlm_model=vlm_model, vlm_processor=vlm_processor, device=target_vlm_device, enable_vlm_verify=True)
    vqa_solver = VisualVQASolver(vlm_model=vlm_model, vlm_processor=vlm_processor, db_manager=db_manager, device=target_vlm_device)
    trake_solver = TRAKESolver(db_manager=db_manager, clip_encoder=None)

    submission_dir = "submission"
    if os.path.exists(submission_dir):
        shutil.rmtree(submission_dir)
    os.makedirs(submission_dir, exist_ok=True)

    for schema in tqdm(parsed_schemas, desc="Phase 3: Solving & Writing CSVs"):
        qid = schema["query_id"]
        task_type = schema["task_type"]
        cands = retrieved_candidates.get(qid, [])
        csv_path = os.path.join(submission_dir, f"{qid}.csv")

        if task_type == "kis":
            preds = kis_solver.solve(schema, cands, total_preds=100, top_k_verify=5)
            with open(csv_path, "w", encoding="utf-8", newline="") as f_out:
                for p in preds:
                    f_out.write(f"{p['video_id']},{p['frame_id']}\n")

        elif task_type == "qa":
            preds = vqa_solver.solve(schema, cands, total_preds=100, eval_candidates_count=4)
            with open(csv_path, "w", encoding="utf-8", newline="") as f_out:
                for p in preds:
                    f_out.write(f"{p['video_id']},{p['frame_id']},{p['answer']}\n")

        elif task_type == "trake":
            preds = trake_precomputed_preds.get(qid)
            if not preds:
                preds = trake_solver.solve(schema, cands, total_preds=100)
            with open(csv_path, "w", encoding="utf-8", newline="") as f_out:
                for p in preds:
                    fids_str = ",".join(str(fid) for fid in p["frame_ids"])
                    f_out.write(f"{p['video_id']},{fids_str}\n")

    # Unload VLM
    if vlm_model is not None:
        del vlm_model
    if vlm_processor is not None:
        del vlm_processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------------------------------------------------
    # GIAI ĐOẠN 4: VALIDATION & CODABENCH ZIP PACKAGING
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[INFO] GIAI ĐOẠN 4: SUBMISSION VALIDATION & PACKAGING")
    print("=" * 80)

    csv_files = glob.glob(os.path.join(submission_dir, "*.csv"))
    print(f"[INFO] Validating {len(csv_files)} exported CSV submission files...")
    
    for cf in csv_files:
        with open(cf, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        if len(lines) == 0:
            print(f"[ERROR] Found empty submission file: {cf}!")
        elif len(lines) < 100:
            print(f"[WARNING] File '{os.path.basename(cf)}' has {len(lines)} lines (expected 100).")

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(submission_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Mirror both 'submission/<file>' and root '<file>'
                zipf.write(file_path, arcname=os.path.join("submission", file))
                zipf.write(file_path, arcname=file)

    print(f"[INFO] All done! Submission archive created at: '{output_zip}'")

if __name__ == "__main__":
    run_master_pipeline()
