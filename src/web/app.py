# ==============================================================================
# AIC 2026 - PRODUCTION FASTAPI WEB SERVICE & AUTOMATED SUBMISSION GENERATOR
# ==============================================================================
import os
import sys
import yaml
import time
import json
import zipfile
import shutil
import torch
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Request, Form, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.database.lancedb_manager import LanceDBManager
from src.preprocessing.llm_query_parser import LLMQueryParser
from src.search.lancedb_hybrid_search import LanceDBHybridSearcher
from src.tasks.task_kis import TextualKISSolver
from src.tasks.task_vqa import VisualVQASolver
from src.tasks.task_trake import TRAKESolver

app = FastAPI(
    title="AI Challenge HCMC 2026 - Multimodal Video Search Engine",
    description="State-of-the-Art Web Service API & Automated Codabench Submission Generator",
    version="2.0.0"
)

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(BASE_DIR, "configs", "lancedb_config.yaml")
SUBMISSION_DIR = os.path.join(BASE_DIR, "submissions")
STATIC_DIR = os.path.join(BASE_DIR, "src", "web", "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "src", "web", "templates")

os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Global Cached Pipeline Instances
GLOBAL_CONFIG = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        GLOBAL_CONFIG = yaml.safe_load(f)

DB_MANAGER = None
SEARCHER = None

def get_db_manager():
    global DB_MANAGER
    if DB_MANAGER is None:
        lancedb_uri = GLOBAL_CONFIG.get("data", {}).get("lancedb_uri", "data/aic_lancedb")
        if not os.path.isabs(lancedb_uri):
            lancedb_uri = os.path.join(BASE_DIR, lancedb_uri)
        DB_MANAGER = LanceDBManager(db_uri=lancedb_uri)
    return DB_MANAGER

def get_searcher():
    global SEARCHER
    if SEARCHER is None:
        SEARCHER = LanceDBHybridSearcher(config=GLOBAL_CONFIG)
    return SEARCHER

@app.get("/", response_class=HTMLResponse)
async def serve_web_ui(request: Request):
    """Serves the Production Web UI with Left Sidebar Search Bar & Interactive Video Player."""
    return templates.TemplateResponse("index.html", {"request": request, "title": "AIC 2026 AI Video Search Engine"})

@app.post("/api/v1/search")
async def execute_search(
    query_vi: str = Form(...),
    task_type: str = Form("kis"),
    query_id: Optional[str] = Form(None)
):
    """
    Executes 3-Stage Accuracy-First Search Pipeline:
    1. Dynamic NLP Query Parsing
    2. LanceDB Hybrid Search & Gaussian Smoothing
    3. Heavy VLM Visual Inspection & Top-1 Promotion
    AUTOMATICALLY exports official Codabench submission CSV for BTC!
    """
    if not query_vi.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    if not query_id or not query_id.strip():
        query_id = f"query_{int(time.time())}_{task_type}"

    t_start = time.time()
    db = get_db_manager()

    # --------------------------------------------------------------------------
    # STAGE 1: NLP QUERY PARSING
    # --------------------------------------------------------------------------
    nlp_model_id = GLOBAL_CONFIG.get("models", {}).get("nlp_llm", "Qwen/Qwen2.5-7B-Instruct")
    llm_parser = LLMQueryParser(model_id=nlp_model_id)
    schema = llm_parser.parse_query_dynamically(query_vi=query_vi, task_type=task_type)
    schema["query_id"] = query_id
    schema["task_type"] = task_type
    llm_parser.unload_model()

    # --------------------------------------------------------------------------
    # STAGE 2: LANCEDB HYBRID CANDIDATE RETRIEVAL
    # --------------------------------------------------------------------------
    searcher = get_searcher()
    candidates = searcher.search_candidates(schema, top_k_videos=100)

    # --------------------------------------------------------------------------
    # STAGE 3: TASK SOLVER & VLM INSPECTION
    # --------------------------------------------------------------------------
    target_vlm_device = "cuda:1" if torch.cuda.device_count() > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu")
    vlm_model_id = GLOBAL_CONFIG.get("models", {}).get("vlm_model", "Qwen/Qwen2.5-VL-7B-Instruct")

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
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True
        )
        vlm_processor = AutoProcessor.from_pretrained(vlm_model_id, min_pixels=100352, max_pixels=301056, trust_remote_code=True)
        vlm_model.eval()
    except Exception as e:
        print(f"[WARNING] VLM Loading notice: {e}")

    # Solve & verify candidates
    if task_type == "kis":
        solver = TextualKISSolver(vlm_model=vlm_model, vlm_processor=vlm_processor, device=target_vlm_device, enable_vlm_verify=True)
        final_preds = solver.solve(schema, candidates, total_preds=100, top_k_verify=5)
    elif task_type == "qa":
        solver = VisualVQASolver(vlm_model=vlm_model, vlm_processor=vlm_processor, db_manager=db, device=target_vlm_device)
        final_preds = solver.solve(schema, candidates, total_preds=100, eval_candidates_count=4)
    else: # trake
        solver = TRAKESolver(db_manager=db, clip_encoder=searcher)
        final_preds = solver.solve(schema, candidates, total_preds=100)

    # Free VLM memory
    if vlm_model is not None:
        del vlm_model
    if vlm_processor is not None:
        del vlm_processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------------------------------------------------
    # STAGE 4: AUTOMATED SUBMISSION CSV GENERATION (FOR BTC)
    # --------------------------------------------------------------------------
    csv_file_path = os.path.join(SUBMISSION_DIR, f"{query_id}.csv")
    with open(csv_file_path, "w", encoding="utf-8", newline="") as f_csv:
        for item in final_preds:
            if task_type == "kis":
                f_csv.write(f"{item['video_id']},{item['frame_id']}\n")
            elif task_type == "qa":
                answer_clean = item.get('answer', 'none').replace(',', ' ')
                f_csv.write(f"{item['video_id']},{item['frame_id']},{answer_clean}\n")
            elif task_type == "trake":
                fids_str = ",".join(str(fid) for fid in item.get('frame_ids', [item['frame_id']]))
                f_csv.write(f"{item['video_id']},{fids_str}\n")

    t_elapsed = round(time.time() - t_start, 2)

    # Enrich web response items with human-readable timestamps and streaming URLs
    enriched_results = []
    for rank, p in enumerate(final_preds, 1):
        vid = p["video_id"]
        fid = p.get("frame_id", 0)
        pts = float(p.get("pts_time", 0.0))
        minutes = int(pts // 60)
        seconds = int(pts % 60)
        ts_str = f"{minutes:02d}:{seconds:02d}"

        # Construct YouTube / Streaming Embed URL
        youtube_url = f"https://www.youtube.com/watch?v={vid}&t={int(pts)}s"
        
        enriched_results.append({
            "rank": rank,
            "video_id": vid,
            "frame_id": fid,
            "pts_time": pts,
            "timestamp_formatted": ts_str,
            "score": round(float(p.get("score", 0.5)), 4),
            "keyframe_caption": p.get("keyframe_caption", "High-precision keyframe candidate"),
            "detected_objects": p.get("detected_objects", ""),
            "answer": p.get("answer", ""),
            "youtube_url": youtube_url,
            "image_path": p.get("image_path", "")
        })

    return JSONResponse({
        "status": "success",
        "query_id": query_id,
        "query_vi": query_vi,
        "task_type": task_type,
        "elapsed_seconds": t_elapsed,
        "submission_csv_path": csv_file_path,
        "parsed_schema": schema,
        "total_results": len(enriched_results),
        "results": enriched_results
    })

@app.get("/api/v1/submission/download/{query_id}")
async def download_single_csv(query_id: str):
    """Downloads official Codabench submission CSV for a specific query."""
    csv_path = os.path.join(SUBMISSION_DIR, f"{query_id}.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"Submission CSV for query '{query_id}' not found.")
    return FileResponse(csv_path, media_type="text/csv", filename=f"{query_id}.csv")

@app.get("/api/v1/submission/download_all")
async def download_all_submissions():
    """Packages all exported query CSVs into a single submission.zip for Codabench."""
    zip_path = os.path.join(SUBMISSION_DIR, "submission.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in os.listdir(SUBMISSION_DIR):
            if file.endswith(".csv"):
                file_path = os.path.join(SUBMISSION_DIR, file)
                zipf.write(file_path, arcname=os.path.join("submission", file))
                zipf.write(file_path, arcname=file)
    return FileResponse(zip_path, media_type="application/zip", filename="submission.zip")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
