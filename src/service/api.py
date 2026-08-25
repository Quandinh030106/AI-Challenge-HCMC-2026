# ==============================================================================
# AIC 2026 - LANCEDB FASTAPI MULTIMODAL SEARCH SERVICE
# ==============================================================================
import os
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from src.database.lancedb_manager import LanceDBManager
from src.preprocessing.llm_query_parser import LLMQueryParser
from src.search.lancedb_hybrid_search import LanceDBHybridSearcher
from src.tasks.task_kis import TextualKISSolver
from src.tasks.task_vqa import VisualVQASolver
from src.tasks.task_trake import TRAKESolver

app = FastAPI(
    title="AIC 2026 Multimodal Retrieval API",
    description="High-Performance Hybrid Multimodal Search Engine (LanceDB, CLIP, Qwen2.5-VL)",
    version="2.0.0"
)

# Global engines
CONFIG = {}
DB_MANAGER = None
SEARCHER = None

def load_system_config():
    global CONFIG, DB_MANAGER, SEARCHER
    config_path = "configs/lancedb_config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            CONFIG = yaml.safe_load(f)
    else:
        CONFIG = {"data": {"lancedb_uri": "data/aic_lancedb"}}

    lancedb_uri = CONFIG.get("data", {}).get("lancedb_uri", "data/aic_lancedb")
    DB_MANAGER = LanceDBManager(db_uri=lancedb_uri)

@app.on_event("startup")
def startup_event():
    print("[INFO] Starting AIC 2026 Multimodal API Service...")
    load_system_config()

class SearchRequest(BaseModel):
    query_text: str = Field(..., description="Vietnamese or English natural language query")
    task_type: str = Field("kis", description="'kis', 'qa', or 'trake'")
    raw_question: Optional[str] = Field("", description="Specific question text for VQA")
    events: Optional[List[str]] = Field(None, description="Event sequence list for TRAKE")
    top_k: int = Field(100, description="Number of results to return (max 100)")
    use_vlm: bool = Field(False, description="Whether to run deep visual verification with VLM")

class SearchResponse(BaseModel):
    query_text: str
    task_type: str
    total_results: int
    results: List[Dict[str, Any]]

@app.get("/health")
def health_check():
    db_status = "ready" if (DB_MANAGER and DB_MANAGER.is_ready()) else "not_connected"
    return {
        "status": "healthy",
        "database_status": db_status,
        "database_uri": DB_MANAGER.db_uri if DB_MANAGER else "none"
    }

@app.post("/search/hybrid", response_model=SearchResponse)
def hybrid_search_api(req: SearchRequest):
    if DB_MANAGER is None or not DB_MANAGER.is_ready():
        raise HTTPException(status_code=503, detail="LanceDB Master Store is not ready. Please run ingest pipeline first.")

    global SEARCHER
    if SEARCHER is None:
        SEARCHER = LanceDBHybridSearcher(config=CONFIG)

    # 1. Linguistic Parsing
    nlp_model_id = CONFIG.get("models", {}).get("nlp_llm", "Qwen/Qwen2.5-7B-Instruct")
    parser = LLMQueryParser(model_id=nlp_model_id)
    schema = parser.parse_query_dynamically(
        query_vi=req.query_text,
        task_type=req.task_type,
        raw_question=req.raw_question or ""
    )
    if req.events:
        schema["events"] = req.events

    # 2. Hybrid Retrieval
    candidates = SEARCHER.search_candidates(schema, top_k_videos=req.top_k)

    # 3. Task Solving Format
    if req.task_type == "kis":
        solver = TextualKISSolver(enable_vlm_verify=req.use_vlm)
        preds = solver.solve(schema, candidates, total_preds=req.top_k)
    elif req.task_type == "qa":
        solver = VisualVQASolver(db_manager=DB_MANAGER)
        preds = solver.solve(schema, candidates, total_preds=req.top_k)
    elif req.task_type == "trake":
        solver = TRAKESolver(db_manager=DB_MANAGER)
        preds = solver.solve(schema, candidates, total_preds=req.top_k)
    else:
        preds = candidates[:req.top_k]

    return {
        "query_text": req.query_text,
        "task_type": req.task_type,
        "total_results": len(preds),
        "results": preds
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
