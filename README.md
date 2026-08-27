# 🚀 AI Challenge HCMC 2026 - Multimodal Video Search Engine & Production Web Service

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LanceDB](https://img.shields.io/badge/LanceDB-0.8%2B-green.svg)](https://lancedb.github.io/lancedb/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Qwen2.5-VL](https://img.shields.io/badge/Qwen2.5--VL-7B--Instruct-purple.svg)](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)

Welcome to the official repository for **AI Challenge HCMC 2026**. This project features a state-of-the-art **Multimodal Video Retrieval Pipeline** and **Production RESTful Web Service API** designed to achieve $R@1 = 1.0$ accuracy while automatically exporting official Codabench submission CSVs for competition evaluation.

---

## 🌟 Key Features & Architectural Highlights

### 1. Normalized 2-Table LanceDB Storage Architecture
- **Table `videos` (Video-Level Metadata, 873 rows)**: Stores clean YouTube/streaming MP4 URLs, titles, descriptions, keywords, duration in seconds, FPS, and total keyframe counts. Eliminates 100% text redundancy across keyframes.
- **Table `keyframes` (Keyframe Visual Store, 177,321 rows)**: Stores L2-normalized 512-dim CLIP vectors, real CSV frame IDs (`frame_id`), formatted human-readable timestamps (`timestamp_formatted`, e.g., `03:22`), Qwen2.5-VL multi-sentence natural language captions, OpenImages object tags, OCR text, and Tantivy Native Rust FTS inverted indexes (`frame_text_weighted`).

### 2. Dual-GPU Hardware Mapping Strategy
- **GPU 0 (`cuda:0`)**: Executes Phase 1 (NLP Query Parsing via `Qwen2.5-7B-Instruct` 4-bit NF4) $\rightarrow$ Unloads VRAM $\rightarrow$ Executes Phase 2 (Max-Sim CLIP 512-dim Dense Vector Search + Tantivy BM25 + 1D Gaussian Temporal Kernel).
- **GPU 1 (`cuda:1`)**: Executes Phase 3 (Heavy VLM `Qwen2.5-VL-7B-Instruct` 4-bit NF4) for Deep Visual Inspection, reading exact scale numbers, bridge names, red hat counts, and promoting verified winners to Rank 1 ($R@1 = 1.0$).

### 3. Production FastAPI Web Application
- **Modern Glassmorphism Web UI**: Features a sleek left-sidebar search panel with Task Tabs (Textual KIS, Visual Q&A, Temporal Event TRAKE).
- **Interactive Video Player**: Embedded player automatically seeks (jumps) to the exact timestamp (`pts_time`) where the AI identified the target action.
- **Automated Codabench Submission Generator**: Every executed search automatically exports the official BTC submission CSV (`/kaggle/working/submissions/{query_id}.csv`) and provides a one-click **"Download Submission Package (submission.zip)"** button.

---

## 🛠 Project Structure

```
AI-Challenge-HCMC-2026/
├── configs/
│   └── lancedb_config.yaml           # Master pipeline & model configuration
├── src/
│   ├── database/
│   │   ├── schema.py                 # PyArrow schemas for videos and keyframes tables
│   │   ├── ingest_pipeline.py        # 2-Table LanceDB ingestion & auto-zipping pipeline
│   │   └── lancedb_manager.py        # Zero-copy memory-safe query manager
│   ├── preprocessing/
│   │   ├── llm_query_parser.py       # Qwen2.5-7B NLP query parser (4-bit GPU)
│   │   └── auto_captioner.py         # Qwen2.5-VL-3B High-Res (784x784) auto-captioner
│   ├── search/
│   │   ├── temporal_smoother.py      # 1D Gaussian Temporal Kernel Smoother
│   │   └── lancedb_hybrid_search.py  # Max-Sim CLIP + Tantivy FTS BM25 hybrid searcher
│   ├── tasks/
│   │   ├── task_kis.py               # Textual KIS solver with VLM verification
│   │   ├── task_vqa.py               # Visual Q&A solver with 3-frame window inspection
│   │   └── task_trake.py             # TRAKE solver with Viterbi DP temporal alignment
│   ├── web/
│   │   ├── app.py                    # FastAPI Web Service API backend
│   │   ├── templates/index.html      # Glassmorphism Web UI layout
│   │   └── static/                   # App CSS styles and JS interaction logic
│   └── export_submission.py          # Master pipeline coordinator for full benchmark runs
└── README.md                         # Documentation
```

---

## 🚀 Step-by-Step Kaggle Execution Guide

### Cell 1: Environment Setup & Pull Latest Code
```python
import os
import torch

print("GPU Available:", torch.cuda.device_count(), "-", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

os.chdir("/kaggle/working")
!rm -rf AI-Challenge-HCMC-2026
!git clone -b web-service-api https://github.com/Quandinh030106/AI-Challenge-HCMC-2026.git

os.chdir("/kaggle/working/AI-Challenge-HCMC-2026")
print("Working Directory:", os.getcwd())

!pip install -U bitsandbytes accelerate
!pip install -q lancedb pyarrow pyyaml tqdm numpy qwen-vl-utils fastapi uvicorn[standard] jinja2 pyngrok
```

### Cell 2: Build 2-Table LanceDB Database Store
```python
import os
import yaml
import src.database.ingest_pipeline
from src.database.ingest_pipeline import MultimodalIngestPipeline

config_path = "configs/lancedb_config.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

pipeline = MultimodalIngestPipeline(config)
tables = pipeline.build_database(overwrite=True)
```

### Cell 3: Database Integrity Audit
```python
import os
import lancedb

db = lancedb.connect("/kaggle/working/aic_lancedb")
tbl_videos = db.open_table("videos")
tbl_keyframes = db.open_table("keyframes")

print(f"Videos count: {len(tbl_videos)} rows (Expected: 873)")
print(f"Keyframes count: {len(tbl_keyframes)} rows (Expected: ~177,321)")
```

### Cell 6: Full Benchmark Run Across All 25 Queries
```python
import os
import yaml
from src.export_submission import run_master_pipeline

config_path = "configs/lancedb_config.yaml"
zip_submission_path = run_master_pipeline(
    config_path=config_path,
    output_dir="/kaggle/working/submissions"
)
print(f"[INFO] Submission ZIP created at: {zip_submission_path}")
```

### Cell 7: Launch FastAPI Web App with Public Ngrok URL
```python
import os
from pyngrok import ngrok

public_url = ngrok.connect(8000).public_url
print(f"  🌟 PUBLIC WEB APP URL: {public_url}")

!uvicorn src.web.app:app --host 0.0.0.0 --port 8000
```

---

## 📜 License & Acknowledgments
Built with ❤️ for **AI Challenge HCMC 2026**. Powered by PyTorch, LanceDB, Hugging Face Transformers, and FastAPI.
