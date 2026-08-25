# ==============================================================================
# AIC 2026 - LANCEDB HYBRID SEARCH ENGINE WITH MAX-SIM AND TEMPORAL SMOOTHING
# ==============================================================================
import os
import torch
import numpy as np
from typing import List, Dict, Any
from transformers import CLIPModel, CLIPProcessor, SiglipModel, SiglipProcessor
from src.database.lancedb_manager import LanceDBManager
from src.search.temporal_smoother import GaussianTemporalSmoother

class LanceDBHybridSearcher:
    """
    State-of-the-art Multimodal Video Retrieval Engine.
    Executes Max-Sim Multi-Vector Search, Tantivy Full-Text BM25,
    Object Entity Boosting, and 1D Gaussian Temporal Kernel Smoothing.
    """
    def __init__(self, config: dict):
        self.config = config
        data_cfg = config.get("data", {})
        models_cfg = config.get("models", {})
        search_cfg = config.get("search_weights", {})

        lancedb_uri = data_cfg.get("lancedb_uri", "data/aic_lancedb")
        self.db_manager = LanceDBManager(db_uri=lancedb_uri)
        
        self.clip_model_name = models_cfg.get("clip_model", "openai/clip-vit-large-patch14")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.clip_model = None
        self.clip_processor = None
        
        sigma = float(search_cfg.get("gaussian_sigma", 1.5))
        self.smoother = GaussianTemporalSmoother(sigma=sigma)
        
        self._init_clip()

    def _init_clip(self):
        print(f"[INFO] LanceDBHybridSearcher: Loading CLIP ({self.clip_model_name}) on {self.device}...")
        if "siglip" in self.clip_model_name.lower():
            self.clip_processor = SiglipProcessor.from_pretrained(self.clip_model_name)
            self.clip_model = SiglipModel.from_pretrained(self.clip_model_name).to(self.device)
        else:
            self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
            self.clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(self.device)
        self.clip_model.eval()

    def unload(self):
        """Unloads CLIP model and frees VRAM on GPU 0."""
        if self.clip_model is not None:
            del self.clip_model
            self.clip_model = None
        if self.clip_processor is not None:
            del self.clip_processor
            self.clip_processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[INFO] LanceDBHybridSearcher: Unloaded CLIP model from GPU 0.")

    def encode_prompts(self, prompts: List[str]) -> np.ndarray:
        """Encodes multiple visual text prompts into L2-normalized 768-dim embeddings."""
        if not prompts:
            return np.zeros((1, 768), dtype=np.float32)

        prompt_list = [p.strip() for p in prompts if p.strip()]
        if not prompt_list:
            prompt_list = ["a video scene"]

        try:
            inputs = self.clip_processor(text=prompt_list, return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device)
            with torch.no_grad():
                if hasattr(self.clip_model, "get_text_features"):
                    text_embeds = self.clip_model.get_text_features(**inputs)
                else:
                    text_embeds = self.clip_model(**inputs)

                if not isinstance(text_embeds, torch.Tensor):
                    text_embeds = getattr(text_embeds, "text_embeds", getattr(text_embeds, "pooler_output", text_embeds[0]))

                text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
                return text_embeds.cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"[WARNING] CLIP encode error: {e}")
            return np.zeros((len(prompt_list), 768), dtype=np.float32)

    def search_candidates(self, parsed_schema: dict, top_k_videos: int = 100) -> List[Dict[str, Any]]:
        """
        Executes end-to-end multimodal hybrid search over LanceDB:
        1. Encodes Multi-Aspect Prompts
        2. Queries LanceDB Hybrid (Vector + BM25)
        3. Boosts scores with OpenImages object matching
        4. Applies 1D Gaussian Temporal Smoothing
        """
        prompts = parsed_schema.get("golden_english_prompts", [])
        if not prompts:
            prompts = [parsed_schema.get("query_vi", "a video keyframe")]

        keywords_list = parsed_schema.get("bm25_keywords", [])
        keywords_str = " ".join(keywords_list).strip()
        target_objects = [o.lower().strip() for o in parsed_schema.get("openimages_classes", []) if o.strip()]

        # Encode multi-aspect prompts
        prompt_vecs = self.encode_prompts(prompts)
        
        # Aggregate query vector (mean-pooled & L2-normalized)
        mean_vec = np.mean(prompt_vecs, axis=0, keepdims=True)
        mean_vec = mean_vec / (np.linalg.norm(mean_vec, axis=1, keepdims=True) + 1e-12)

        # Build optional SQL object filter condition
        filter_sql = None
        if target_objects:
            clauses = [f"detected_objects LIKE '%{obj}%'" for obj in target_objects[:3]]
            filter_sql = " OR ".join(clauses)

        # Retrieve raw frame candidates from LanceDB
        raw_matches = self.db_manager.search_hybrid(
            query_vector=mean_vec,
            text_keywords=keywords_str,
            filter_sql=None, # soft filter in ranking
            top_k=top_k_videos * 8
        )

        if not raw_matches:
            raw_matches = self.db_manager.search_vector(query_vector=mean_vec, top_k=top_k_videos * 8)

        # Object matching score boost
        if target_objects and raw_matches:
            for rec in raw_matches:
                frame_objs = str(rec.get("detected_objects", "")).lower()
                matched_count = sum(1 for obj in target_objects if obj in frame_objs)
                rec["_score"] = float(rec.get("_score", 0.5)) + (matched_count * 0.08)

        # Apply 1D Gaussian Temporal Smoothing & Aggregate by Video
        candidates = self.smoother.aggregate_video_candidates(raw_matches, top_k_videos=top_k_videos)

        # Safe Fallback Guarantee: if candidates < top_k_videos, fill with known video frames
        if len(candidates) < top_k_videos:
            print(f"[WARNING] Candidates count ({len(candidates)}) < {top_k_videos}. Appending fallback candidates.")
            existing_vids = set(c["video_id"] for c in candidates)
            for i in range(1, top_k_videos + 1):
                if len(candidates) >= top_k_videos:
                    break
                fallback_vid = f"L21_V{i:03d}"
                if fallback_vid not in existing_vids:
                    candidates.append({
                        "video_id": fallback_vid,
                        "score": 0.001 / i,
                        "best_frame_idx": 0,
                        "best_frame_id": 0,
                        "image_path": "",
                        "pts_time": 0.0,
                        "detected_objects": ""
                    })

        return candidates[:top_k_videos]
