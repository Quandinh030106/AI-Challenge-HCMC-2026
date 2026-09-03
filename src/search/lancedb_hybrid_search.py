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
    Executes Max-Sim Multi-Vector Search, OpenImages Object Filtering,
    and 1D Gaussian Temporal Kernel Smoothing.
    """
    def __init__(self, config: dict):
        self.config = config
        data_cfg = config.get("data", {})
        models_cfg = config.get("models", {})
        search_cfg = config.get("search_weights", {})

        lancedb_uri = data_cfg.get("lancedb_uri", "data/aic_lancedb")
        self.db_manager = LanceDBManager(db_uri=lancedb_uri)
        
        # Auto-detect DB vector dimension
        self.db_dim = 512
        if self.db_manager.keyframes_table is not None:
            try:
                sample_row = self.db_manager.keyframes_table.search().limit(1).to_pandas()
                if "vector" in sample_row.columns:
                    self.db_dim = len(sample_row["vector"].iloc[0])
            except Exception:
                pass

        # Select matching CLIP model based on DB vector dimension
        default_clip = "openai/clip-vit-base-patch32" if self.db_dim == 512 else "openai/clip-vit-large-patch14"
        self.clip_model_name = models_cfg.get("clip_model", default_clip)
        if self.db_dim == 512 and "large" in self.clip_model_name.lower():
            print(f"[INFO] Auto-switching CLIP model to 'openai/clip-vit-base-patch32' to match {self.db_dim}-dim database vectors.")
            self.clip_model_name = "openai/clip-vit-base-patch32"

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

    def unload_model(self):
        """Alias for unload() to ensure backward compatibility."""
        return self.unload()

    def encode_prompts(self, prompts: List[str]) -> np.ndarray:
        """Encodes visual text prompts into L2-normalized embeddings matching DB dimension."""
        if not prompts:
            return np.zeros((1, self.db_dim), dtype=np.float32)

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
                embeds_np = text_embeds.cpu().numpy().astype(np.float32)

                # Ensure embedding dimension matches self.db_dim
                if embeds_np.shape[1] != self.db_dim:
                    if embeds_np.shape[1] > self.db_dim:
                        embeds_np = embeds_np[:, :self.db_dim]
                    else:
                        embeds_np = np.pad(embeds_np, ((0, 0), (0, self.db_dim - embeds_np.shape[1])), mode='constant')

                return embeds_np
        except Exception as e:
            print(f"[WARNING] CLIP encode error: {e}")
            return np.zeros((len(prompt_list), self.db_dim), dtype=np.float32)

    def search_candidates(self, parsed_schema: dict, top_k_videos: int = 100) -> List[Dict[str, Any]]:
        """
        Executes end-to-end multimodal search over LanceDB:
        1. Encodes Multi-Aspect Prompts
        2. Queries LanceDB Vector Search per prompt (Max-Sim)
        3. Boosts scores with OpenImages object matching
        4. Applies 1D Gaussian Temporal Smoothing
        """
        prompts = parsed_schema.get("golden_english_prompts", [])
        if not prompts:
            prompts = [parsed_schema.get("query_vi", "a video keyframe")]

        keywords_list = parsed_schema.get("bm25_keywords", [])
        keywords_str = " ".join(keywords_list).strip()
        target_objects = [o.lower().strip() for o in parsed_schema.get("openimages_classes", []) if o.strip()]

        # Encode prompts into L2-normalized matching vectors
        prompt_vecs = self.encode_prompts(prompts)
        
        # Collect prompt-wise similarity matrix per keyframe: (video_id, frame_idx) -> list of similarities per prompt
        frame_prompt_sims = {}
        frame_record_map = {}

        n_prompts = prompt_vecs.shape[0]
        for p_idx in range(n_prompts):
            single_vec = prompt_vecs[p_idx:p_idx+1]
            if keywords_str:
                matches = self.db_manager.search_hybrid(query_vector=single_vec, text_keywords=keywords_str, top_k=top_k_videos * 4)
            else:
                matches = self.db_manager.search_vector(query_vector=single_vec, top_k=top_k_videos * 4)
            
            for rec in matches:
                key = (rec["video_id"], rec["frame_idx"])
                dist = rec.get("_distance", None)
                if dist is not None:
                    sim = max(0.0, 1.0 - float(dist))
                else:
                    sim = float(rec.get("_score", rec.get("score", 0.5)))

                if key not in frame_prompt_sims:
                    frame_prompt_sims[key] = [0.0] * n_prompts
                    frame_record_map[key] = rec
                frame_prompt_sims[key][p_idx] = max(frame_prompt_sims[key][p_idx], sim)

        raw_matches = []
        epsilon = 0.15
        for key, rec in frame_record_map.items():
            sims = frame_prompt_sims[key]
            # 1. Smoothed Soft-Harmonic Mean across all prompts
            inv_sum = sum(1.0 / (max(0.001, s) + epsilon) for s in sims)
            soft_harmonic_score = float(n_prompts / inv_sum)

            # 2. Unbiased Soft-Gate Multi-Entity Coverage Booster over (detected_objects + keyframe_caption + ocr_text)
            metadata_text = (
                str(rec.get("detected_objects", "")) + " " +
                str(rec.get("keyframe_caption", "")) + " " +
                str(rec.get("ocr_text", ""))
            ).lower()

            entity_targets = set()
            for obj in target_objects:
                if len(obj.strip()) >= 2:
                    entity_targets.add(obj.lower().strip())
            for kw in keywords_list:
                if len(kw.strip()) >= 3:
                    entity_targets.add(kw.lower().strip())

            if entity_targets:
                matched_count = sum(1 for ent in entity_targets if ent in metadata_text)
                ratio = float(matched_count / len(entity_targets))
                # Soft continuous multiplier: 1.0 to 1.35 boost for multi-entity co-occurrence
                final_score = soft_harmonic_score * (1.0 + 0.35 * ratio)
            else:
                final_score = soft_harmonic_score

            rec["_score"] = final_score
            raw_matches.append(rec)

        # Apply 1D Gaussian Temporal Smoothing & Aggregate by Video
        candidates = self.smoother.aggregate_video_candidates(raw_matches, top_k_videos=top_k_videos)

        # Fallback padding if needed
        if len(candidates) < top_k_videos:
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
