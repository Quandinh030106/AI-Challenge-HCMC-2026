import os
import glob
import json
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor, SiglipModel, SiglipProcessor

class DenseSearchEngine:
    """CLIP/SigLIP Dense Feature Vector Similarity Search Engine with Kaggle Auto-Discovery."""
    def __init__(self, features_dir, model_name="openai/clip-vit-large-patch14"):
        self.features_dir = self._resolve_features_dir(features_dir)
        self.model_name = model_name
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.feature_files = []
        self._init_model()
        self._scan_features()

    def _resolve_features_dir(self, path):
        if path and os.path.exists(path):
            npy_files = glob.glob(os.path.join(path, "*.npy"))
            if npy_files:
                return path
        # Auto-Discovery on Kaggle / workspace
        search_roots = ["/kaggle/input", "data/features", "data"]
        for s_root in search_roots:
            if os.path.exists(s_root):
                for root, _, files in os.walk(s_root):
                    if any(f.endswith(".npy") for f in files):
                        print(f"[INFO] DenseSearchEngine: Auto-discovered features directory at '{root}'")
                        return root
        return path

    def _init_model(self):
        print(f"[INFO] DenseSearchEngine: Loading {self.model_name} on {self.device}...")
        if "siglip" in self.model_name.lower():
            self.processor = SiglipProcessor.from_pretrained(self.model_name)
            self.model = SiglipModel.from_pretrained(self.model_name).to(self.device)
        else:
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def _scan_features(self):
        if self.features_dir and os.path.exists(self.features_dir):
            self.feature_files = sorted(glob.glob(os.path.join(self.features_dir, "*.npy")))
            print(f"[INFO] DenseSearchEngine: Found {len(self.feature_files)} feature files in '{self.features_dir}'")
        else:
            print(f"[WARNING] DenseSearchEngine: Feature dir '{self.features_dir}' does not exist.")

    def unload(self):
        """Unloads CLIP model and processor from VRAM."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[INFO] DenseSearchEngine: Unloaded CLIP model from VRAM.")

    def search(self, golden_english_prompts, top_k=100):
        if not self.feature_files or not golden_english_prompts:
            return []

        prompt_list = golden_english_prompts if isinstance(golden_english_prompts, list) else [golden_english_prompts]
        
        try:
            # Enforce 77-token CLIP truncation limit
            inputs = self.processor(text=prompt_list, return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device)
            with torch.no_grad():
                if hasattr(self.model, "get_text_features"):
                    text_embeds = self.model.get_text_features(**inputs)
                else:
                    text_embeds = self.model(**inputs)
                    
                if not isinstance(text_embeds, torch.Tensor):
                    text_embeds = getattr(text_embeds, "text_embeds", getattr(text_embeds, "pooler_output", text_embeds[0]))
                    
                text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
                query_vec = text_embeds.mean(dim=0, keepdim=True)
                query_vec = (query_vec / query_vec.norm(p=2, dim=-1, keepdim=True)).cpu().numpy()
        except Exception as e:
            print(f"[WARNING] DenseSearchEngine search error ({e}).")
            return []

        results = []
        for fpath in self.feature_files:
            video_id = os.path.splitext(os.path.basename(fpath))[0]
            try:
                video_feats = np.load(fpath)
                if video_feats.ndim == 1:
                    video_feats = video_feats.reshape(1, -1)
                
                sims = np.dot(video_feats, query_vec.T).squeeze(axis=1)
                best_idx = int(np.argmax(sims))
                max_score = float(sims[best_idx])

                results.append({
                    "video_id": video_id,
                    "score": max_score,
                    "best_frame_idx": best_idx,
                    "all_scores": sims
                })
            except Exception:
                continue

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


class SparseSearchEngine:
    """BM25 Text Search Engine over Metadata and OCR text files with Auto-Discovery."""
    def __init__(self, metadata_dir, ocr_dir=None):
        self.metadata_dir = self._resolve_dir(metadata_dir, "metadata")
        self.ocr_dir = self._resolve_dir(ocr_dir or metadata_dir, "ocr")
        self.corpus = []
        self.video_ids = []
        self.bm25 = None
        self._build_index()

    def _resolve_dir(self, path, hint="metadata"):
        if path and os.path.exists(path):
            return path
        search_roots = ["/kaggle/input", "data/metadata", "data"]
        for s_root in search_roots:
            if os.path.exists(s_root):
                for root, dirs, files in os.walk(s_root):
                    if hint in root.lower() or any(f.endswith((".json", ".csv", ".txt")) for f in files):
                        return root
        return path

    def _build_index(self):
        from rank_bm25 import BM25Okapi
        
        meta_files = []
        if self.metadata_dir and os.path.exists(self.metadata_dir):
            meta_files = glob.glob(os.path.join(self.metadata_dir, "*.json")) + glob.glob(os.path.join(self.metadata_dir, "*.txt"))
            if not meta_files:
                for root, _, files in os.walk(self.metadata_dir):
                    for f in files:
                        if f.endswith(".json") or f.endswith(".txt"):
                            meta_files.append(os.path.join(root, f))

        meta_dict = {}
        for mf in meta_files:
            vid = os.path.splitext(os.path.basename(mf))[0]
            try:
                with open(mf, "r", encoding="utf-8") as f:
                    content = f.read()
                meta_dict[vid] = meta_dict.get(vid, "") + " " + content
            except Exception:
                pass

        if self.ocr_dir and os.path.exists(self.ocr_dir) and self.ocr_dir != self.metadata_dir:
            ocr_files = glob.glob(os.path.join(self.ocr_dir, "*.json")) + glob.glob(os.path.join(self.ocr_dir, "*.txt"))
            for of in ocr_files:
                vid = os.path.splitext(os.path.basename(of))[0]
                try:
                    with open(of, "r", encoding="utf-8") as f:
                        content = f.read()
                    meta_dict[vid] = meta_dict.get(vid, "") + " " + content
                except Exception:
                    pass

        self.video_ids = sorted(list(meta_dict.keys()))
        tokenized_corpus = [meta_dict[vid].lower().split() for vid in self.video_ids]
        
        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)
            print(f"[INFO] SparseSearchEngine: Indexed BM25 for {len(self.video_ids)} videos.")
        else:
            print("[WARNING] SparseSearchEngine: Corpus empty.")

    def search(self, keywords, top_k=100):
        if not self.bm25 or not keywords:
            return []

        query_tokens = [k.lower() for k in keywords]
        scores = self.bm25.get_scores(query_tokens)
        
        results = []
        for idx, score in enumerate(scores):
            if score > 0:
                results.append({
                    "video_id": self.video_ids[idx],
                    "score": float(score)
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


class ObjectSearchEngine:
    """Google OpenImages Object Detection Search Engine with Auto-Discovery."""
    def __init__(self, objects_dir):
        self.objects_dir = self._resolve_dir(objects_dir)

    def _resolve_dir(self, path):
        if path and os.path.exists(path):
            return path
        search_roots = ["/kaggle/input", "data/objects", "data"]
        for s_root in search_roots:
            if os.path.exists(s_root):
                for root, dirs, _ in os.walk(s_root):
                    if "object" in root.lower():
                        return root
        return path

    def get_frame_objects(self, video_id, frame_idx):
        """Resolves 3-digit, 4-digit, 5-digit, or raw frame object json files."""
        if not self.objects_dir or not os.path.exists(self.objects_dir):
            return []

        fname_3d = f"{frame_idx:03d}.json"
        fname_4d = f"{frame_idx:04d}.json"
        fname_5d = f"{frame_idx:05d}.json"
        fname_raw = f"{frame_idx}.json"

        level = video_id.split('_')[0] if '_' in video_id else ""
        candidates = [
            os.path.join(self.objects_dir, video_id, fname_3d),
            os.path.join(self.objects_dir, video_id, fname_4d),
            os.path.join(self.objects_dir, video_id, fname_5d),
            os.path.join(self.objects_dir, video_id, fname_raw),
            os.path.join(self.objects_dir, level, video_id, fname_3d),
            os.path.join(self.objects_dir, level, video_id, fname_4d)
        ]

        for c in candidates:
            if os.path.exists(c):
                try:
                    with open(c, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            return [item.get("detection_class_entities", "") for item in data if isinstance(item, dict)]
                        elif isinstance(data, dict):
                            return data.get("detection_class_entities", [])
                except Exception:
                    pass
        return []

    def score_video_objects(self, video_id, target_classes, best_frame_idx=0):
        if not target_classes:
            return 0.0
            
        detected = self.get_frame_objects(video_id, best_frame_idx)
        if not detected:
            return 0.0

        detected_lower = [d.lower() for d in detected]
        matches = 0
        for cls in target_classes:
            if cls.lower() in detected_lower:
                matches += 1

        return float(matches / max(1, len(target_classes)))


class GenericHybridSearcher:
    """Consolidated Self-Contained Search Engine with RRF Fusion and Guaranteed Fallback."""
    def __init__(self, config):
        self.config = config
        data_cfg = config.get("data", {})
        models_cfg = config.get("models", {})

        features_dir = data_cfg.get("features_dir", "")
        metadata_dir = data_cfg.get("metadata_dir", "")
        ocr_dir = data_cfg.get("ocr_dir", metadata_dir)
        objects_dir = data_cfg.get("objects_dir", "")
        clip_name = models_cfg.get("clip_model", "openai/clip-vit-large-patch14")

        self.dense_engine = DenseSearchEngine(features_dir, clip_name)
        self.sparse_engine = SparseSearchEngine(metadata_dir, ocr_dir)
        self.object_engine = ObjectSearchEngine(objects_dir)

    def _get_known_video_pool(self):
        pool = []
        if self.dense_engine.feature_files:
            pool.extend([os.path.splitext(os.path.basename(f))[0] for f in self.dense_engine.feature_files])
        if self.sparse_engine.video_ids:
            pool.extend(self.sparse_engine.video_ids)
        pool = sorted(list(set(pool)))
        if not pool:
            # Synthetic default IDs to guarantee minimum valid submission
            pool = [f"L21_V{i:03d}" for i in range(1, 101)]
        return pool

    def search_candidates(self, parsed_schema, top_k_videos=100):
        prompts = parsed_schema.get("golden_english_prompts", [])
        keywords = parsed_schema.get("bm25_keywords", [])
        classes = parsed_schema.get("openimages_classes", [])
        
        w_dense = float(parsed_schema.get("dense_weight", 0.75))
        w_sparse = float(parsed_schema.get("sparse_weight", 0.25))

        dense_res = self.dense_engine.search(prompts, top_k=top_k_videos * 2)
        sparse_res = self.sparse_engine.search(keywords, top_k=top_k_videos * 2)

        rrf_scores = {}
        candidate_info = {}

        for rank, item in enumerate(dense_res):
            vid = item["video_id"]
            score = w_dense * (1.0 / (60.0 + rank + 1))
            rrf_scores[vid] = rrf_scores.get(vid, 0.0) + score
            candidate_info[vid] = {"video_id": vid, "dense_info": item}

        for rank, item in enumerate(sparse_res):
            vid = item["video_id"]
            score = w_sparse * (1.0 / (60.0 + rank + 1))
            rrf_scores[vid] = rrf_scores.get(vid, 0.0) + score
            if vid not in candidate_info:
                candidate_info[vid] = {"video_id": vid, "dense_info": {}}

        for vid in list(candidate_info.keys()):
            dense_info = candidate_info[vid].get("dense_info", {})
            f_idx = dense_info.get("best_frame_idx", 0)
            obj_boost = self.object_engine.score_video_objects(vid, classes, best_frame_idx=f_idx)
            rrf_scores[vid] += (obj_boost * 0.05)

        sorted_vids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        final_candidates = []
        for vid in sorted_vids[:top_k_videos]:
            info = candidate_info[vid]
            info["rrf_score"] = rrf_scores[vid]
            final_candidates.append(info)

        # Fallback Guarantee: If search returned 0 candidates, populate from known video pool
        if not final_candidates:
            print(f"[WARNING] No candidates found for query '{parsed_schema.get('query_id', 'unknown')}'. Using fallback video pool.")
            pool = self._get_known_video_pool()
            for rank, vid in enumerate(pool[:top_k_videos]):
                final_candidates.append({
                    "video_id": vid,
                    "dense_info": {"best_frame_idx": rank % 100},
                    "rrf_score": 1.0 / (100.0 + rank)
                })

        return final_candidates

    def unload(self):
        """Unloads dense engine and cleans VRAM."""
        if hasattr(self, "dense_engine") and self.dense_engine is not None:
            self.dense_engine.unload()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
