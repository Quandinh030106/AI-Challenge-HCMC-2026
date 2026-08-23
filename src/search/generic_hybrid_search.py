import os
import glob
import json
import re
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from rank_bm25 import BM25Okapi
from transformers import (
    CLIPModel, CLIPProcessor, 
    SiglipModel, SiglipProcessor, 
    AutoModel, AutoProcessor
)

class DenseSearchEngine:
    """Sub-module Tìm kiếm Vector Mật độ cao (CLIP / EVA-CLIP Embedding Search)."""
    def __init__(self, model_name="openai/clip-vit-large-patch14", features_dir=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.features_dir = features_dir
        
        print(f"DenseSearchEngine: Khởi tạo mô hình {self.model_name} trên {self.device}...")
        try:
            if "siglip" in self.model_name.lower():
                self.processor = SiglipProcessor.from_pretrained(self.model_name)
                self.model = SiglipModel.from_pretrained(self.model_name).to(self.device)
            elif "clip" in self.model_name.lower():
                self.processor = CLIPProcessor.from_pretrained(self.model_name)
                self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            else:
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"DenseSearchEngine: Cảnh báo không nạp được {self.model_name} ({e}).")
            self.model = None
            self.processor = None
            
        self.global_tensor = None
        self.video_metadata_list = []
        self.all_video_ids = []
        self.video_features_dict = {}
        
        if self.features_dir:
            self.load_and_build_global_matrix()

    def load_and_build_global_matrix(self):
        """Nạp toàn bộ vector đặc trưng .npy vào GPU thành ma trận duy nhất."""
        if not self.features_dir or not os.path.exists(self.features_dir):
            return
            
        feature_files = sorted(glob.glob(os.path.join(self.features_dir, "**", "*.npy"), recursive=True))
        if not feature_files:
            return

        all_vectors = []
        current_idx = 0
        
        for file_path in feature_files:
            video_id = os.path.splitext(os.path.basename(file_path))[0]
            try:
                feats = np.load(file_path)
                norms = np.linalg.norm(feats, axis=-1, keepdims=True)
                norms[norms == 0] = 1e-10
                feats_norm = feats / norms
                
                n_frames = feats_norm.shape[0]
                self.video_features_dict[video_id] = feats_norm
                
                all_vectors.append(feats_norm)
                self.video_metadata_list.append({
                    "video_id": video_id,
                    "start_idx": current_idx,
                    "end_idx": current_idx + n_frames,
                    "n_frames": n_frames
                })
                self.all_video_ids.append(video_id)
                current_idx += n_frames
            except Exception:
                pass
                
        if all_vectors:
            concat_matrix = np.vstack(all_vectors)
            if self.device == "cuda":
                self.global_tensor = torch.from_numpy(concat_matrix).half().to(self.device)
            else:
                self.global_tensor = torch.from_numpy(concat_matrix).float()

    def search(self, prompt_list, top_k_videos=100):
        """Mã hóa danh sách Prompt Tiếng Anh và tìm kiếm tương đồng trên ma trận GPU."""
        if self.model is None or self.global_tensor is None:
            return []
            
        try:
            inputs = self.processor(text=prompt_list, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                if hasattr(self.model, "get_text_features"):
                    text_embeds = self.model.get_text_features(**inputs)
                else:
                    text_embeds = self.model.encode_text(inputs.input_ids)
                    
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            if self.device == "cuda":
                text_embeds = text_embeds.half()
                
            sim_matrix = torch.matmul(text_embeds, self.global_tensor.T)
            max_sim_per_frame, _ = torch.max(sim_matrix, dim=0)
            frame_scores = max_sim_per_frame.cpu().numpy()
            
            video_scores = []
            for meta in self.video_metadata_list:
                v_scores = frame_scores[meta["start_idx"]:meta["end_idx"]]
                if len(v_scores) > 0:
                    top_score = float(np.max(v_scores))
                    video_scores.append({
                        "video_id": meta["video_id"],
                        "score": top_score,
                        "dense_info": {
                            "all_scores": v_scores,
                            "best_frame_idx": int(np.argmax(v_scores)),
                            "max_score": top_score
                        }
                    })
                    
            video_scores.sort(key=lambda x: x["score"], reverse=True)
            return video_scores[:top_k_videos]
        except Exception as e:
            print(f"DenseSearchEngine Lỗi ({e}).")
            return []


class SparseSearchEngine:
    """Sub-module Tìm kiếm Từ khóa BM25 Mở trên Metadata & OCR Transcripts."""
    def __init__(self, metadata_dir=None, ocr_dir=None):
        self.metadata_dir = metadata_dir
        self.ocr_dir = ocr_dir
        self.bm25_model = None
        self.video_ids = []
        self.corpus_docs = []
        
        self._build_bm25_index()

    def _build_bm25_index(self):
        """Xây dựng chỉ mục BM25 mở từ file Metadata / OCR."""
        all_docs = []
        v_ids = []
        
        search_dirs = []
        if self.metadata_dir and os.path.exists(self.metadata_dir):
            search_dirs.append(self.metadata_dir)
        if self.ocr_dir and os.path.exists(self.ocr_dir):
            search_dirs.append(self.ocr_dir)
            
        json_files = []
        for d in search_dirs:
            json_files.extend(glob.glob(os.path.join(d, "**", "*.json"), recursive=True))
            json_files.extend(glob.glob(os.path.join(d, "**", "*.csv"), recursive=True))
            
        video_text_map = {}
        for jf in json_files:
            vid = os.path.splitext(os.path.basename(jf))[0]
            try:
                with open(jf, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().lower()
                tokens = [w.strip() for w in re.split(r'[,.\s\?\!\:\;\-\"\']+', content) if len(w.strip()) >= 2]
                video_text_map[vid] = video_text_map.get(vid, []) + tokens
            except Exception:
                pass
                
        for vid, tokens in video_text_map.items():
            if tokens:
                v_ids.append(vid)
                all_docs.append(tokens)
                
        if all_docs:
            self.video_ids = v_ids
            self.corpus_docs = all_docs
            self.bm25_model = BM25Okapi(all_docs)

    def search(self, sparse_text, top_k_videos=50):
        """Tìm kiếm từ khóa BM25."""
        if not self.bm25_model or not sparse_text:
            return []
            
        query_tokens = [w.strip() for w in re.split(r'[,.\s\?\!\:\;\-\"\']+', sparse_text.lower()) if len(w.strip()) >= 2]
        if not query_tokens:
            return []
            
        doc_scores = self.bm25_model.get_scores(query_tokens)
        results = []
        for idx, score in enumerate(doc_scores):
            if score > 0:
                results.append({
                    "video_id": self.video_ids[idx],
                    "score": float(score)
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k_videos]


class ObjectSearchEngine:
    """Sub-module Đọc & Thưởng điểm Vật thể từ OpenImages JSON (Hỗ trợ 001.json, 0001.json, 1.json)."""
    def __init__(self, objects_dir=None):
        self.objects_dir = objects_dir

    def get_frame_objects(self, video_id, frame_idx):
        """Đọc file JSON object của 1 frame hỗ trợ cả 001.json (3-digit) và 0001.json (4-digit)."""
        if not self.objects_dir or not os.path.exists(self.objects_dir):
            return []
            
        idx_3d = f"{frame_idx:03d}"
        idx_4d = f"{frame_idx:04d}"
        idx_5d = f"{frame_idx:05d}"
        idx_raw = str(frame_idx)
        idx_3d_1based = f"{frame_idx + 1:03d}"
        idx_4d_1based = f"{frame_idx + 1:04d}"

        candidate_paths = [
            os.path.join(self.objects_dir, video_id, f"{idx_3d}.json"),
            os.path.join(self.objects_dir, video_id, f"{idx_4d}.json"),
            os.path.join(self.objects_dir, video_id, f"{idx_3d_1based}.json"),
            os.path.join(self.objects_dir, video_id, f"{idx_4d_1based}.json"),
            os.path.join(self.objects_dir, video_id, f"{idx_raw}.json"),
            os.path.join(self.objects_dir, video_id, f"{idx_5d}.json"),
        ]

        for jp in candidate_paths:
            if os.path.exists(jp):
                try:
                    with open(jp, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    names = data.get("detection_class_names", [])
                    scores = data.get("detection_scores", [])
                    return [{"entity": n, "score": float(s)} for n, s in zip(names, scores)]
                except Exception:
                    pass
        return []


class GenericHybridSearcher:
    """
    MÔ-ĐUN TÌM KIẾM ĐA PHƯƠNG THỨC NGUYÊN KHỐI TỔNG QUÁT (TỰ CHỨA - SELF CONTAINED).
    Bao gồm Dense CLIP + Sparse BM25 + OpenImages Boosting + Dynamic RRF Fusion (100% Zero-Bias).
    """
    def __init__(self, config=None, dense_engine=None, sparse_engine=None, object_engine=None):
        self.config = config or {}
        
        model_name = self.config.get("models", {}).get("clip_model", "openai/clip-vit-large-patch14")
        features_dir = self.config.get("data", {}).get("features_dir", None)
        metadata_dir = self.config.get("data", {}).get("metadata_dir", None)
        ocr_dir = self.config.get("data", {}).get("ocr_dir", None)
        objects_dir = self.config.get("data", {}).get("objects_dir", None)
        
        self.dense_engine = dense_engine or DenseSearchEngine(model_name=model_name, features_dir=features_dir)
        self.sparse_engine = sparse_engine or SparseSearchEngine(metadata_dir=metadata_dir, ocr_dir=ocr_dir)
        self.object_engine = object_engine or ObjectSearchEngine(objects_dir=objects_dir)

    def search_candidates(self, parsed_schema, top_k_videos=100):
        """
        Thực thi tìm kiếm đa phương thức hoàn toàn dựa trên JSON Cấu trúc Ngữ nghĩa Động từ File 1.
        """
        golden_prompts = parsed_schema.get("golden_english_prompts", [])
        bm25_keywords = parsed_schema.get("bm25_keywords", [])
        open_classes = parsed_schema.get("openimages_classes", [])
        query_vi = parsed_schema.get("query_vi", "")

        dense_weight = parsed_schema.get("dense_weight")
        sparse_weight = parsed_schema.get("sparse_weight")
        
        if dense_weight is None or sparse_weight is None:
            dense_weight, sparse_weight = 0.75, 0.25

        dense_results = []
        if self.dense_engine and golden_prompts:
            dense_results = self.dense_engine.search(golden_prompts, top_k_videos=top_k_videos)
            dense_results = self._apply_gaussian_smoothing(dense_results)

        sparse_results = []
        if self.sparse_engine:
            sparse_text = " ".join(bm25_keywords) if bm25_keywords else query_vi
            sparse_results = self.sparse_engine.search(sparse_text, top_k_videos=top_k_videos // 2)

        fused_candidates = self._reciprocal_rank_fusion(
            dense_results, sparse_results,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight
        )

        if self.object_engine and open_classes:
            fused_candidates = self._boost_with_object_detection(
                fused_candidates, open_classes
            )

        return fused_candidates[:top_k_videos]

    def _apply_gaussian_smoothing(self, dense_results, sigma=1.5):
        """Làm mượt chuỗi điểm tương đồng theo thời gian để khử nhiễu đơn frame."""
        for cand in dense_results:
            dense_info = cand.get("dense_info")
            if dense_info and "all_scores" in dense_info:
                raw_scores = np.array(dense_info["all_scores"], dtype=np.float32)
                if len(raw_scores) > 3:
                    smoothed_scores = gaussian_filter1d(raw_scores, sigma=sigma)
                    dense_info["all_scores"] = smoothed_scores
                    dense_info["best_frame_idx"] = int(np.argmax(smoothed_scores))
                    dense_info["max_score"] = float(np.max(smoothed_scores))
        return dense_results

    def _reciprocal_rank_fusion(self, dense_list, sparse_list, dense_weight=0.75, sparse_weight=0.25, k=60):
        """Hòa trộn thứ hạng RRF với trọng số động từ LLM."""
        scores = {}
        candidate_map = {}

        for rank, cand in enumerate(dense_list):
            vid = cand["video_id"]
            rrf_score = dense_weight * (1.0 / (k + rank + 1))
            scores[vid] = scores.get(vid, 0.0) + rrf_score
            candidate_map[vid] = dict(cand)

        for rank, cand in enumerate(sparse_list):
            vid = cand["video_id"]
            rrf_score = sparse_weight * (1.0 / (k + rank + 1))
            scores[vid] = scores.get(vid, 0.0) + rrf_score
            if vid not in candidate_map:
                candidate_map[vid] = dict(cand)

        sorted_vids = sorted(scores.keys(), key=lambda v: scores[v], reverse=True)
        fused = []
        for v in sorted_vids:
            item = candidate_map[v]
            item["rrf_score"] = float(scores[v])
            fused.append(item)

        return fused

    def _boost_with_object_detection(self, candidates, openimages_classes):
        """Thưởng điểm tự động theo các lớp OpenImages mở mà LLM trích xuất."""
        if not openimages_classes or not candidates:
            return candidates

        target_classes_lower = {c.lower() for c in openimages_classes}
        
        for cand in candidates[:30]:
            dense_info = cand.get("dense_info")
            if not dense_info or "all_scores" not in dense_info:
                continue

            vid = cand["video_id"]
            scores = np.array(dense_info["all_scores"], dtype=np.float32)
            top_idxs = np.argsort(scores)[::-1][:10]

            match_count = 0
            for f_idx in top_idxs:
                objs = self.object_engine.get_frame_objects(vid, int(f_idx))
                if not objs:
                    continue
                for obj in objs:
                    ent_name = obj.get("entity", "").lower()
                    if ent_name in target_classes_lower or any(tc in ent_name for tc in target_classes_lower):
                        match_count += 1
                        scores[f_idx] *= 1.10

            if match_count > 0:
                cand["dense_info"]["all_scores"] = scores
                cand["dense_info"]["best_frame_idx"] = int(np.argmax(scores))
                cand["dense_info"]["max_score"] = float(np.max(scores))
                cand["rrf_score"] += 0.05 * min(match_count, 5)

        candidates.sort(key=lambda c: c.get("rrf_score", 0.0), reverse=True)
        return candidates
