# ==============================================================================
# AIC 2026 - MULTIMODAL INGESTION PIPELINE FOR LANCEDB
# ==============================================================================
import os
import glob
import json
import csv
import numpy as np
import pyarrow as pa
import lancedb
from tqdm import tqdm
from src.database.schema import get_aic_master_schema

class MultimodalIngestPipeline:
    """
    Automated pipeline that scans raw competition folders:
    (clip-features-32, map-keyframes, media-info, objects, keyframes)
    and constructs a unified, indexed LanceDB multimodal table.
    """
    def __init__(self, config: dict):
        self.config = config
        data_cfg = config.get("data", {})
        
        self.features_dir = self._resolve_dir(data_cfg.get("features_dir", ""), "features", ".npy")
        self.keyframes_dir = self._resolve_dir(data_cfg.get("keyframes_dir", ""), "keyframes")
        self.map_keyframes_dir = self._resolve_dir(data_cfg.get("map_keyframes_dir", ""), "map-keyframes", ".csv")
        self.media_info_dir = self._resolve_dir(data_cfg.get("media_info_dir", ""), "media-info", ".json")
        self.objects_dir = self._resolve_dir(data_cfg.get("objects_dir", ""), "objects")
        self.lancedb_uri = data_cfg.get("lancedb_uri", "data/aic_lancedb")
        self.table_name = "aic_master_table"

    def _resolve_dir(self, path: str, hint: str = "", ext_hint: str = None) -> str:
        """Auto-resolves actual directory path across Kaggle and local environments."""
        if path and os.path.exists(path):
            if ext_hint:
                if glob.glob(os.path.join(path, f"*{ext_hint}")):
                    return path
            else:
                return path

        search_roots = ["/kaggle/input", "data", "."]
        for s_root in search_roots:
            if os.path.exists(s_root):
                for root, dirs, files in os.walk(s_root):
                    if hint.lower() in root.lower():
                        if ext_hint:
                            if any(f.endswith(ext_hint) for f in files):
                                print(f"[INFO] IngestPipeline: Auto-discovered '{hint}' at '{root}'")
                                return root
                        else:
                            print(f"[INFO] IngestPipeline: Auto-discovered '{hint}' at '{root}'")
                            return root
        return path

    def _load_media_info(self, video_id: str) -> dict:
        """Loads and extracts video title, description, and keywords."""
        res = {"title": "", "description": "", "keywords": ""}
        if not self.media_info_dir or not os.path.exists(self.media_info_dir):
            return res

        candidates = [
            os.path.join(self.media_info_dir, f"{video_id}.json"),
            os.path.join(self.media_info_dir, "media-info", f"{video_id}.json")
        ]
        for c in candidates:
            if os.path.exists(c):
                try:
                    with open(c, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        root_data = data.get("root", data) if isinstance(data, dict) else data
                        if isinstance(root_data, dict):
                            res["title"] = str(root_data.get("title", "")).strip()
                            res["description"] = str(root_data.get("description", "")).strip()
                            raw_kw = root_data.get("keywords", [])
                            if isinstance(raw_kw, list):
                                res["keywords"] = " ".join([str(k).strip() for k in raw_kw if k])
                            elif isinstance(raw_kw, str):
                                res["keywords"] = raw_kw.strip()
                        return res
                except Exception:
                    pass
        return res

    def _load_map_keyframes(self, video_id: str) -> dict:
        """
        Parses map-keyframes CSV file for exact frame ID and timestamp mapping.
        Returns dict: { frame_idx_0based: {'frame_id': int, 'pts_time': float} }
        """
        mapping = {}
        if not self.map_keyframes_dir or not os.path.exists(self.map_keyframes_dir):
            return mapping

        candidates = [
            os.path.join(self.map_keyframes_dir, f"{video_id}.csv"),
            os.path.join(self.map_keyframes_dir, "map-keyframes", f"{video_id}.csv")
        ]
        for c in candidates:
            if os.path.exists(c):
                try:
                    with open(c, "r", encoding="utf-8") as f:
                        reader = csv.reader(f)
                        rows = [r for r in reader if r]
                        if not rows:
                            continue
                        
                        # Detect header row
                        header_offset = 1 if not rows[0][0].strip().isdigit() else 0
                        for row_idx in range(header_offset, len(rows)):
                            row = rows[row_idx]
                            idx_0based = row_idx - header_offset
                            
                            # Standard format: n, pts_time, fps, frame_idx
                            pts_time = 0.0
                            frame_id = idx_0based
                            if len(row) >= 4:
                                try:
                                    pts_time = float(row[1].strip())
                                    frame_id = int(float(row[3].strip()))
                                except Exception:
                                    pass
                            elif len(row) >= 2:
                                try:
                                    pts_time = float(row[0].strip())
                                    frame_id = int(float(row[1].strip()))
                                except Exception:
                                    pass
                            
                            mapping[idx_0based] = {
                                "frame_id": frame_id,
                                "pts_time": pts_time
                            }
                        return mapping
                except Exception:
                    pass
        return mapping

    def _load_frame_objects(self, video_id: str, frame_idx_0based: int) -> str:
        """Extracts high-confidence OpenImages objects (score >= 0.10) for target frame."""
        if not self.objects_dir or not os.path.exists(self.objects_dir):
            return ""

        f_1based = frame_idx_0based + 1
        name_patterns = [
            f"{f_1based:03d}.json", f"{f_1based:04d}.json", f"{f_1based:05d}.json", f"{f_1based}.json",
            f"{frame_idx_0based:03d}.json", f"{frame_idx_0based:04d}.json", f"{frame_idx_0based}.json"
        ]

        for np_name in name_patterns:
            c_path = os.path.join(self.objects_dir, video_id, np_name)
            if not os.path.exists(c_path):
                level = video_id.split('_')[0] if '_' in video_id else ""
                c_path = os.path.join(self.objects_dir, level, video_id, np_name)

            if os.path.exists(c_path):
                try:
                    with open(c_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        root_data = data.get("root", data) if isinstance(data, dict) else data
                        if isinstance(root_data, dict):
                            scores = root_data.get("detection_scores", [])
                            entities = root_data.get("detection_class_entities", [])
                            
                            filtered_entities = []
                            for s, ent in zip(scores, entities):
                                try:
                                    if float(s) >= 0.10 and ent:
                                        filtered_entities.append(str(ent).strip().lower())
                                except Exception:
                                    pass
                            
                            # Deduplicate preserving order
                            seen = set()
                            unique_entities = [x for x in filtered_entities if not (x in seen or seen.add(x))]
                            return ", ".join(unique_entities)
                except Exception:
                    pass
        return ""

    def _find_image_path(self, video_id: str, frame_idx_0based: int) -> str:
        """Finds physical keyframe image path."""
        if not self.keyframes_dir or not os.path.exists(self.keyframes_dir):
            return ""

        f_1based = frame_idx_0based + 1
        name_patterns = [
            f"{f_1based:03d}.jpg", f"{f_1based:04d}.jpg", f"{f_1based:05d}.jpg", f"{f_1based}.jpg",
            f"{frame_idx_0based:03d}.jpg", f"{frame_idx_0based:04d}.jpg", f"{frame_idx_0based}.jpg"
        ]

        level = video_id.split('_')[0] if '_' in video_id else ""
        for np_name in name_patterns:
            candidates = [
                os.path.join(self.keyframes_dir, f"Keyframes_{level}", "keyframes", video_id, np_name),
                os.path.join(self.keyframes_dir, f"Keyframes_{level}", video_id, np_name),
                os.path.join(self.keyframes_dir, level, "keyframes", video_id, np_name),
                os.path.join(self.keyframes_dir, "keyframes", video_id, np_name),
                os.path.join(self.keyframes_dir, video_id, np_name)
            ]
            for c in candidates:
                if os.path.exists(c):
                    return c
        return ""

    def build_database(self, overwrite: bool = True) -> lancedb.table.Table:
        """
        Executes full multimodal ingestion, normalizes vectors,
        constructs PyArrow Table, and persists to LanceDB.
        """
        print(f"[INFO] IngestPipeline: Connecting to LanceDB at '{self.lancedb_uri}'...")
        os.makedirs(self.lancedb_uri, exist_ok=True)
        db = lancedb.connect(self.lancedb_uri)

        feature_files = sorted(glob.glob(os.path.join(self.features_dir, "*.npy")))
        if not feature_files:
            print(f"[ERROR] IngestPipeline: No .npy feature files found in '{self.features_dir}'!")
            return None

        print(f"[INFO] IngestPipeline: Found {len(feature_files)} videos to ingest.")

        records = []
        schema = get_aic_master_schema(vector_dim=768)

        for fpath in tqdm(feature_files, desc="Ingesting Multimodal Data into LanceDB"):
            video_id = os.path.splitext(os.path.basename(fpath))[0]
            
            try:
                feats = np.load(fpath)
                if feats.ndim == 1:
                    feats = feats.reshape(1, -1)
            except Exception as e:
                print(f"[WARNING] IngestPipeline: Failed to load '{fpath}' ({e})")
                continue

            # L2 normalize feature vectors
            norms = np.linalg.norm(feats, axis=1, keepdims=True)
            norms[norms == 0] = 1e-12
            feats_normalized = (feats / norms).astype(np.float32)

            media_info = self._load_media_info(video_id)
            map_data = self._load_map_keyframes(video_id)
            
            title = media_info.get("title", "")
            description = media_info.get("description", "")
            keywords = media_info.get("keywords", "")

            n_frames = feats_normalized.shape[0]
            for f_idx in range(n_frames):
                vec = feats_normalized[f_idx].tolist()
                
                frame_map_info = map_data.get(f_idx, {"frame_id": f_idx, "pts_time": 0.0})
                real_frame_id = int(frame_map_info.get("frame_id", f_idx))
                pts_time = float(frame_map_info.get("pts_time", 0.0))
                
                img_path = self._find_image_path(video_id, f_idx)
                obj_text = self._load_frame_objects(video_id, f_idx)
                
                # Construct weighted text for BM25 (Title x3, Keywords x2, OCR/Objects x1.5, Desc x1)
                all_text = f"{title} {title} {title} {keywords} {keywords} {obj_text} {obj_text} {description}".strip()

                records.append({
                    "vector": vec,
                    "video_id": video_id,
                    "frame_idx": int(f_idx),
                    "frame_id": real_frame_id,
                    "pts_time": pts_time,
                    "image_path": img_path,
                    "detected_objects": obj_text,
                    "ocr_text": "",
                    "video_title": title,
                    "video_description": description,
                    "video_keywords": keywords,
                    "all_text_weighted": all_text
                })

        print(f"[INFO] IngestPipeline: Building PyArrow Table with {len(records)} keyframe rows...")
        pa_table = pa.Table.from_pylist(records, schema=schema)

        if overwrite and self.table_name in db.table_names():
            db.drop_table(self.table_name)

        print(f"[INFO] IngestPipeline: Creating LanceDB table '{self.table_name}'...")
        table = db.create_table(self.table_name, pa_table)

        # Create Tantivy BM25 Full-Text Search Index on text fields
        try:
            print("[INFO] IngestPipeline: Creating Tantivy BM25 Full-Text Index on 'all_text_weighted'...")
            table.create_fts_index(["all_text_weighted", "detected_objects", "video_title"])
            print("[INFO] IngestPipeline: Full-Text Index created successfully.")
        except Exception as e:
            print(f"[WARNING] IngestPipeline: Tantivy FTS Index warning ({e}). Proceeding...")

        print(f"[INFO] IngestPipeline: Ingestion completed! LanceDB stored at '{self.lancedb_uri}'.")
        return table

if __name__ == "__main__":
    import yaml
    config_path = "configs/lancedb_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    pipeline = MultimodalIngestPipeline(cfg)
    pipeline.build_database(overwrite=True)
