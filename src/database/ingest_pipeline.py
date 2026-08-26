# ==============================================================================
# AIC 2026 - MULTIMODAL INGESTION PIPELINE FOR LANCEDB (2-TABLE ARCHITECTURE)
# ==============================================================================
import os
import glob
import json
import csv
import numpy as np
import pyarrow as pa
import lancedb
from tqdm import tqdm
from src.database.schema import get_videos_schema, get_keyframes_schema

class MultimodalIngestPipeline:
    """
    Automated pipeline that scans raw competition folders:
    (clip-features-32, map-keyframes, media-info, objects, keyframes)
    and constructs a normalized 2-table LanceDB store:
    - Table 1: `videos` (Video-level metadata, 1 row per video)
    - Table 2: `keyframes` (Frame-level visual features, captions, objects, exact frame IDs)
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
        
        self.videos_table_name = "videos"
        self.keyframes_table_name = "keyframes"

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
                        
                        header_offset = 1 if not rows[0][0].strip().isdigit() else 0
                        for row_idx in range(header_offset, len(rows)):
                            row = rows[row_idx]
                            idx_0based = row_idx - header_offset
                            
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

    def build_database(self, overwrite: bool = True) -> dict:
        """
        Executes full multimodal ingestion to construct the normalized 2-table LanceDB store:
        1. Table `videos` (Video-level metadata)
        2. Table `keyframes` (Frame-level visual vectors, captions, objects, exact frame IDs)
        """
        print(f"[INFO] IngestPipeline: Connecting to LanceDB at '{self.lancedb_uri}'...")
        os.makedirs(self.lancedb_uri, exist_ok=True)
        db = lancedb.connect(self.lancedb_uri)

        feature_files = sorted(glob.glob(os.path.join(self.features_dir, "*.npy")))
        if not feature_files:
            print(f"[ERROR] IngestPipeline: No .npy feature files found in '{self.features_dir}'!")
            return {}

        print(f"[INFO] IngestPipeline: Found {len(feature_files)} videos to ingest into 2-Table Store.")

        video_records = []
        keyframe_records = []

        # Auto-detect vector dimension from the first feature file
        detected_dim = 768
        for fpath in feature_files:
            try:
                sample_feats = np.load(fpath)
                if sample_feats.ndim == 1:
                    sample_feats = sample_feats.reshape(1, -1)
                detected_dim = int(sample_feats.shape[1])
                print(f"[INFO] IngestPipeline: Auto-detected Feature Vector Dimension = {detected_dim}-dim")
                break
            except Exception:
                pass

        videos_schema = get_videos_schema()
        keyframes_schema = get_keyframes_schema(vector_dim=detected_dim)

        for fpath in tqdm(feature_files, desc="Ingesting Data into 2-Table LanceDB Store"):
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

            # 1. Populate Video-Level Record (1 row per video)
            all_video_text = f"{title} {keywords} {description}".strip()
            video_records.append({
                "video_id": video_id,
                "video_title": title,
                "video_description": description,
                "video_keywords": keywords,
                "all_video_text": all_video_text
            })

            # Load cached captions dictionary if available
            cached_captions = {}
            caption_paths = [
                "data/keyframe_captions.json",
                "/kaggle/working/keyframe_captions.json",
                "/kaggle/input/keyframe-captions/keyframe_captions.json"
            ]
            for cp in caption_paths:
                if os.path.exists(cp):
                    try:
                        with open(cp, "r", encoding="utf-8") as f:
                            cached_captions = json.load(f)
                        print(f"[INFO] IngestPipeline: Loaded {len(cached_captions)} keyframe captions from '{cp}'.")
                        break
                    except Exception:
                        pass

            # 2. Populate Keyframe-Level Records
            n_frames = feats_normalized.shape[0]
            for f_idx in range(n_frames):
                row_vec = feats_normalized[f_idx]
                if len(row_vec) != detected_dim:
                    if len(row_vec) > detected_dim:
                        row_vec = row_vec[:detected_dim]
                    else:
                        row_vec = np.pad(row_vec, (0, detected_dim - len(row_vec)), mode='constant')
                
                vec = row_vec.tolist()
                
                frame_map_info = map_data.get(f_idx, {"frame_id": f_idx, "pts_time": 0.0})
                real_frame_id = int(frame_map_info.get("frame_id", f_idx))
                pts_time = float(frame_map_info.get("pts_time", 0.0))
                
                img_path = self._find_image_path(video_id, f_idx)
                obj_text = self._load_frame_objects(video_id, f_idx)
                
                # Fetch keyframe caption if available
                cap_data = cached_captions.get(img_path, {})
                kf_caption = cap_data.get("caption", "") if isinstance(cap_data, dict) else str(cap_data)
                kf_ocr = cap_data.get("ocr_text", "") if isinstance(cap_data, dict) else ""

                # Frame-level weighted text (caption + ocr + detected objects)
                frame_text_weighted = f"{kf_caption} {kf_ocr} {obj_text}".strip()

                keyframe_records.append({
                    "vector": vec,
                    "video_id": video_id,
                    "frame_idx": int(f_idx),
                    "frame_id": real_frame_id,
                    "pts_time": pts_time,
                    "image_path": img_path,
                    "keyframe_caption": kf_caption,
                    "detected_objects": obj_text,
                    "ocr_text": kf_ocr,
                    "text_genre": "GENERAL",
                    "frame_text_weighted": frame_text_weighted
                })

        print(f"[INFO] IngestPipeline: Building PyArrow Tables ({len(video_records)} videos, {len(keyframe_records)} keyframes)...")
        pa_videos_table = pa.Table.from_pylist(video_records, schema=videos_schema)
        pa_keyframes_table = pa.Table.from_pylist(keyframe_records, schema=keyframes_schema)

        # Build Table 1: videos
        if overwrite and self.videos_table_name in db.table_names():
            db.drop_table(self.videos_table_name)
        v_table = db.create_table(self.videos_table_name, pa_videos_table)

        # Build Table 2: keyframes
        if overwrite and self.keyframes_table_name in db.table_names():
            db.drop_table(self.keyframes_table_name)
        kf_table = db.create_table(self.keyframes_table_name, pa_keyframes_table)

        # Create Tantivy FTS Index on Table 1 (videos) and Table 2 (keyframes)
        try:
            print("[INFO] IngestPipeline: Creating Tantivy FTS Index on 'videos' table...")
            v_table.create_fts_index(["all_video_text", "video_title"])
        except Exception as e:
            print(f"[WARNING] FTS Index on 'videos' warning ({e})")

        try:
            print("[INFO] IngestPipeline: Creating Tantivy FTS Index on 'keyframes' table...")
            kf_table.create_fts_index(["frame_text_weighted", "detected_objects"])
        except Exception as e:
            print(f"[WARNING] FTS Index on 'keyframes' warning ({e})")

        print(f"[INFO] IngestPipeline: 2-Table Store built successfully at '{self.lancedb_uri}'!")
        return {"videos": v_table, "keyframes": kf_table}

if __name__ == "__main__":
    import yaml
    config_path = "configs/lancedb_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    pipeline = MultimodalIngestPipeline(cfg)
    tables = pipeline.build_database(overwrite=True)
    
    if tables:
        print("=" * 80)
        print("[INFO] VERIFICATION: LANCEDB 2-TABLE STORE INSPECTION REPORT")
        print("=" * 80)
        print(f"  - Table 'videos'    : {len(tables['videos'])} records")
        print(f"  - Table 'keyframes' : {len(tables['keyframes'])} records")
        print("=" * 80)

