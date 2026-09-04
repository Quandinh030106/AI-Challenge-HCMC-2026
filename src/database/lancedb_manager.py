# ==============================================================================
# AIC 2026 - LANCEDB MANAGER INTERFACE FOR 2-TABLE STORE
# ==============================================================================
import os
import lancedb
import numpy as np
import pyarrow as pa
from typing import List, Dict, Any, Optional

class LanceDBManager:
    """
    Unified manager interface for Normalized 2-Table LanceDB Store:
    - Table 1: `videos` (Video-level metadata)
    - Table 2: `keyframes` (Frame-level visual vectors, captions, objects, exact frame IDs)
    """
    def __init__(self, db_uri: str = "data/aic_lancedb"):
        self.db_uri = self._resolve_db_uri(db_uri)
        self.db = None
        self.videos_table = None
        self.keyframes_table = None
        self._connect()

    def _resolve_db_uri(self, uri: str) -> str:
        if uri and os.path.exists(uri):
            # Check if nested aic_lancedb exists inside uri
            nested = os.path.join(uri, "aic_lancedb")
            if os.path.exists(os.path.join(nested, "keyframes.lance")) or (os.path.exists(nested) and not os.path.exists(os.path.join(uri, "keyframes.lance"))):
                print(f"[INFO] LanceDBManager: Auto-unwrapped nested database at '{nested}'")
                return nested
            return uri

        # Check known candidate locations with nesting
        candidates = [
            "/kaggle/input/aic-db-models/dataset/aic_lancedb/aic_lancedb",
            "/kaggle/input/datasets/quninhphmanh/aic-db-models/dataset/aic_lancedb/aic_lancedb",
            "/kaggle/input/aic-db-models/dataset/aic_lancedb",
            "/kaggle/input/datasets/quninhphmanh/aic-db-models/dataset/aic_lancedb",
            "/kaggle/working/aic_lancedb",
            "data/aic_lancedb"
        ]
        for c in candidates:
            if os.path.exists(c):
                nested = os.path.join(c, "aic_lancedb")
                if os.path.exists(os.path.join(nested, "keyframes.lance")) or (os.path.exists(nested) and not os.path.exists(os.path.join(c, "keyframes.lance"))):
                    print(f"[INFO] LanceDBManager: Auto-discovered nested database at '{nested}'")
                    return nested
                print(f"[INFO] LanceDBManager: Auto-discovered database at '{c}'")
                return c

        import glob
        discovered = glob.glob("/kaggle/input/**/keyframes.lance", recursive=True)
        if discovered:
            parent_dir = os.path.dirname(discovered[0])
            print(f"[INFO] LanceDBManager: Auto-discovered keyframes table at '{parent_dir}'")
            return parent_dir

        return uri

    def _connect(self):
        if not os.path.exists(self.db_uri):
            print(f"[WARNING] LanceDBManager: Database path '{self.db_uri}' does not exist yet.")
            return

        try:
            self.db = lancedb.connect(self.db_uri)
            try:
                tbl_names = self.db.table_names()
                print(f"[INFO] LanceDBManager: Tables in '{self.db_uri}': {tbl_names}")
            except Exception:
                tbl_names = []
            
            for t_cand in ["keyframes", "keyframes.lance", "aic_master_table"]:
                if t_cand in tbl_names:
                    try:
                        self.keyframes_table = self.db.open_table(t_cand)
                        print(f"[INFO] LanceDBManager: Connected to keyframes table '{t_cand}' ({len(self.keyframes_table)} rows).")
                        break
                    except Exception:
                        pass

            # Fallback: attempt direct open if table_names() did not catch it
            if self.keyframes_table is None:
                for t_name in ["keyframes", "aic_master_table"]:
                    try:
                        self.keyframes_table = self.db.open_table(t_name)
                        print(f"[INFO] LanceDBManager: Directly opened '{t_name}' table ({len(self.keyframes_table)} rows).")
                        break
                    except Exception:
                        pass

            for v_cand in ["videos", "videos.lance"]:
                if v_cand in tbl_names:
                    try:
                        self.videos_table = self.db.open_table(v_cand)
                        print(f"[INFO] LanceDBManager: Connected to videos table '{v_cand}' ({len(self.videos_table)} rows).")
                        break
                    except Exception:
                        pass
        except Exception as e:
            print(f"[ERROR] LanceDBManager connection failed: {e}")

    def is_ready(self) -> bool:
        return self.keyframes_table is not None

    def search_vector(self, query_vector: np.ndarray, top_k: int = 200, filter_sql: Optional[str] = None) -> List[Dict[str, Any]]:
        """Searches nearest neighbor keyframe vectors using cosine distance."""
        if self.keyframes_table is None:
            return []

        if query_vector.ndim > 1:
            query_vector = query_vector.squeeze()

        query_vec_list = query_vector.tolist()
        try:
            query = self.keyframes_table.search(query_vec_list, vector_column_name="vector").metric("cosine").limit(top_k)
        except Exception:
            query = self.keyframes_table.search(query_vec_list).metric("cosine").limit(top_k)
        
        if filter_sql:
            query = query.where(filter_sql)

        # PyArrow column projection: load ONLY required lightweight columns to save RAM
        try:
            df = query.select([
                "video_id", "frame_idx", "frame_id", "pts_time", 
                "timestamp_formatted", "image_path", "keyframe_caption", 
                "detected_objects", "ocr_text", "_distance"
            ]).to_pandas()
        except Exception:
            df = query.to_pandas()

        return df.to_dict(orient="records")

    def search_hybrid(
        self,
        query_vector: np.ndarray,
        text_keywords: str,
        filter_sql: Optional[str] = None,
        top_k: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Executes LanceDB Native Hybrid Search with safe fallback to vector search.
        """
        if self.keyframes_table is None:
            return []

        if query_vector.ndim > 1:
            query_vector = query_vector.squeeze()

        query_vec_list = query_vector.tolist()

        if text_keywords and str(text_keywords).strip():
            try:
                query = self.keyframes_table.search(query_type="hybrid").vector(query_vec_list).text(text_keywords).limit(top_k)
                if filter_sql:
                    query = query.where(filter_sql)
                df = query.to_pandas()
                return df.to_dict(orient="records")
            except Exception:
                pass
        
        # Fallback to vector search
        return self.search_vector(query_vector, top_k=top_k, filter_sql=filter_sql)

    def fetch_video_timeline(self, video_id: str) -> List[Dict[str, Any]]:
        """Retrieves all chronological keyframes of a video for TRAKE and temporal reasoning."""
        if self.keyframes_table is None:
            return []

        try:
            df = self.keyframes_table.search().where(f"video_id = '{video_id}'").limit(5000).to_pandas()
            if not df.empty:
                df = df.sort_values(by="frame_idx", ascending=True)
                return df.to_dict(orient="records")
        except Exception:
            pass
        return []

    def fetch_frames_by_indices(self, video_id: str, frame_indices: List[int]) -> List[Dict[str, Any]]:
        """Fetches exact keyframe image paths and real frame IDs for target frame indices."""
        if self.keyframes_table is None or not frame_indices:
            return []

        indices_str = ", ".join([str(i) for i in frame_indices])
        try:
            df = self.keyframes_table.search().where(f"video_id = '{video_id}' AND frame_idx IN ({indices_str})").limit(len(frame_indices) + 5).to_pandas()
            if not df.empty:
                df = df.sort_values(by="frame_idx", ascending=True)
                return df.to_dict(orient="records")
        except Exception:
            pass
        return []

