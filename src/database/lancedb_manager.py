# ==============================================================================
# AIC 2026 - LANCEDB MANAGER INTERFACE
# ==============================================================================
import os
import lancedb
import numpy as np
import pyarrow as pa
from typing import List, Dict, Any, Optional

class LanceDBManager:
    """
    Unified manager interface for LanceDB Multimodal Table.
    Provides optimized APIs for Vector Search, BM25 Text Search,
    Object Filtering, and Keyframe Image Retrieval.
    """
    def __init__(self, db_uri: str = "data/aic_lancedb", table_name: str = "aic_master_table"):
        self.db_uri = self._resolve_db_uri(db_uri)
        self.table_name = table_name
        self.db = None
        self.table = None
        self._connect()

    def _resolve_db_uri(self, uri: str) -> str:
        if uri and os.path.exists(uri):
            return uri
        search_roots = ["/kaggle/input", "/kaggle/working", "data", "."]
        for s_root in search_roots:
            if os.path.exists(s_root):
                for root, dirs, _ in os.walk(s_root):
                    if "lancedb" in root.lower() or "aic_master_table" in dirs:
                        print(f"[INFO] LanceDBManager: Auto-discovered database at '{root}'")
                        return root
        return uri

    def _connect(self):
        if not os.path.exists(self.db_uri):
            print(f"[WARNING] LanceDBManager: Database path '{self.db_uri}' does not exist yet.")
            return

        try:
            self.db = lancedb.connect(self.db_uri)
            if self.table_name in self.db.table_names():
                self.table = self.db.open_table(self.table_name)
                print(f"[INFO] LanceDBManager: Connected to table '{self.table_name}' ({len(self.table)} keyframes).")
            else:
                print(f"[WARNING] LanceDBManager: Table '{self.table_name}' not found in '{self.db_uri}'.")
        except Exception as e:
            print(f"[ERROR] LanceDBManager connection failed: {e}")

    def is_ready(self) -> bool:
        return self.table is not None

    def search_vector(self, query_vector: np.ndarray, top_k: int = 200, filter_sql: Optional[str] = None) -> List[Dict[str, Any]]:
        """Searches nearest neighbor keyframe vectors using cosine distance."""
        if self.table is None:
            return []

        if query_vector.ndim > 1:
            query_vector = query_vector.squeeze()

        query_vec_list = query_vector.tolist()
        query = self.table.search(query_vec_list).metric("cosine").limit(top_k)
        
        if filter_sql:
            query = query.where(filter_sql)

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
        Executes LanceDB Native Hybrid Search combining CLIP dense vector and Tantivy BM25.
        """
        if self.table is None:
            return []

        if query_vector.ndim > 1:
            query_vector = query_vector.squeeze()

        query_vec_list = query_vector.tolist()
        
        try:
            query = self.table.search(query_type="hybrid").vector(query_vec_list).text(text_keywords).limit(top_k)
            if filter_sql:
                query = query.where(filter_sql)
            df = query.to_pandas()
            return df.to_dict(orient="records")
        except Exception as e:
            # Fallback to vector search if full-text index is not available
            return self.search_vector(query_vector, top_k=top_k, filter_sql=filter_sql)

    def fetch_video_timeline(self, video_id: str) -> List[Dict[str, Any]]:
        """Retrieves all chronological keyframes of a video for TRAKE and temporal reasoning."""
        if self.table is None:
            return []

        try:
            df = self.table.search().where(f"video_id = '{video_id}'").limit(5000).to_pandas()
            if not df.empty:
                df = df.sort_values(by="frame_idx", ascending=True)
                return df.to_dict(orient="records")
        except Exception:
            pass
        return []

    def fetch_frames_by_indices(self, video_id: str, frame_indices: List[int]) -> List[Dict[str, Any]]:
        """Fetches exact keyframe image paths and real frame IDs for target frame indices."""
        if self.table is None or not frame_indices:
            return []

        indices_str = ", ".join([str(i) for i in frame_indices])
        try:
            df = self.table.search().where(f"video_id = '{video_id}' AND frame_idx IN ({indices_str})").limit(len(frame_indices) + 5).to_pandas()
            if not df.empty:
                df = df.sort_values(by="frame_idx", ascending=True)
                return df.to_dict(orient="records")
        except Exception:
            pass
        return []
