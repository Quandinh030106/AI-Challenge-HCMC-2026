# Sparse Search module using BM25 on Metadata and OCR text
class SparseSearcher:
    def __init__(self, config):
        self.config = config
        
    def search(self, query_text):
        # Return top matching video_ids and frame_ids based on text search
        print(f"Searching text index for: '{query_text}'")
        return []
