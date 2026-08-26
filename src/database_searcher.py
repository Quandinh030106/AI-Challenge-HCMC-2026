import psycopg2
from psycopg2.extras import execute_values

class DatabaseManager:
    def __init__(self, db_config):
        self.conn = psycopg2.connect(**db_config)
        self.cursor = self.conn.cursor()

    def search_candidates(self, query_vector, category=None, target_objects=None, top_k=10):
        """
        Truy vấn kết hợp Filter Metadata (Bảng 1 & 2) và Similarity Search (Vector)
        """
        query = """
            SELECT 
                k.frame_id, 
                k.video_id, 
                v.title, 
                v.video_path, 
                k.prompt_description,
                1 - (k.embedding <=> %s::vector) AS similarity_score
            FROM keyframes k
            JOIN videos v ON k.video_id = v.video_id
            WHERE 1=1
        """
        params = [query_vector.tolist()]

        # Filter theo thể loại video (Bảng 1)
        if category:
            query += " AND v.category = %s"
            params.append(category)

        # Filter theo objects trong frame (Bảng 2)
        if target_objects:
            query += " AND k.detected_objects && %s"
            params.append(target_objects)

        query += " ORDER BY k.embedding <=> %s::vector LIMIT %s;"
        params.extend([query_vector.tolist(), top_k])

        self.cursor.execute(query, params)
        return self.cursor.fetchall()