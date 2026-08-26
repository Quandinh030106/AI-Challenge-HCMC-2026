import psycopg2
import json

class DatabaseSearcher:
    def __init__(self, db_config):
        self.conn = psycopg2.connect(**db_config)

    def search_candidates(self, query_embedding, tier1_objects=None, top_k=100):
        cur = self.conn.cursor()
        query_vec_str = str(query_embedding.tolist())
        
        # 1. Hard Filter theo Tier 1 Objects (nếu có)
        where_clause = ""
        if tier1_objects and len(tier1_objects) > 0:
            json_objs = json.dumps(tier1_objects)
            where_clause = f"WHERE objects_tier1 @> '{json_objs}'::jsonb"

        # 2. Truy vấn kết hợp Filter & Cosine Distance trên DB
        sql = f"""
        SELECT 
            frame_id, 
            video_id,
            (1 - (clip_embedding <=> %s::vector)) AS dense_score
        FROM keyframe_features
        {where_clause}
        ORDER BY dense_score DESC
        LIMIT %s;
        """
        
        cur.execute(sql, (query_vec_str, top_k))
        results = cur.fetchall()
        cur.close()
        
        return results

    def close(self):
        self.conn.close()
