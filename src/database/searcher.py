import json
from .config import get_connection
from .schema import init_db_schema

class DatabaseSearcher:
    def __init__(self, db_config):
        self.conn = get_connection(db_config)
        init_db_schema(self.conn)

    def search_candidates(self, query_embedding, category=None, tier1_objects=None, top_k=100):
        cur = self.conn.cursor()
        query_vec_str = str(query_embedding.tolist())
        conditions = []
        params = [query_vec_str]
        if category:
            conditions.append("v.category = %s")
            params.append(category)
        if tier1_objects and len(tier1_objects) > 0:
            conditions.append("k.objects_tier1 @> %s::jsonb")
            params.append(json.dumps(tier1_objects))
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        sql = f"""
        SELECT 
            k.frame_id, 
            k.video_id,
            v.title,
            v.video_path,
            k.frame_prompt,
            (1 - (k.clip_embedding <=> %s::vector)) AS dense_score
        FROM keyframes k
        JOIN videos v ON k.video_id = v.video_id
        {where_clause}
        ORDER BY dense_score DESC
        LIMIT %s;
        """
        params.append(top_k)
        cur.execute(sql, tuple(params))
        results = cur.fetchall()
        cur.close()
        return results

    def close(self):
        self.conn.close()
