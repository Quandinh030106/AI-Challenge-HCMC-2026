import yaml

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def normalize_query_item(raw_item):
    """
    Tu dong chuan hoa cac bien the ten truong (keys) bat ke BTC dat ten la gi:
    - query_id / qid / id / q_id / queryId -> 'query_id'
    - query / text / prompt / description / query_text / caption -> 'query'
    - question / q / qa_question / query_question -> 'question'
    - video_id / vid / video / videoId / video_name -> 'video_id'
    - events / sub_events / event_list / actions / stages -> 'events'
    """
    if not isinstance(raw_item, dict):
        return {"query_id": "unknown", "query": str(raw_item), "question": "", "video_id": "", "events": [], "raw": {}}
        
    query_id = (
        raw_item.get("query_id") or 
        raw_item.get("qid") or 
        raw_item.get("q_id") or 
        raw_item.get("id") or 
        raw_item.get("queryId") or 
        "unknown"
    )
    
    query_text = (
        raw_item.get("query") or 
        raw_item.get("text") or 
        raw_item.get("prompt") or 
        raw_item.get("description") or 
        raw_item.get("query_text") or 
        raw_item.get("caption") or 
        ""
    )
    
    question = (
        raw_item.get("question") or 
        raw_item.get("q") or 
        raw_item.get("qa_question") or 
        raw_item.get("query_question") or 
        ""
    )
    
    video_id = (
        raw_item.get("video_id") or 
        raw_item.get("vid") or 
        raw_item.get("video") or 
        raw_item.get("videoId") or 
        raw_item.get("video_name") or 
        ""
    )
    
    frame_start = (
        raw_item.get("frame_start") or 
        raw_item.get("start") or 
        raw_item.get("start_frame") or 
        raw_item.get("from") or 
        0
    )
    
    frame_end = (
        raw_item.get("frame_end") or 
        raw_item.get("end") or 
        raw_item.get("end_frame") or 
        raw_item.get("to") or 
        0
    )
    
    answer = (
        raw_item.get("answer") or 
        raw_item.get("ans") or 
        raw_item.get("ground_truth") or 
        raw_item.get("gt") or 
        ""
    )
    
    events_raw = (
        raw_item.get("events") or 
        raw_item.get("sub_events") or 
        raw_item.get("event_list") or 
        raw_item.get("actions") or 
        []
    )
    
    events = []
    events_dicts = []
    for ev in events_raw:
        if isinstance(ev, str):
            events.append(ev)
            events_dicts.append({"name": ev, "frame_start": 0, "frame_end": 0})
        elif isinstance(ev, dict):
            ev_name = ev.get("name") or ev.get("event_name") or ev.get("action") or ev.get("text") or ev.get("desc") or ""
            events.append(ev_name)
            events_dicts.append({
                "name": ev_name,
                "frame_start": ev.get("frame_start") or ev.get("start") or 0,
                "frame_end": ev.get("frame_end") or ev.get("end") or 0
            })
            
    return {
        "query_id": str(query_id),
        "query": str(query_text),
        "question": str(question),
        "video_id": str(video_id),
        "frame_start": frame_start,
        "frame_end": frame_end,
        "answer": str(answer),
        "events": events,
        "events_dicts": events_dicts,
        "raw": raw_item
    }
