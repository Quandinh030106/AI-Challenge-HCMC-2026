import json
import yaml
import os
import re
from pathlib import Path


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



_MEDIA_LIST_CACHE = {}
_VIDEO_DIR_CACHE = {}

def natural_sort_key(value):
    name = os.path.basename(str(value))
    stem = os.path.splitext(name)[0]
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", stem)]

def _find_video_dir(root_dir, video_id):
    cache_key = (os.path.abspath(str(root_dir)), str(video_id))
    if cache_key in _VIDEO_DIR_CACHE:
        return _VIDEO_DIR_CACHE[cache_key]

    root_dir = str(root_dir)
    level = video_id.split("_")[0] if "_" in video_id else ""

    candidates = [
        os.path.join(root_dir, f"Keyframes_{level}", "keyframes", video_id),
        os.path.join(root_dir, f"Keyframes_{level}", video_id),
        os.path.join(root_dir, level, "keyframes", video_id),
        os.path.join(root_dir, "keyframes", video_id),
        os.path.join(root_dir, video_id),
    ]

    for folder in candidates:
        if os.path.isdir(folder):
            _VIDEO_DIR_CACHE[cache_key] = folder
            return folder

    if os.path.isdir(root_dir):
        for current_root, dirs, _ in os.walk(root_dir):
            if video_id in dirs:
                folder = os.path.join(current_root, video_id)
                _VIDEO_DIR_CACHE[cache_key] = folder
                return folder

    _VIDEO_DIR_CACHE[cache_key] = None
    return None

def list_keyframe_files(keyframes_dir, video_id):
    cache_key = ("keyframes", os.path.abspath(str(keyframes_dir)), str(video_id))
    if cache_key in _MEDIA_LIST_CACHE:
        return _MEDIA_LIST_CACHE[cache_key]

    folder = _find_video_dir(keyframes_dir, video_id)
    if not folder:
        _MEDIA_LIST_CACHE[cache_key] = []
        return []

    files = []
    for name in os.listdir(folder):
        p = os.path.join(folder, name)
        if os.path.isfile(p) and os.path.splitext(name)[1].lower() in {".jpg", ".jpeg", ".png"}:
            files.append(p)

    files.sort(key=natural_sort_key)
    _MEDIA_LIST_CACHE[cache_key] = files
    return files

def get_keyframe_path_by_index(keyframes_dir, video_id, keyframe_idx):
    files = list_keyframe_files(keyframes_dir, video_id)
    idx = int(keyframe_idx)
    if idx < 0 or idx >= len(files):
        raise IndexError(
            f"Keyframe index out of range: video={video_id}, idx={idx}, count={len(files)}"
        )
    return files[idx]

def list_json_files_in_video_dir(root_dir, video_id):
    cache_key = ("json", os.path.abspath(str(root_dir)), str(video_id))
    if cache_key in _MEDIA_LIST_CACHE:
        return _MEDIA_LIST_CACHE[cache_key]

    folder = _find_video_dir(root_dir, video_id)
    if not folder:
        _MEDIA_LIST_CACHE[cache_key] = []
        return []

    files = [
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.lower().endswith(".json")
        and os.path.isfile(os.path.join(folder, name))
    ]
    files.sort(key=natural_sort_key)
    _MEDIA_LIST_CACHE[cache_key] = files
    return files


_map_keyframes_rows_cache = {}


def load_map_keyframes_rows(metadata_dir, video_id):
    """
    Doc toan bo Map-Keyframes CSV cho 1 video, tra ve list dict THEO DUNG
    THU TU keyframe ordinal (khop voi feature .npy va thu muc keyframes):
        {"keyframe_ordinal": int, "n": int, "pts_time": float,
         "fps": float, "frame_idx": int}

    Dung chung schema (n, pts_time, fps, frame_idx) voi
    TemporalRefiner._load_map_rows. Khong tao du lieu gia neu CSV thieu
    hoac thieu cot - raise loi ro rang de caller tu quyet dinh bo qua
    candidate do (khong duoc suy doan frame_id).
    """
    global _map_keyframes_rows_cache

    cache_key = (str(metadata_dir), str(video_id))
    if cache_key in _map_keyframes_rows_cache:
        return _map_keyframes_rows_cache[cache_key]

    level = video_id.split("_")[0] if "_" in video_id else ""
    candidate_csvs = []
    if metadata_dir:
        candidate_csvs.extend([
            os.path.join(metadata_dir, f"{video_id}.csv"),
            os.path.join(metadata_dir, "map-keyframes", f"{video_id}.csv"),
            os.path.join(metadata_dir, f"map-keyframes-{level}", f"{video_id}.csv"),
            os.path.join(
                metadata_dir, "map-keyframes-aic25-b1", "map-keyframes", f"{video_id}.csv"
            ),
            os.path.join(
                os.path.dirname(metadata_dir),
                "map-keyframes-aic25-b1", "map-keyframes", f"{video_id}.csv",
            ),
            os.path.join(os.path.dirname(metadata_dir), "map-keyframes", f"{video_id}.csv"),
            os.path.join(os.path.dirname(metadata_dir), f"{video_id}.csv"),
        ])
    if os.path.exists("/kaggle/input"):
        candidate_csvs.extend([
            f"/kaggle/input/ai-challenge-hcmc-2026-metadata/map-keyframes-aic25-b1/map-keyframes/{video_id}.csv",
            f"/kaggle/input/ai-challenge-hcmc-2026-metadata/map-keyframes/{video_id}.csv",
            f"/kaggle/input/datasets/quninhphmanh/ai-challenge-hcmc-2026-metadata/map-keyframes-aic25-b1/map-keyframes/{video_id}.csv",
            f"/kaggle/input/datasets/quninhphmanh/ai-challenge-hcmc-2026-metadata/map-keyframes/{video_id}.csv",
        ])

    target_csv_path = next(
        (path for path in candidate_csvs if path and os.path.isfile(path)),
        None,
    )
    if not target_csv_path:
        raise FileNotFoundError(f"Map-Keyframes CSV not found for video {video_id}.")

    import pandas as pd

    df = pd.read_csv(target_csv_path)
    normalized = {str(col).strip().lower(): col for col in df.columns}
    required = ("n", "pts_time", "fps", "frame_idx")
    missing = [col for col in required if col not in normalized]
    if missing:
        raise ValueError(
            f"CSV {target_csv_path} thieu cot bat buoc: {missing}. "
            f"Columns={list(df.columns)}"
        )

    n_series = pd.to_numeric(df[normalized["n"]], errors="coerce")
    pts_series = pd.to_numeric(df[normalized["pts_time"]], errors="coerce")
    fps_series = pd.to_numeric(df[normalized["fps"]], errors="coerce")
    frame_series = pd.to_numeric(df[normalized["frame_idx"]], errors="coerce")

    if (
        n_series.isna().any() or pts_series.isna().any()
        or fps_series.isna().any() or frame_series.isna().any()
    ):
        raise ValueError(f"Invalid/non-numeric values trong {target_csv_path}")

    frame_values = [int(v) for v in frame_series.tolist()]
    if any(frame_values[i] > frame_values[i + 1] for i in range(len(frame_values) - 1)):
        raise ValueError(f"frame_idx khong tang dan trong {target_csv_path}")

    rows = [
        {
            "keyframe_ordinal": ordinal,
            "n": int(n_series.iloc[ordinal]),
            "pts_time": float(pts_series.iloc[ordinal]),
            "fps": float(fps_series.iloc[ordinal]),
            "frame_idx": frame_values[ordinal],
        }
        for ordinal in range(len(frame_values))
    ]

    _map_keyframes_rows_cache[cache_key] = rows
    return rows


class FrameOCRStore:
    """
    Doc OCR keyframe-level tu `<video_id>_ocr.json` (sinh boi
    src/preprocessing/run_ocr.py) va anh xa sang keyframe ordinal 0-based.

    Dung chung cho Task 1 (KIS frame localization), Task 2 (VQA evidence
    frame). Neu khong tim thay file OCR cho video, tra dict rong (an toan,
    khong raise loi, khong lam candidate bien mat).
    """

    def __init__(self, ocr_dir, keyframes_dir):
        self.ocr_dir = Path(ocr_dir) if ocr_dir else None
        self.keyframes_dir = keyframes_dir
        self._path_cache = {}
        self._data_cache = {}
        self._stem_cache = {}
        self._ocr_index = None

    def _build_index(self):
        if self._ocr_index is not None:
            return
        self._ocr_index = {}
        if self.ocr_dir is None or not self.ocr_dir.is_dir():
            return
        for path in self.ocr_dir.rglob("*_ocr.json"):
            video_id = path.name[:-len("_ocr.json")]
            self._ocr_index.setdefault(video_id, path)

    def _find_path(self, video_id):
        video_id = str(video_id)
        if video_id in self._path_cache:
            return self._path_cache[video_id]
        if self.ocr_dir is None:
            self._path_cache[video_id] = None
            return None
        candidates = [
            self.ocr_dir / (video_id + "_ocr.json"),
            self.ocr_dir / "ocr" / (video_id + "_ocr.json"),
        ]
        for candidate in candidates:
            if candidate.is_file():
                self._path_cache[video_id] = candidate
                return candidate
        self._build_index()
        found = (self._ocr_index or {}).get(video_id)
        self._path_cache[video_id] = found
        return found

    def _keyframe_stems(self, video_id):
        video_id = str(video_id)
        if video_id in self._stem_cache:
            return self._stem_cache[video_id]
        stems = {}
        ordinal = 0
        while True:
            try:
                path = get_keyframe_path_by_index(
                    self.keyframes_dir,
                    video_id,
                    ordinal,
                )
            except (IndexError, FileNotFoundError, ValueError):
                break
            stems[Path(path).stem] = ordinal
            ordinal += 1
        self._stem_cache[video_id] = stems
        return stems

    def get_by_ordinal(self, video_id):
        """Tra ve dict {keyframe_ordinal(int): ocr_text(str)} cho 1 video."""
        video_id = str(video_id)
        if video_id in self._data_cache:
            return self._data_cache[video_id]
        path = self._find_path(video_id)
        if path is None:
            self._data_cache[video_id] = {}
            return {}
        try:
            with path.open("r", encoding="utf-8-sig") as file_obj:
                raw = json.load(file_obj)
        except (OSError, json.JSONDecodeError):
            self._data_cache[video_id] = {}
            return {}

        stem_lookup = self._keyframe_stems(video_id)
        output = {}
        if isinstance(raw, dict):
            for key, value in raw.items():
                stem = Path(str(key)).stem
                ordinal = stem_lookup.get(stem)
                if ordinal is None:
                    continue
                if isinstance(value, str):
                    text = value.strip()
                elif isinstance(value, list):
                    text = " ".join(str(item) for item in value).strip()
                else:
                    text = str(value or "").strip()
                if text:
                    output[int(ordinal)] = text
        self._data_cache[video_id] = output
        return output


_KEYWORD_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def ocr_keyword_match_score(query_text, ocr_text):
    """
    Diem match tho giua tu khoa cau query va van ban OCR cua 1 keyframe,
    tra ve trong [0, 1]. KHONG dung BM25 (qua nang khi goi lap cho tung
    frame); dung token-overlap co trong so theo do dai token (uu tien ten
    rieng/so/tu hiem hon stopword ngan).
    """
    if not query_text or not ocr_text:
        return 0.0
    query_tokens = {
        t.lower() for t in _KEYWORD_TOKEN_RE.findall(str(query_text)) if len(t) > 2
    }
    if not query_tokens:
        return 0.0
    ocr_tokens = {
        t.lower() for t in _KEYWORD_TOKEN_RE.findall(str(ocr_text)) if len(t) > 1
    }
    if not ocr_tokens:
        return 0.0
    matched = query_tokens & ocr_tokens
    if not matched:
        return 0.0
    weight = sum(len(t) for t in matched)
    total = sum(len(t) for t in query_tokens) or 1
    return float(min(1.0, weight / total))