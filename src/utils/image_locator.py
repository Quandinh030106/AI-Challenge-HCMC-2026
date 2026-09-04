# ==============================================================================
# AIC 2026 - DYNAMIC MULTI-BATCH UNIVERSAL KEYFRAME IMAGE LOCATOR
# ==============================================================================
import os
import glob
from typing import Optional, Dict

# In-memory fast cache: video_id -> directory containing its jpg images
_VIDEO_FOLDER_CACHE: Dict[str, str] = {}
_INITIAL_SCAN_DONE = False
_ALL_KEYFRAME_ROOTS = []

def _discover_all_keyframe_roots():
    """Scans all search roots to find all keyframe dataset directories across L21 to L30."""
    global _ALL_KEYFRAME_ROOTS, _INITIAL_SCAN_DONE
    if _INITIAL_SCAN_DONE:
        return

    search_roots = ["/kaggle/input", "/kaggle/working", "data", "."]
    candidates = []

    for s_root in search_roots:
        if os.path.exists(s_root):
            for root, dirs, _ in os.walk(s_root):
                r_lower = root.lower()
                if "keyframes" in r_lower or "keyframe" in r_lower:
                    candidates.append(root)

    _ALL_KEYFRAME_ROOTS = sorted(list(set(candidates)), key=len, reverse=True)
    _INITIAL_SCAN_DONE = True


def resolve_keyframe_path(video_id: str, frame_idx_0based: int, fallback_path: str = "") -> str:
    """
    Universal dynamic keyframe image finder.
    Works seamlessly across L21, L22, L23, L24, L25, L26, L27, L28, L29, L30 on Kaggle or Local PC.
    """
    # 1. Quick check: if fallback_path already exists on disk, return immediately
    if fallback_path and os.path.exists(fallback_path):
        return fallback_path

    global _VIDEO_FOLDER_CACHE
    f_1based = frame_idx_0based + 1
    name_patterns = [
        f"{f_1based:03d}.jpg", f"{f_1based:04d}.jpg", f"{f_1based:05d}.jpg", f"{f_1based}.jpg",
        f"{frame_idx_0based:03d}.jpg", f"{frame_idx_0based:04d}.jpg", f"{frame_idx_0based:05d}.jpg", f"{frame_idx_0based}.jpg"
    ]

    # 2. Check if folder for this video_id is already cached
    if video_id in _VIDEO_FOLDER_CACHE:
        v_folder = _VIDEO_FOLDER_CACHE[video_id]
        for np_name in name_patterns:
            full_p = os.path.join(v_folder, np_name)
            if os.path.exists(full_p):
                return full_p

    # 3. Discover roots if not done yet
    _discover_all_keyframe_roots()

    level = video_id.split('_')[0] if '_' in video_id else ""

    # 4. Search across all discovered keyframe roots
    for root in _ALL_KEYFRAME_ROOTS:
        potential_folders = [
            os.path.join(root, video_id),
            os.path.join(root, "keyframes", video_id),
            os.path.join(root, f"Keyframes_{level}", "keyframes", video_id),
            os.path.join(root, f"Keyframes_{level}", video_id),
            os.path.join(root, level, "keyframes", video_id),
            os.path.join(root, level, video_id)
        ]
        for f_cand in potential_folders:
            if os.path.exists(f_cand) and os.path.isdir(f_cand):
                _VIDEO_FOLDER_CACHE[video_id] = f_cand
                for np_name in name_patterns:
                    full_p = os.path.join(f_cand, np_name)
                    if os.path.exists(full_p):
                        return full_p

    # 5. Direct search in /kaggle/input if not yet matched
    search_base = "/kaggle/input"
    if os.path.exists(search_base):
        for root, dirs, _ in os.walk(search_base):
            if video_id in dirs:
                f_cand = os.path.join(root, video_id)
                _VIDEO_FOLDER_CACHE[video_id] = f_cand
                for np_name in name_patterns:
                    full_p = os.path.join(f_cand, np_name)
                    if os.path.exists(full_p):
                        return full_p

    return fallback_path
