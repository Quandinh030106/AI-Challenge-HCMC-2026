import os
import glob
import numpy as np
from src.utils import ocr_keyword_match_score


_video_folder_cache = {}
_csv_map_cache = {}


class FrameMappingError(RuntimeError):
    pass


def get_frame_id_from_idx(keyframes_dir, video_id, frame_idx, metadata_dir=None):
    """
    Map keyframe/vector ordinal (0-based) -> actual frame_idx của video gốc.

    `frame_idx` tham số của hàm này là index của vector/keyframe trong mảng,
    KHÔNG phải actual video frame index.
    """
    global _csv_map_cache

    keyframe_idx = int(frame_idx)
    if keyframe_idx < 0:
        raise FrameMappingError(
            f"Negative keyframe index: video={video_id}, idx={keyframe_idx}"
        )

    cache_key = (str(metadata_dir), video_id)
    if cache_key in _csv_map_cache:
        values = _csv_map_cache[cache_key]
        if keyframe_idx >= len(values):
            raise FrameMappingError(
                f"Map index out of range: video={video_id}, "
                f"idx={keyframe_idx}, map_rows={len(values)}"
            )
        return str(int(values[keyframe_idx]))

    level = video_id.split("_")[0] if "_" in video_id else ""
    candidate_csvs = []

    if metadata_dir:
        candidate_csvs.extend([
            os.path.join(metadata_dir, f"{video_id}.csv"),
            os.path.join(metadata_dir, "map-keyframes", f"{video_id}.csv"),
            os.path.join(metadata_dir, f"map-keyframes-{level}", f"{video_id}.csv"),
            os.path.join(
                metadata_dir,
                "map-keyframes-aic25-b1",
                "map-keyframes",
                f"{video_id}.csv",
            ),
            os.path.join(
                os.path.dirname(metadata_dir),
                "map-keyframes-aic25-b1",
                "map-keyframes",
                f"{video_id}.csv",
            ),
            os.path.join(
                os.path.dirname(metadata_dir),
                "map-keyframes",
                f"{video_id}.csv",
            ),
            os.path.join(os.path.dirname(metadata_dir), f"{video_id}.csv"),
        ])

    if os.path.exists("/kaggle/input"):
        candidate_csvs.extend([
            f"/kaggle/input/ai-challenge-hcmc-2026-metadata/"
            f"map-keyframes-aic25-b1/map-keyframes/{video_id}.csv",
            f"/kaggle/input/ai-challenge-hcmc-2026-metadata/"
            f"map-keyframes/{video_id}.csv",
            f"/kaggle/input/datasets/quninhphmanh/"
            f"ai-challenge-hcmc-2026-metadata/"
            f"map-keyframes-aic25-b1/map-keyframes/{video_id}.csv",
            f"/kaggle/input/datasets/quninhphmanh/"
            f"ai-challenge-hcmc-2026-metadata/"
            f"map-keyframes/{video_id}.csv",
        ])

    target_csv_path = next(
        (path for path in candidate_csvs if path and os.path.isfile(path)),
        None,
    )

    if not target_csv_path:
        raise FrameMappingError(
            f"Map-Keyframes CSV not found for video {video_id}. "
            "Refusing to generate a fake frame_id."
        )

    try:
        import pandas as pd

        df = pd.read_csv(target_csv_path)

        normalized = {
            str(col).strip().lower(): col
            for col in df.columns
        }

        # Dataset BTC thực tế dùng `frame_idx`.
        # Không đoán bằng cột numeric lớn nhất.
        if "frame_idx" not in normalized:
            raise FrameMappingError(
                f"CSV {target_csv_path} does not contain required column 'frame_idx'. "
                f"Columns={list(df.columns)}"
            )

        series = pd.to_numeric(
            df[normalized["frame_idx"]],
            errors="coerce",
        )

        if series.isna().any():
            raise FrameMappingError(
                f"Invalid/non-numeric frame_idx values in {target_csv_path}"
            )

        values = [int(v) for v in series.tolist()]

        if any(values[i] > values[i + 1] for i in range(len(values) - 1)):
            raise FrameMappingError(
                f"frame_idx is not monotonic in {target_csv_path}"
            )

        _csv_map_cache[cache_key] = values

        if keyframe_idx >= len(values):
            raise FrameMappingError(
                f"Map index out of range: video={video_id}, "
                f"idx={keyframe_idx}, map_rows={len(values)}"
            )

        return str(values[keyframe_idx])

    except FrameMappingError:
        raise
    except Exception as exc:
        raise FrameMappingError(
            f"Failed reading Map-Keyframes for {video_id}: "
            f"{target_csv_path}: {exc}"
        ) from exc


def gaussian_smooth_scores(scores, sigma=1.5):
    """Lam min chuoi diem thoi gian bang Gaussian Kernel."""
    if len(scores) < 3:
        return scores
    radius = int(3 * sigma)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / np.sum(kernel)
    return np.convolve(scores, kernel, mode="same")


def solve_task1(query_text, fused_candidates, keyframes_dir, metadata_dir=None, sigma=1.5):
    """Giai quyet Task 1 (Textual KIS)."""
    if not fused_candidates:
        return {"video_id": "none", "frame_id": None, "score": 0.0}

    best_candidate = fused_candidates[0]
    video_id = best_candidate["video_id"]
    dense_info = best_candidate.get("dense_info")

    if dense_info is None or "all_scores" not in dense_info:
        return {
            "video_id": video_id,
            "frame_id": None,
            "score": best_candidate.get("rrf_score", 0.0),
        }

    scores = dense_info["all_scores"]
    smoothed_scores = gaussian_smooth_scores(scores, sigma=sigma)
    best_frame_idx = int(np.argmax(smoothed_scores))
    frame_id = get_frame_id_from_idx(
        keyframes_dir,
        video_id,
        best_frame_idx,
        metadata_dir=metadata_dir,
    )

    return {
        "video_id": video_id,
        "frame_id": frame_id,
        "score": float(smoothed_scores[best_frame_idx]),
    }


def _append_unique(target, seen, value, upper_bound):
    """Them keyframe ordinal hop le, khong trung lap."""
    idx = int(value)
    if 0 <= idx < upper_bound and idx not in seen:
        target.append(idx)
        seen.add(idx)


def _build_ranked_frame_indices(
    dense_info,
    ocr_by_ordinal=None,
    query_text=None,
    ocr_weight=0.0,
):
    """
    Tao danh sach keyframe ordinal cho mot video theo thu tu chat luong.

    Prompt 9 - Phan A: neu duoc cung cap `ocr_by_ordinal` (dict {ordinal:
    text} tu FrameOCRStore) va `query_text`, ket hop OCR keyword-match voi
    CLIP score de CHON primary/peak dau tien - vi CLIP kem doc chu/so/bien
    hieu. Mac dinh (ocr_weight=0 hoac khong truyen ocr_by_ordinal) hanh vi
    giu NGUYEN 100% nhu truoc (CLIP-only) - tuong thich nguoc voi moi
    caller cu (main.py, export_codabench_submission.py, export_submission.py,
    ui.py deu khong doi).
    """
    if not dense_info or "all_scores" not in dense_info:
        return [0]

    scores = np.asarray(dense_info["all_scores"], dtype=np.float32).reshape(-1)
    n_frames = len(scores)
    if n_frames == 0:
        return []

    clean_scores = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
    smoothed = np.asarray(
        gaussian_smooth_scores(clean_scores, sigma=1.5),
        dtype=np.float32,
    ).reshape(-1)
    if len(smoothed) != n_frames:
        smoothed = clean_scores

    sorted_indices = np.argsort(smoothed)[::-1]
    primary = int(sorted_indices[0])

    if ocr_by_ordinal and query_text and ocr_weight > 0:
        ocr_scores = np.zeros(n_frames, dtype=np.float32)
        for ordinal, text in ocr_by_ordinal.items():
            ordinal = int(ordinal)
            if 0 <= ordinal < n_frames:
                ocr_scores[ordinal] = ocr_keyword_match_score(query_text, text)

        if float(np.max(ocr_scores)) > 0.0:
            low, high = float(smoothed.min()), float(smoothed.max())
            dense_norm = (
                (smoothed - low) / (high - low)
                if high - low > 1e-8
                else np.zeros_like(smoothed)
            )
            weight = min(1.0, max(0.0, float(ocr_weight)))
            combined = (1.0 - weight) * dense_norm + weight * ocr_scores
            sorted_indices = np.argsort(combined)[::-1]
            primary = int(sorted_indices[0])

    ranked = []
    seen = set()
    _append_unique(ranked, seen, primary, n_frames)

    local_neighbors = [
        primary + delta
        for delta in (-1, 1, -2, 2)
        if 0 <= primary + delta < n_frames
    ]
    local_neighbors.sort(key=lambda idx: float(smoothed[idx]), reverse=True)
    for idx in local_neighbors:
        _append_unique(ranked, seen, idx, n_frames)

    selected_peaks = []
    min_distance = 12
    for idx in sorted_indices:
        idx = int(idx)
        if all(abs(idx - peak) >= min_distance for peak in selected_peaks):
            selected_peaks.append(idx)
        if len(selected_peaks) >= 8:
            break

    for peak in selected_peaks:
        _append_unique(ranked, seen, peak, n_frames)

    for peak in selected_peaks[:5]:
        neighbor_indices = [
            peak + delta
            for delta in (-3, 3, -6, 6, -10, 10, -15, 15)
            if 0 <= peak + delta < n_frames
        ]
        neighbor_indices.sort(key=lambda idx: float(smoothed[idx]), reverse=True)
        for idx in neighbor_indices:
            _append_unique(ranked, seen, idx, n_frames)

    for idx in np.linspace(0, n_frames - 1, min(10, n_frames), dtype=int):
        _append_unique(ranked, seen, int(idx), n_frames)

    for idx in sorted_indices:
        _append_unique(ranked, seen, int(idx), n_frames)

    return ranked


def generate_diversity_top100_kis(
    fused_candidates,
    keyframes_dir,
    metadata_dir=None,
    total_preds=100,
    ocr_store=None,
    query_text=None,
    ocr_weight=0.0,
):
    """
    Tao Top-100 theo rank-aware candidate allocation (logic phan bo giu
    NGUYEN nhu Prompt 4). Prompt 9: them tham so OCR tuy chon, mac dinh
    khong dung (backward compatible).
    """
    try:
        target_count = int(total_preds)
    except (TypeError, ValueError):
        target_count = 100
    target_count = min(target_count, 100)

    if target_count <= 0 or not fused_candidates:
        return []

    candidates = list(fused_candidates[:target_count])
    frame_lists = []
    for candidate in candidates:
        ocr_by_ordinal = None
        if ocr_store is not None and ocr_weight > 0:
            ocr_by_ordinal = ocr_store.get_by_ordinal(candidate.get("video_id"))
        frame_lists.append(
            _build_ranked_frame_indices(
                candidate.get("dense_info"),
                ocr_by_ordinal=ocr_by_ordinal,
                query_text=query_text,
                ocr_weight=ocr_weight,
            )
        )

    cursors = [0] * len(candidates)
    primary_emitted = set()
    emitted_pairs = set()
    predictions = []

    def emit_next(video_rank):
        if len(predictions) >= target_count:
            return False
        if video_rank < 0 or video_rank >= len(candidates):
            return False
        video_id = candidates[video_rank].get("video_id")
        if not video_id:
            return False
        frame_indices = frame_lists[video_rank]
        while cursors[video_rank] < len(frame_indices):
            keyframe_idx = frame_indices[cursors[video_rank]]
            cursors[video_rank] += 1
            frame_id = get_frame_id_from_idx(
                keyframes_dir, video_id, keyframe_idx, metadata_dir=metadata_dir,
            )
            output_key = (str(video_id), str(frame_id))
            if output_key in emitted_pairs:
                continue
            emitted_pairs.add(output_key)
            predictions.append({"video_id": video_id, "frame_id": frame_id})
            primary_emitted.add(video_rank)
            return True
        return False

    def fill_unseen_primaries(target_slot):
        target_slot = min(target_count, int(target_slot))
        for video_rank in range(len(candidates)):
            if len(predictions) >= target_slot:
                break
            if video_rank not in primary_emitted:
                emit_next(video_rank)

    def emit_one_alternative_each(max_video_count):
        upper = min(len(candidates), int(max_video_count))
        for video_rank in range(upper):
            if len(predictions) >= target_count:
                break
            emit_next(video_rank)

    for video_rank in (0, 1, 0, 2, 3):
        if len(predictions) >= min(5, target_count):
            break
        emit_next(video_rank)

    fill_unseen_primaries(20)
    emit_one_alternative_each(5)
    fill_unseen_primaries(50)
    emit_one_alternative_each(10)
    fill_unseen_primaries(80)
    emit_one_alternative_each(20)

    while len(predictions) < target_count:
        made_progress = False
        for video_rank in range(len(candidates)):
            if len(predictions) >= target_count:
                break
            if emit_next(video_rank):
                made_progress = True
        if not made_progress:
            break

    return predictions[:target_count]

