import json
import os
import re
from pathlib import Path

import numpy as np
import torch

from src.utils import load_map_keyframes_rows


def align_events_dynamic_programming(scores_matrix, min_gap=8):
    """
    [GIU LAI DE TUONG THICH NGUOC - KHONG XOA]

    Ham nay van con duoc import (nhung khong con duoc goi) tu
    export_codabench_submission.py. KHONG dung ham nay cho pipeline moi vi
    min_gap tinh theo so dong ma tran (keyframe ordinal), khong co y nghia
    thoi gian co dinh giua cac video co mat do keyframe khac nhau.

    Xem align_events_time_aware() cho DP moi dung pts_time thuc.
    """
    T, N = scores_matrix.shape
    if T < N:
        return list(range(T)) + [T - 1] * (N - T), 0.0

    eff_gap = min(min_gap, max(1, (T - N) // max(1, N)))

    dp = np.full((T, N + 1), -np.inf)
    parent = np.full((T, N + 1), -1)

    for t in range(T):
        dp[t, 0] = 0.0

    for j in range(1, N + 1):
        for t in range(0, T):
            if t > 0 and dp[t - 1, j] > dp[t, j]:
                dp[t, j] = dp[t - 1, j]
                parent[t, j] = t - 1

            prev_t = t - eff_gap
            if j == 1:
                prev_score = 0.0
            elif prev_t >= 0:
                prev_score = dp[prev_t, j - 1]
            else:
                prev_score = -np.inf

            current_score = prev_score + scores_matrix[t, j - 1]

            if current_score > dp[t, j]:
                dp[t, j] = current_score
                parent[t, j] = -2

    aligned_frames = []
    t = T - 1
    j = N
    while j > 0 and t >= 0:
        if parent[t, j] == -2 or t < eff_gap * (j - 1):
            aligned_frames.append(t)
            j -= 1
            t = t - eff_gap
        else:
            t = parent[t, j]

    if len(aligned_frames) < N:
        aligned_frames = [int(x) for x in np.linspace(0, T - 1, N)]
    else:
        aligned_frames.reverse()

    return aligned_frames, float(dp[T - 1, N])


def align_events_time_aware(scores_matrix, pts_times, min_gap_seconds=1.0):
    """
    Can chinh N event tren T keyframe ordinal, dung DUNG pts_time (giay) tu
    Map-Keyframes de ap rang buoc khoang cach toi thieu THUC giua 2 event
    lien tiep. Khong dung so dong ma tran lam don vi khoang cach vi mat do
    keyframe khac nhau giua cac video (Prompt 8, muc "khong duoc dung
    min_gap=8 voi y nghia mo ho").

    DP: dp[t, j] = diem tot nhat can chinh j event dau tien, voi event thu j
    dat o keyframe ordinal t.
    - j == 1: khong co rang buoc (event dau tien co the o bat ky dau).
    - j >= 2: predecessor t' phai thoa pts_times[t] - pts_times[t'] >= min_gap.
      Duyet bang two-pointer O(T) cho moi j vi pts_times tang dan.

    Return:
        aligned_indices: list[int] (do dai N) - keyframe ordinal cho tung event
        feasible: bool - DP tim duoc loi giai thoa rang buoc hay phai fallback
        dp_score: float - tong diem DP (0.0 neu fallback)
        raw_peaks: list[int] - argmax khong rang buoc cua tung event (chan doan)
        order_error: bool - raw_peaks co bi dao thu tu hay khong (chan doan)
    """
    scores_matrix = np.asarray(scores_matrix, dtype=np.float64)
    T, N = scores_matrix.shape
    pts_times = np.asarray(pts_times, dtype=np.float64).reshape(-1)

    if len(pts_times) != T:
        common_len = min(T, len(pts_times))
        scores_matrix = scores_matrix[:common_len]
        pts_times = pts_times[:common_len]
        T = common_len

    if T == 0:
        return [0] * N, False, 0.0, [], False

    raw_peaks = [int(np.argmax(scores_matrix[:, j])) for j in range(N)]
    order_error = any(
        raw_peaks[idx] >= raw_peaks[idx + 1] for idx in range(len(raw_peaks) - 1)
    )

    if T < N:
        fallback = [int(x) for x in np.linspace(0, T - 1, N)]
        return fallback, False, 0.0, raw_peaks, order_error

    min_gap_seconds = max(0.0, float(min_gap_seconds))
    neg_inf = float("-1e18")
    dp = np.full((T, N + 1), neg_inf, dtype=np.float64)
    parent = np.full((T, N + 1), -1, dtype=np.int64)

    for t in range(T):
        dp[t, 1] = scores_matrix[t, 0]
        parent[t, 1] = -2  # danh dau: diem bat dau chuoi, khong co predecessor

    for j in range(2, N + 1):
        left = 0
        best_prev_value = neg_inf
        best_prev_index = -1
        for t in range(T):
            threshold = pts_times[t] - min_gap_seconds
            while left < T and pts_times[left] <= threshold:
                if dp[left, j - 1] > best_prev_value:
                    best_prev_value = dp[left, j - 1]
                    best_prev_index = left
                left += 1
            if best_prev_index >= 0:
                candidate_value = best_prev_value + scores_matrix[t, j - 1]
                if candidate_value > dp[t, j]:
                    dp[t, j] = candidate_value
                    parent[t, j] = best_prev_index

    best_t = int(np.argmax(dp[:, N]))
    best_value = float(dp[best_t, N])

    if best_value <= neg_inf / 2:
        fallback = [int(x) for x in np.linspace(0, T - 1, N)]
        return fallback, False, 0.0, raw_peaks, order_error

    aligned = [best_t]
    t = best_t
    j = N
    while j > 1:
        prev_t = int(parent[t, j])
        aligned.append(prev_t)
        t = prev_t
        j -= 1
    aligned.reverse()

    return aligned, True, best_value, raw_peaks, order_error


def _rank_score(rank, decay):
    """Chuyen rank 1-based thanh [0, 1], cung cong thuc voi sequence_search.py."""
    if rank is None:
        return 0.0
    rank = max(1, int(rank))
    decay = max(1.0, float(decay))
    return float((decay + 1.0) / (decay + rank))


def _load_video_features(dense_searcher, video_id):
    """
    Lay CLIP feature (da chuan hoa L2) cua 1 video.

    Uu tien dense_searcher.video_features_dict (da chuan hoa san trong
    dense_search.py). Fallback doc truc tiep .npy CHI khi video khong nam
    trong ma tran toan cuc (vi du khong lot top_k_videos ban dau) - va PHAI
    tu chuan hoa L2 lai, neu khong event_matching_score se khong the so
    sanh duoc giua cac video (bug da phat hien trong ban cu, khong chuan hoa
    fallback path).
    """
    video_features = dense_searcher.video_features_dict.get(video_id)
    if video_features is not None:
        return video_features

    level = video_id.split("_")[0] if "_" in video_id else ""
    candidate_paths = [
        os.path.join(dense_searcher.features_dir, f"{video_id}.npy"),
        os.path.join(dense_searcher.features_dir, f"clip-features-{level}", f"{video_id}.npy"),
        os.path.join(dense_searcher.features_dir, f"clip_features_{level}", f"{video_id}.npy"),
        os.path.join(dense_searcher.features_dir, "clip-features-32", f"{video_id}.npy"),
        os.path.join(
            dense_searcher.features_dir,
            "clip-features-32-aic25-b1",
            "clip-features-32",
            f"{video_id}.npy",
        ),
    ]
    for candidate_path in candidate_paths:
        if os.path.exists(candidate_path):
            try:
                raw = np.load(candidate_path)
            except Exception as exc:
                print(f"TRAKE: Canh bao khong nap duoc feature {candidate_path}: {exc}")
                continue
            norms = np.linalg.norm(raw, axis=-1, keepdims=True)
            norms[norms == 0] = 1e-10
            return raw / norms
    return None


def _raw_refine_event_sequence(
    video_id,
    events_text,
    event_translations,
    coarse_actual_frames,
    map_rows,
    temporal_refiner,
    query_processor,
    raw_refine_config,
):
    """
    Tinh chinh tuan tu tung event tren raw video, dam bao TUYET DOI
    E1 < E2 < ... < EN sau khi refine.

    Cua so tim kiem cua event i bi gioi han boi coarse-frame cua event lang
    gieng (i-1, i+1) de khong "lan" sang vung cua event khac, va boi actual
    frame da chon cua event (i-1) de dam bao tinh don dieu tuyet doi ke ca
    khi raw-refine di lech khoi coarse alignment.
    """
    n_events = len(events_text)
    max_actual_frame = (
        map_rows[-1]["frame_idx"] if map_rows else coarse_actual_frames[-1]
    )
    refined = list(int(f) for f in coarse_actual_frames)
    trace = []
    prev_refined = -1

    window_seconds = raw_refine_config.get("fine_window_seconds")
    sample_fps = raw_refine_config.get("fine_sample_fps")

    for index in range(n_events):
        coarse = int(coarse_actual_frames[index])
        left_neighbor = coarse_actual_frames[index - 1] if index > 0 else 0
        right_neighbor = (
            coarse_actual_frames[index + 1] if index < n_events - 1 else max_actual_frame
        )
        lower_bound = max(int(left_neighbor), prev_refined + 1)
        upper_bound = max(lower_bound, int(right_neighbor))

        prompts = None
        if query_processor is not None:
            prompts = query_processor.generate_prompt_ensemble(
                event_translations[index],
                query_vi=events_text[index],
            )
        if not prompts:
            prompts = [event_translations[index] or events_text[index]]

        result = None
        try:
            result = temporal_refiner.refine_trake_event(
                video_id=video_id,
                coarse_actual_frame=coarse,
                prompts=prompts,
                lower_bound_frame=lower_bound,
                upper_bound_frame=upper_bound,
                window_seconds=window_seconds,
                sample_fps=sample_fps,
            )
        except Exception as exc:
            print(f"TRAKE raw-refine warning video={video_id} event={index + 1}: {exc}")

        if (
            result is not None
            and lower_bound <= result["actual_frame"] <= upper_bound
            and result["actual_frame"] > prev_refined
        ):
            refined[index] = int(result["actual_frame"])
            trace.append({
                "event_index": index + 1,
                "status": "refined",
                "coarse_actual_frame": coarse,
                "refined_actual_frame": refined[index],
                "score": result.get("score"),
                "search_window": [
                    result.get("search_start_frame"),
                    result.get("search_end_frame"),
                ],
            })
        else:
            refined[index] = max(coarse, prev_refined + 1)
            trace.append({
                "event_index": index + 1,
                "status": "fallback_coarse",
                "coarse_actual_frame": coarse,
                "refined_actual_frame": refined[index],
            })
        prev_refined = refined[index]

    return refined, trace


def _write_trake_trace(trace, log_dir, query_id):
    if not log_dir:
        return
    output_dir = Path(log_dir)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(query_id or "query"))
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / (safe_id + ".json")).open("w", encoding="utf-8") as file_obj:
            json.dump(trace, file_obj, ensure_ascii=False, indent=2)
    except OSError as exc:
        print("TRAKE: Canh bao khong ghi duoc evidence log: %s" % exc)


def solve_task3_batch(
    query_events,
    fused_candidates,
    keyframes_dir,
    dense_searcher,
    metadata_dir=None,
    query_processor=None,
    total_preds=100,
    config=None,
    temporal_refiner=None,
    query_id=None,
):
    """
    Giai Task 3 (TRAKE) cho toan bo candidate video (Prompt 8).

    Pipeline:
      1. fused_candidates da duoc retrieve tu CONTEXT (khong doi o day - dung
         nguyen tac "TRAKE khong phai nhieu KIS doc lap": chi 1 lan retrieval
         theo context, KHONG retrieve rieng theo tung event).
      2. [Coarse] DP can chinh N event tren keyframe CLIP feature co san,
         rang buoc THEO GIAY THUC (pts_time tu Map-Keyframes).
      3. [Re-rank] video_score = w1*retrieval_rank_score
                               + w2*event_matching_score (min-max trong batch)
                               + w3*sequence_consistency_score (DP feasible?)
      4. [Fine] Raw-video refine (TemporalRefiner) cho Top-N video sau rerank,
         BAO TOAN thu tu E1 < E2 < ... < EN.
      5. Map ve actual frame_id qua Map-Keyframes - KHONG BAO GIO suy doan.

    Output schema KHONG DOI: [{"video_id": str, "frame_ids": [str, ...]}, ...]
    """
    if not fused_candidates or not query_events:
        return []

    n_events = len(query_events)

    trake_config = (config or {}).get("search", {}).get("trake_alignment", {})
    min_gap_seconds = float(trake_config.get("min_event_gap_seconds", 1.0))
    rank_decay = float(trake_config.get("rank_decay", 60.0))
    weights = {
        "retrieval": float(trake_config.get("video_retrieval_weight", 0.35)),
        "event": float(trake_config.get("event_matching_weight", 0.45)),
        "sequence": float(trake_config.get("sequence_consistency_weight", 0.20)),
    }
    weight_total = sum(max(0.0, value) for value in weights.values()) or 1.0

    raw_refine_config = trake_config.get("raw_refine", {})
    raw_refine_enabled = (
        bool(raw_refine_config.get("enabled", True)) and temporal_refiner is not None
    )
    raw_refine_limit = max(0, int(raw_refine_config.get("candidate_limit", 10)))

    log_evidence = bool(trake_config.get("log_evidence", True))
    log_dir = trake_config.get("log_dir", "output/trake_evidence")

    target_count = min(100, int(total_preds))
    pool_size = min(
        len(fused_candidates),
        max(target_count, int(trake_config.get("coarse_pool_size", target_count))),
    )
    candidate_pool = list(fused_candidates[:pool_size])

    # 1. Vector hoa event MOT LAN duy nhat (dung chung cho toan bo candidate).
    event_translations = []
    event_vectors = []
    for event_text in query_events:
        en_event = (
            query_processor.translate_vi_to_en(event_text)
            if query_processor else event_text
        )
        event_translations.append(en_event)
        vec = dense_searcher.encode_text(en_event)
        if isinstance(vec, torch.Tensor):
            vec = vec.float().cpu().numpy().squeeze(0)
        event_vectors.append(vec)
    event_vectors = np.array(event_vectors)

    # 2. Coarse alignment cho tung candidate (re, vi da co san feature).
    scored_candidates = []
    for original_rank, cand in enumerate(candidate_pool):
        video_id = str(cand.get("video_id"))

        video_features = _load_video_features(dense_searcher, video_id)
        if video_features is None:
            continue

        try:
            map_rows = load_map_keyframes_rows(metadata_dir, video_id)
        except (FileNotFoundError, ValueError, Exception) as exc:
            print(f"TRAKE: Canh bao thieu/loi Map-Keyframes cho {video_id}: {exc}")
            continue
        if not map_rows:
            continue

        n_frames = min(len(video_features), len(map_rows))
        if n_frames == 0:
            continue

        scores_matrix = np.dot(video_features[:n_frames], event_vectors.T)
        pts_times = [row["pts_time"] for row in map_rows[:n_frames]]

        aligned_ordinals, feasible, dp_score, raw_peaks, order_error = (
            align_events_time_aware(
                scores_matrix,
                pts_times,
                min_gap_seconds=min_gap_seconds,
            )
        )

        per_event_scores = [
            float(scores_matrix[aligned_ordinals[idx], idx])
            for idx in range(n_events)
        ]
        coarse_actual_frames = [
            int(map_rows[ordinal]["frame_idx"]) for ordinal in aligned_ordinals
        ]

        scored_candidates.append({
            "video_id": video_id,
            "original_rank": original_rank,
            "map_rows": map_rows,
            "aligned_ordinals": aligned_ordinals,
            "coarse_actual_frames": coarse_actual_frames,
            "per_event_scores": per_event_scores,
            "feasible": feasible,
            "dp_score": dp_score,
            "raw_peaks": raw_peaks,
            "order_error": order_error,
            "retrieval_score": _rank_score(original_rank + 1, rank_decay),
            "sequence_score": 1.0 if feasible else 0.2,
        })

    if not scored_candidates:
        return []

    # 3. Re-rank: ket hop retrieval + event matching (min-max normalize) + sequence.
    mean_event_scores = [
        float(np.mean(item["per_event_scores"])) if item["per_event_scores"] else 0.0
        for item in scored_candidates
    ]
    lo, hi = min(mean_event_scores), max(mean_event_scores)
    span = hi - lo
    for item, raw_mean in zip(scored_candidates, mean_event_scores):
        event_matching_score = 1.0 if span <= 1e-8 else (raw_mean - lo) / span
        item["event_matching_score"] = float(event_matching_score)
        item["mean_event_score_raw"] = raw_mean
        item["video_score"] = float((
            weights["retrieval"] * item["retrieval_score"]
            + weights["event"] * event_matching_score
            + weights["sequence"] * item["sequence_score"]
        ) / weight_total)

    ranked = sorted(scored_candidates, key=lambda item: item["video_score"], reverse=True)
    ranked = ranked[:target_count]

    # 4. Raw-video fine refine cho Top-N sau rerank (bao ve Top-1/Top-5, gioi
    # han chi phi decode video cho phan con lai cua Top-100).
    for rank, item in enumerate(ranked):
        if raw_refine_enabled and rank < raw_refine_limit:
            final_frames, refine_trace = _raw_refine_event_sequence(
                video_id=item["video_id"],
                events_text=query_events,
                event_translations=event_translations,
                coarse_actual_frames=item["coarse_actual_frames"],
                map_rows=item["map_rows"],
                temporal_refiner=temporal_refiner,
                query_processor=query_processor,
                raw_refine_config=raw_refine_config,
            )
            item["final_actual_frames"] = final_frames
            item["raw_refine_trace"] = refine_trace
        else:
            item["final_actual_frames"] = item["coarse_actual_frames"]
            item["raw_refine_trace"] = None

    predictions = [
        {
            "video_id": item["video_id"],
            "frame_ids": [str(int(f)) for f in item["final_actual_frames"]],
        }
        for item in ranked
    ]

    print(
        "TRAKE: query=%s | events=%d | candidates_evaluated=%d | pool=%d"
        % (query_id or "query", n_events, len(scored_candidates), pool_size)
    )
    for rank, item in enumerate(ranked[:5], start=1):
        print(
            "  #%d %s | video_score=%.4f (retrieval=%.2f event=%.2f seq=%.2f) | feasible=%s"
            % (
                rank,
                item["video_id"],
                item["video_score"],
                item["retrieval_score"],
                item["event_matching_score"],
                item["sequence_score"],
                item["feasible"],
            )
        )

    if log_evidence:
        trace = {
            "query_id": str(query_id or ""),
            "events": list(query_events),
            "event_translations": event_translations,
            "min_event_gap_seconds": min_gap_seconds,
            "weights": weights,
            "coarse_pool_size": pool_size,
            "raw_refine": {
                "enabled": raw_refine_enabled,
                "candidate_limit": raw_refine_limit,
            },
            "candidates_evaluated": len(scored_candidates),
            "ranked_candidates": [
                {
                    "final_rank": rank + 1,
                    "video_id": item["video_id"],
                    "original_retrieval_rank": item["original_rank"] + 1,
                    "video_score": item["video_score"],
                    "score_components": {
                        "retrieval_score": item["retrieval_score"],
                        "event_matching_score": item["event_matching_score"],
                        "sequence_consistency_score": item["sequence_score"],
                    },
                    "dp_feasible": item["feasible"],
                    "dp_score": item["dp_score"],
                    "raw_peak_order_error": item["order_error"],
                    "per_event_scores": item["per_event_scores"],
                    "coarse_actual_frames": item["coarse_actual_frames"],
                    "final_actual_frames": [int(f) for f in item["final_actual_frames"]],
                    "raw_refine_trace": item["raw_refine_trace"],
                }
                for rank, item in enumerate(ranked[:20])
            ],
        }
        _write_trake_trace(trace, log_dir, query_id)

    return predictions


def solve_task3(
    query_events,
    fused_candidates,
    keyframes_dir,
    dense_searcher,
    metadata_dir=None,
    query_processor=None,
    config=None,
    temporal_refiner=None,
    query_id=None,
):
    res_batch = solve_task3_batch(
        query_events,
        fused_candidates[:1],
        keyframes_dir,
        dense_searcher,
        metadata_dir=metadata_dir,
        query_processor=query_processor,
        total_preds=1,
        config=config,
        temporal_refiner=temporal_refiner,
        query_id=query_id,
    )
    return res_batch[0] if res_batch else {"video_id": "none", "frame_ids": []}