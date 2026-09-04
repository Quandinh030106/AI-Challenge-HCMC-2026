import json
import re
from pathlib import Path

import numpy as np


def _safe_float(value, default=0.0):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None if default is None else float(default)
    if np.isfinite(number):
        return number
    return None if default is None else float(default)


def _rank_score(rank, decay):
    """Chuyen rank 1-based thanh [0, 1], khong tron lan scale Dense/BM25."""
    if rank is None:
        return 0.0
    rank = max(1, int(rank))
    decay = max(1.0, float(decay))
    return float((decay + 1.0) / (decay + rank))


class KISQueryDecomposer:
    """Tach query KIS thanh cac semantic anchor theo quy tac nhe, deterministic."""

    _marker_pattern = re.compile(
        r"(?P<reverse>\btrước\s+đó\b)"
        r"|(?P<forward>"
        r"\b(?:sau\s+đó|tiếp\s+theo|tiếp\s+đến|kế\s+tiếp)\b"
        r"|\bngay\s+sau(?:\s+cảnh\s+này)?\b"
        r"|\bsau\s+vài\s+(?:giây|phút|khoảnh\s+khắc)(?:\s+nghỉ)?\b"
        r"|\b(?:đoạn\s+clip|đoạn\s+phim|mẩu\s+tin|cảnh\s+quay)?\s*"
        r"bắt\s+đầu\s+(?:bằng|với)\b"
        r"|\b(?:đoạn\s+clip|đoạn\s+phim|cảnh\s+quay)?\s*"
        r"(?:kết\s+thúc\s+(?:bằng|với|khi)|kết\s+thúc)\b"
        r"|\bcảnh\s+quay\s+tiếp\s+theo(?:\s+là)?\b"
        r"|\bđầu\s+tiên\s+là\b"
        r"|\bslide\s+bài\s+giảng\b"
        r")",
        flags=re.IGNORECASE,
    )

    _leading_noise = re.compile(
        r"^(?:đoạn\s+clip|đoạn\s+phim|mẩu\s+tin|cảnh\s+quay|hình\s+ảnh)"
        r"\s*(?:cho\s+thấy|là|về)?\s*",
        flags=re.IGNORECASE,
    )

    def __init__(self, min_event_words=4, max_events=4):
        self.min_event_words = max(2, int(min_event_words))
        self.max_events = max(2, int(max_events))

    @staticmethod
    def _normalize(text):
        text = str(text or "").replace("\r", "\n")
        text = re.sub(r"\s+", " ", text)
        return text.strip(" \t\n,;:-")

    def _clean_anchor(self, text):
        anchor = self._normalize(text)
        anchor = self._leading_noise.sub("", anchor).strip(" ,;:-.")
        return anchor

    def _is_meaningful(self, text):
        return len(re.findall(r"\w+", text, flags=re.UNICODE)) >= self.min_event_words

    def _limit_events(self, events):
        events = list(events)
        while len(events) > self.max_events:
            pair_index = min(
                range(len(events) - 1),
                key=lambda idx: len(events[idx].split()) + len(events[idx + 1].split()),
            )
            merged = "%s. %s" % (events[pair_index], events[pair_index + 1])
            events[pair_index:pair_index + 2] = [merged]
        return events

    def decompose(self, query_text):
        query = self._normalize(query_text)
        matches = list(self._marker_pattern.finditer(query))
        if not matches:
            return {
                "mode": "single_scene",
                "events": [query] if query else [],
                "markers": [],
                "reason": "no_strong_temporal_marker",
            }

        events = []
        marker_log = []
        prefix = self._clean_anchor(query[:matches[0].start()])
        if self._is_meaningful(prefix):
            events.append(prefix)

        for marker_index, match in enumerate(matches):
            end = matches[marker_index + 1].start() if marker_index + 1 < len(matches) else len(query)
            anchor = self._clean_anchor(query[match.end():end])
            relation = "reverse" if match.group("reverse") else "forward"
            marker_log.append({
                "marker": match.group(0).strip(),
                "relation": relation,
            })
            if not self._is_meaningful(anchor):
                continue

            if relation == "reverse" and events:
                # "Trước đó" mô tả event đứng trước cảnh vừa nêu (query p1-12).
                events.insert(max(0, len(events) - 1), anchor)
            else:
                events.append(anchor)

        # Loai anchor lap do marker long nhau hoac van ban lap.
        unique_events = []
        seen = set()
        for event in events:
            key = re.sub(r"\W+", " ", event.lower(), flags=re.UNICODE).strip()
            if key and key not in seen:
                unique_events.append(event)
                seen.add(key)

        unique_events = self._limit_events(unique_events)
        if len(unique_events) < 2:
            return {
                "mode": "single_scene",
                "events": [query] if query else [],
                "markers": marker_log,
                "reason": "fewer_than_two_meaningful_events",
            }

        return {
            "mode": "multi_event",
            "events": unique_events,
            "markers": marker_log,
            "reason": "strong_temporal_structure",
        }


def _temporal_alignment(event_scores, min_gap=1):
    """
    Can chinh nhe bang DP tren keyframe ordinal.

    Day chi tao temporal consistency cho video ranking. Khong map sang actual
    frame_id va khong thay raw-video frame refinement.
    """
    arrays = [np.asarray(scores, dtype=np.float32).reshape(-1) for scores in event_scores]
    if len(arrays) < 2 or not arrays or min(len(arr) for arr in arrays) == 0:
        return 0.0, [], [], False

    n_frames = min(len(arr) for arr in arrays)
    arrays = [arr[:n_frames] for arr in arrays]
    raw_peaks = []
    normalized = []
    for arr in arrays:
        clean = np.nan_to_num(arr, nan=-1e9, posinf=1e9, neginf=-1e9)
        peak = int(np.argmax(clean))
        raw_peaks.append(peak)
        low = float(np.min(clean))
        high = float(np.max(clean))
        if high - low <= 1e-8:
            normalized.append(np.zeros_like(clean, dtype=np.float32))
        else:
            normalized.append(((clean - low) / (high - low)).astype(np.float32))

    gap = max(0, int(min_gap))
    if gap * (len(arrays) - 1) >= n_frames:
        gap = 0

    neg_inf = -1e30
    dp = normalized[0].astype(np.float64)
    backpointers = []
    for event_index in range(1, len(normalized)):
        next_dp = np.full(n_frames, neg_inf, dtype=np.float64)
        back = np.full(n_frames, -1, dtype=np.int64)
        best_value = neg_inf
        best_index = -1
        for frame_index in range(n_frames):
            predecessor = frame_index - gap
            if predecessor >= 0 and dp[predecessor] > best_value:
                best_value = float(dp[predecessor])
                best_index = predecessor
            if best_index >= 0:
                next_dp[frame_index] = best_value + float(normalized[event_index][frame_index])
                back[frame_index] = best_index
        dp = next_dp
        backpointers.append(back)

    last_index = int(np.argmax(dp))
    best_value = float(dp[last_index])
    if best_value <= neg_inf / 2:
        order_error = any(
            raw_peaks[idx] + gap > raw_peaks[idx + 1]
            for idx in range(len(raw_peaks) - 1)
        )
        return 0.0, raw_peaks, [], order_error

    aligned = [last_index]
    for back in reversed(backpointers):
        last_index = int(back[last_index])
        aligned.append(last_index)
    aligned.reverse()

    consistency = float(np.clip(best_value / len(normalized), 0.0, 1.0))
    order_error = any(
        raw_peaks[idx] + gap > raw_peaks[idx + 1]
        for idx in range(len(raw_peaks) - 1)
    )
    return consistency, raw_peaks, aligned, order_error


def _aggregate_event_dense_info(video_id, dense_lookups):
    score_arrays = []
    for lookup in dense_lookups:
        item = lookup.get(video_id)
        if item is not None and item.get("all_scores") is not None:
            score_arrays.append(np.asarray(item["all_scores"], dtype=np.float32).reshape(-1))
    if not score_arrays:
        return None
    n_frames = min(len(scores) for scores in score_arrays)
    if n_frames <= 0:
        return None
    stacked = np.stack([scores[:n_frames] for scores in score_arrays], axis=0)
    aggregate = np.max(stacked, axis=0)
    best_idx = int(np.argmax(aggregate))
    return {
        "video_id": video_id,
        "max_score": float(aggregate[best_idx]),
        "best_frame_idx": best_idx,
        "all_scores": aggregate,
        "source": "sequence_event_max_fallback",
    }


def _write_trace(trace, config, query_id):
    sequence_config = config.get("search", {}).get("sequence_aware", {})
    if not sequence_config.get("log_evidence", True):
        return
    output_dir = Path(sequence_config.get("log_dir", "output/sequence_evidence"))
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(query_id or "query"))
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / (safe_id + ".json")).open("w", encoding="utf-8") as file_obj:
            json.dump(trace, file_obj, ensure_ascii=False, indent=2)
    except OSError as exc:
        print("SequenceAwareKIS: Canh bao khong ghi duoc evidence log: %s" % exc)


def rerank_sequence_aware_kis(
    query_text,
    fused_candidates,
    dense_searcher,
    sparse_searcher,
    query_processor,
    config,
    pre_object_candidates=None,
    query_id=None,
):
    """
    Re-rank video KIS bang evidence cua chuoi event; giu fallback query don.

    Return:
        (ranked_candidates, trace)

    Candidate schema cu van duoc giu. Cac truong diagnostic bo sung khong di
    vao submission CSV; generate_diversity_top100_kis chi doc video_id/dense_info.
    """
    sequence_config = config.get("search", {}).get("sequence_aware", {})
    enabled = bool(sequence_config.get("enabled", True))
    decomposer = KISQueryDecomposer(
        min_event_words=sequence_config.get("min_event_words", 4),
        max_events=sequence_config.get("max_events", 4),
    )
    decomposition = decomposer.decompose(query_text)
    base_trace = {
        "query_id": str(query_id or ""),
        "query": str(query_text),
        "mode": decomposition["mode"],
        "reason": decomposition["reason"],
        "events": decomposition["events"],
        "markers": decomposition["markers"],
        "coordinate_system": "keyframe_ordinal_0_based",
        "weights_are_configurable_not_claimed_optimal": True,
        "top_candidates": [],
    }

    if not enabled or decomposition["mode"] != "multi_event" or not fused_candidates:
        base_trace["applied"] = False
        base_trace["reason"] = "disabled" if not enabled else decomposition["reason"]
        _write_trace(base_trace, config, query_id)
        return list(fused_candidates or []), base_trace

    event_top_k = max(1, int(sequence_config.get("event_top_k_videos", 100)))
    pool_size = max(1, int(sequence_config.get("candidate_pool_size", 100)))
    coverage_rank = max(1, int(sequence_config.get("coverage_rank_threshold", 20)))
    rank_decay = max(1.0, float(sequence_config.get("rank_decay", 60.0)))
    temporal_min_gap = max(0, int(sequence_config.get("temporal_min_gap", 1)))

    weights = {
        "baseline": _safe_float(sequence_config.get("baseline_weight", 0.20)),
        "semantic": _safe_float(sequence_config.get("semantic_weight", 0.30)),
        "coverage": _safe_float(sequence_config.get("coverage_weight", 0.20)),
        "temporal": _safe_float(sequence_config.get("temporal_weight", 0.15)),
        "event_sparse": _safe_float(sequence_config.get("event_sparse_weight", 0.10)),
        "object": _safe_float(sequence_config.get("object_weight", 0.05)),
    }
    weight_total = sum(max(0.0, value) for value in weights.values())
    if weight_total <= 0.0:
        base_trace["applied"] = False
        base_trace["reason"] = "all_sequence_weights_are_zero"
        _write_trace(base_trace, config, query_id)
        return list(fused_candidates), base_trace

    event_dense_results = []
    event_dense_lookups = []
    event_dense_ranks = []
    event_sparse_results = []
    event_sparse_ranks = []
    event_query_info = []

    for event_text in decomposition["events"]:
        query_info = query_processor.process(event_text)
        dense_results = dense_searcher.search(
            query_info["semantic_views"],
            top_k_videos=None,
        )
        sparse_results = sparse_searcher.search(
            event_text,
            top_k_videos=event_top_k,
        )
        dense_lookup = {str(item["video_id"]): item for item in dense_results}
        dense_ranks = {
            str(item["video_id"]): rank
            for rank, item in enumerate(dense_results, start=1)
        }
        sparse_ranks = {
            str(item["video_id"]): rank
            for rank, item in enumerate(sparse_results, start=1)
        }
        event_query_info.append({
            "event_text": event_text,
            "query_en": query_info.get("query_en", ""),
            "intent": query_info.get("intent_info", {}).get("intent"),
        })
        event_dense_results.append(dense_results)
        event_dense_lookups.append(dense_lookup)
        event_dense_ranks.append(dense_ranks)
        event_sparse_results.append(sparse_results)
        event_sparse_ranks.append(sparse_ranks)

    post_candidates = list(fused_candidates)
    pre_candidates = list(pre_object_candidates or fused_candidates)
    post_lookup = {str(item.get("video_id")): item for item in post_candidates}
    pre_lookup = {str(item.get("video_id")): item for item in pre_candidates}
    pre_rank = {
        str(item.get("video_id")): rank
        for rank, item in enumerate(pre_candidates, start=1)
    }

    candidate_ids = []
    seen_ids = set()

    def add_video(video_id):
        video_id = str(video_id or "")
        if video_id and video_id not in seen_ids:
            candidate_ids.append(video_id)
            seen_ids.add(video_id)

    for item in post_candidates[:pool_size]:
        add_video(item.get("video_id"))
    for dense_results in event_dense_results:
        for item in dense_results[:event_top_k]:
            add_video(item.get("video_id"))
    for sparse_results in event_sparse_results:
        for item in sparse_results[:event_top_k]:
            add_video(item.get("video_id"))

    # ObjectSearcher da cap nhat post-object rrf_score; pre_object giu base RRF.
    object_deltas = {}
    max_positive_delta = 0.0
    max_negative_delta = 0.0
    for video_id in candidate_ids:
        post_value = _safe_float((post_lookup.get(video_id) or {}).get("rrf_score"))
        pre_value = _safe_float((pre_lookup.get(video_id) or {}).get("rrf_score"))
        delta = post_value - pre_value
        object_deltas[video_id] = delta
        max_positive_delta = max(max_positive_delta, delta)
        max_negative_delta = max(max_negative_delta, -delta)

    ranked = []
    for video_id in candidate_ids:
        dense_ranks = [rank_map.get(video_id) for rank_map in event_dense_ranks]
        sparse_ranks = [rank_map.get(video_id) for rank_map in event_sparse_ranks]
        semantic_component = float(np.mean([
            _rank_score(rank, rank_decay) for rank in dense_ranks
        ]))
        sparse_component = float(np.mean([
            _rank_score(rank, rank_decay) for rank in sparse_ranks
        ]))
        coverage_component = float(np.mean([
            1.0 if rank is not None and rank <= coverage_rank else 0.0
            for rank in dense_ranks
        ]))
        baseline_component = _rank_score(pre_rank.get(video_id), rank_decay)

        event_arrays = [lookup[video_id]["all_scores"] for lookup in event_dense_lookups]
        temporal_raw, raw_peaks, aligned_peaks, order_error = _temporal_alignment(
            event_arrays,
            min_gap=temporal_min_gap,
        )
        # Temporal bonus khong duoc cuu video khong cover event.
        temporal_component = float(temporal_raw * coverage_component)

        object_delta = object_deltas.get(video_id, 0.0)
        if object_delta > 0.0 and max_positive_delta > 0.0:
            object_component = object_delta / max_positive_delta
        elif object_delta < 0.0 and max_negative_delta > 0.0:
            object_component = object_delta / max_negative_delta
        else:
            object_component = 0.0

        components = {
            "baseline_rank_score": baseline_component,
            "semantic_event_score": semantic_component,
            "event_coverage": coverage_component,
            "temporal_consistency": temporal_component,
            "temporal_alignment_raw": temporal_raw,
            "event_sparse_score": sparse_component,
            "object_signal": float(object_component),
            "object_rrf_delta": float(object_delta),
            "raw_peak_order_error": bool(order_error),
        }
        weighted_sum = (
            weights["baseline"] * baseline_component
            + weights["semantic"] * semantic_component
            + weights["coverage"] * coverage_component
            + weights["temporal"] * temporal_component
            + weights["event_sparse"] * sparse_component
            + weights["object"] * object_component
        )
        sequence_score = float(weighted_sum / weight_total)

        source_candidate = post_lookup.get(video_id) or pre_lookup.get(video_id) or {}
        candidate = dict(source_candidate)
        candidate["video_id"] = video_id
        candidate["pre_sequence_rrf_score"] = _safe_float(source_candidate.get("rrf_score"))
        if not candidate.get("dense_info"):
            candidate["dense_info"] = _aggregate_event_dense_info(
                video_id,
                event_dense_lookups,
            )

        event_evidence = []
        for event_index, event_text in enumerate(decomposition["events"]):
            dense_item = event_dense_lookups[event_index].get(video_id, {})
            sparse_item = next(
                (
                    item for item in event_sparse_results[event_index]
                    if str(item.get("video_id")) == video_id
                ),
                {},
            )
            event_evidence.append({
                "event_index": event_index + 1,
                "event_text": event_text,
                "query_en": event_query_info[event_index]["query_en"],
                "dense_rank": dense_ranks[event_index],
                "dense_score": _safe_float(dense_item.get("max_score"), default=None),
                "sparse_rank": sparse_ranks[event_index],
                "sparse_score": _safe_float(sparse_item.get("sparse_score"), default=None),
                "raw_peak_keyframe_ordinal": raw_peaks[event_index] if raw_peaks else None,
                "aligned_keyframe_ordinal": aligned_peaks[event_index] if aligned_peaks else None,
            })

        candidate["sequence_score"] = sequence_score
        candidate["sequence_score_components"] = components
        candidate["sequence_event_evidence"] = event_evidence
        # VisualReRanker cong delta vao rrf_score, nen dua sequence score vao day.
        candidate["rrf_score"] = sequence_score
        ranked.append(candidate)

    ranked.sort(
        key=lambda item: (
            _safe_float(item.get("sequence_score")),
            _safe_float(item.get("pre_sequence_rrf_score")),
        ),
        reverse=True,
    )
    ranked = ranked[:pool_size]

    base_trace.update({
        "applied": True,
        "weights": weights,
        "coverage_rank_threshold": coverage_rank,
        "rank_decay": rank_decay,
        "temporal_min_gap": temporal_min_gap,
        "event_queries": event_query_info,
        "top_candidates": [
            {
                "rank": rank,
                "video_id": item["video_id"],
                "sequence_score": item["sequence_score"],
                "score_components": item["sequence_score_components"],
                "event_evidence": item["sequence_event_evidence"],
            }
            for rank, item in enumerate(ranked[:20], start=1)
        ],
    })
    _write_trace(base_trace, config, query_id)

    print(
        "SequenceAwareKIS: query=%s | events=%d | candidates=%d"
        % (query_id or "query", len(decomposition["events"]), len(ranked))
    )
    for rank, item in enumerate(ranked[:5], start=1):
        score_parts = item["sequence_score_components"]
        print(
            "  #%d %s | score=%.4f coverage=%.2f temporal=%.2f"
            % (
                rank,
                item["video_id"],
                item["sequence_score"],
                score_parts["event_coverage"],
                score_parts["temporal_consistency"],
            )
        )
    return ranked, base_trace
