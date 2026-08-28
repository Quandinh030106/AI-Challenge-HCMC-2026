import csv
import json
import math
import os
import re
from pathlib import Path

import cv2
import numpy as np
import torch


class TemporalRefinementError(RuntimeError):
    pass


def _finite_positive(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def _safe_float(value, default=None):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


class TemporalRefiner:
    """
    Coarse-to-fine raw-video localization cho Textual KIS.

    - Tâm cửa sổ lấy từ đúng hàng Map-Keyframes.
    - Decode từng vùng nhỏ bằng OpenCV, không load nguyên video.
    - Dùng lại image/text encoder của DenseSearcher.
    - Trả actual frame ordinal 0-based của raw MP4.
    """

    VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}

    def __init__(self, config, dense_searcher):
        self.config = config
        self.dense_searcher = dense_searcher
        self.settings = config.get("search", {}).get("temporal_refinement", {})
        self.enabled = bool(self.settings.get("enabled", True))
        self.raw_dir = Path(config.get("data", {}).get("raw_dir", ""))
        map_dir = (
            config.get("data", {}).get("map_keyframes_dir")
            or config.get("data", {}).get("metadata_dir", "")
        )
        self.map_dir = Path(map_dir)
        self._video_index = None
        self._map_path_cache = {}
        self._map_rows_cache = {}
        self._text_feature_cache = {}

    def _build_video_index(self):
        if self._video_index is not None:
            return
        self._video_index = {}
        if not self.raw_dir.is_dir():
            return

        # Quet mot lan, cache cho tat ca query. Khong quet rong /kaggle/input.
        for root, _, files in os.walk(str(self.raw_dir)):
            for filename in files:
                path = Path(root) / filename
                if path.suffix.lower() not in self.VIDEO_EXTENSIONS:
                    continue
                video_id = path.stem
                # Neu trung ID, uu tien file tim thay dau tien theo os.walk.
                self._video_index.setdefault(video_id, path)

    def _find_video_path(self, video_id):
        self._build_video_index()
        return (self._video_index or {}).get(str(video_id))

    def _find_map_path(self, video_id):
        video_id = str(video_id)
        if video_id in self._map_path_cache:
            return self._map_path_cache[video_id]

        direct_candidates = [
            self.map_dir / (video_id + ".csv"),
            self.map_dir / "map-keyframes" / (video_id + ".csv"),
        ]
        for candidate in direct_candidates:
            if candidate.is_file():
                self._map_path_cache[video_id] = candidate
                return candidate

        found = None
        if self.map_dir.is_dir():
            matches = list(self.map_dir.rglob(video_id + ".csv"))
            if matches:
                found = matches[0]
        self._map_path_cache[video_id] = found
        return found

    def _load_map_rows(self, video_id):
        video_id = str(video_id)
        if video_id in self._map_rows_cache:
            return self._map_rows_cache[video_id]

        map_path = self._find_map_path(video_id)
        if map_path is None:
            raise TemporalRefinementError(
                "Map-Keyframes CSV not found for video %s" % video_id
            )

        with map_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            normalized = {
                str(column).strip().lower(): column
                for column in (reader.fieldnames or [])
            }
            required = ("n", "pts_time", "fps", "frame_idx")
            missing = [column for column in required if column not in normalized]
            if missing:
                raise TemporalRefinementError(
                    "Map-Keyframes %s missing columns: %s"
                    % (map_path, ", ".join(missing))
                )

            rows = []
            for ordinal, raw_row in enumerate(reader):
                try:
                    row = {
                        "keyframe_ordinal": ordinal,
                        "n": int(float(raw_row[normalized["n"]])),
                        "pts_time": float(raw_row[normalized["pts_time"]]),
                        "fps": float(raw_row[normalized["fps"]]),
                        "frame_idx": int(float(raw_row[normalized["frame_idx"]])),
                    }
                except (TypeError, ValueError, OverflowError) as exc:
                    raise TemporalRefinementError(
                        "Invalid Map-Keyframes row %d in %s: %s"
                        % (ordinal + 2, map_path, exc)
                    ) from exc
                rows.append(row)

        if not rows:
            raise TemporalRefinementError("Empty Map-Keyframes CSV: %s" % map_path)
        if any(rows[index]["frame_idx"] > rows[index + 1]["frame_idx"] for index in range(len(rows) - 1)):
            raise TemporalRefinementError(
                "frame_idx is not monotonic in %s" % map_path
            )
        self._map_rows_cache[video_id] = rows
        return rows

    def _row_for_actual_frame(self, video_id, actual_frame_id):
        target = int(actual_frame_id)
        rows = self._load_map_rows(video_id)
        # Coarse predictions duoc tao tu Map-Keyframes nen phai exact-match.
        for row in rows:
            if row["frame_idx"] == target:
                return row
        raise TemporalRefinementError(
            "Coarse actual frame %s is absent from Map-Keyframes for %s"
            % (target, video_id)
        )

    def _select_anchor_prompts(
        self,
        video_id,
        coarse_keyframe_ordinal,
        default_prompts,
        fused_lookup,
        query_processor,
    ):
        candidate = fused_lookup.get(str(video_id), {})
        evidence = candidate.get("sequence_event_evidence") or []
        usable = [
            item for item in evidence
            if item.get("raw_peak_keyframe_ordinal") is not None
        ]
        if not usable or query_processor is None:
            return list(default_prompts), None

        closest = min(
            usable,
            key=lambda item: abs(
                int(item["raw_peak_keyframe_ordinal"])
                - int(coarse_keyframe_ordinal)
            ),
        )
        event_text = str(closest.get("event_text") or "").strip()
        query_en = str(closest.get("query_en") or "").strip()
        if not event_text or not query_en:
            return list(default_prompts), None

        prompts = query_processor.generate_prompt_ensemble(
            query_en,
            query_vi=event_text,
        )
        return prompts or list(default_prompts), {
            "event_index": closest.get("event_index"),
            "event_text": event_text,
            "query_en": query_en,
        }

    def _get_text_features(self, prompts):
        cache_key = tuple(str(prompt) for prompt in prompts)
        cached = self._text_feature_cache.get(cache_key)
        if cached is not None:
            return cached
        features = self.dense_searcher.encode_text_matrix(list(cache_key))
        self._text_feature_cache[cache_key] = features
        return features

    def _score_rgb_batch(self, rgb_frames, text_features):
        if not rgb_frames:
            return []
        processor = self.dense_searcher.processor
        model = self.dense_searcher.model
        device = self.dense_searcher.device

        image_inputs = processor(
            images=rgb_frames,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model.get_image_features(**image_inputs)
            if isinstance(outputs, torch.Tensor):
                image_features = outputs
            elif hasattr(outputs, "image_embeds"):
                image_features = outputs.image_embeds
            elif hasattr(outputs, "pooler_output"):
                image_features = outputs.pooler_output
            else:
                image_features = outputs[0]

            image_features = image_features / image_features.norm(
                p=2,
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-12)
            image_features = image_features.to(dtype=text_features.dtype)
            similarities = torch.matmul(image_features, text_features.T)
            if text_features.shape[0] > 1:
                max_scores = torch.max(similarities, dim=-1).values
                mean_scores = torch.mean(similarities, dim=-1)
                scores = 0.5 * max_scores + 0.5 * mean_scores
            else:
                scores = similarities.squeeze(-1)
        return [float(value) for value in scores.float().cpu().tolist()]

    def _scan_interval(
        self,
        video_path,
        start_frame,
        end_frame,
        step_frames,
        text_features,
        force_frames=None,
    ):
        batch_size = max(1, int(self.settings.get("batch_size", 8)))
        force_frames = {int(value) for value in (force_frames or [])}
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise TemporalRefinementError("Cannot open raw video: %s" % video_path)

        capture.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
        rgb_batch = []
        index_batch = []
        ranked = []

        def flush_batch():
            if not rgb_batch:
                return
            scores = self._score_rgb_batch(rgb_batch, text_features)
            ranked.extend(
                {"frame_idx": int(frame_idx), "score": float(score)}
                for frame_idx, score in zip(index_batch, scores)
            )
            rgb_batch.clear()
            index_batch.clear()

        try:
            while True:
                reported_before = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
                if reported_before > int(end_frame):
                    break
                ok, bgr_frame = capture.read()
                if not ok:
                    break
                reported_after = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
                actual_frame = reported_after - 1
                if actual_frame < int(start_frame):
                    actual_frame = reported_before
                if actual_frame > int(end_frame):
                    break

                should_score = (
                    (actual_frame - int(start_frame)) % max(1, int(step_frames)) == 0
                    or actual_frame in force_frames
                )
                if should_score:
                    rgb_batch.append(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
                    index_batch.append(actual_frame)
                    if len(rgb_batch) >= batch_size:
                        flush_batch()
        finally:
            flush_batch()
            capture.release()

        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked

    @staticmethod
    def _capture_metadata(video_path):
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise TemporalRefinementError("Cannot open raw video: %s" % video_path)
        try:
            fps = _finite_positive(capture.get(cv2.CAP_PROP_FPS))
            frame_count_raw = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            frame_count = int(round(frame_count_raw)) if frame_count_raw > 0 else None
            backend = capture.getBackendName() if hasattr(capture, "getBackendName") else "unknown"
            return {
                "capture_fps": fps,
                "frame_count": frame_count,
                "backend": backend,
            }
        finally:
            capture.release()

    def _refine_one(
        self,
        prediction_rank,
        prediction,
        default_prompts,
        fused_lookup,
        query_processor,
    ):
        video_id = str(prediction.get("video_id", ""))
        coarse_frame = int(prediction.get("frame_id"))
        map_row = self._row_for_actual_frame(video_id, coarse_frame)
        video_path = self._find_video_path(video_id)
        if video_path is None:
            raise TemporalRefinementError("Raw video not found for %s" % video_id)

        video_meta = self._capture_metadata(video_path)
        map_fps = _finite_positive(map_row.get("fps"))
        video_fps = video_meta["capture_fps"]
        effective_fps = map_fps or video_fps
        if effective_fps is None:
            raise TemporalRefinementError(
                "No valid FPS in Map-Keyframes or decoder for %s" % video_id
            )

        prompts, semantic_anchor = self._select_anchor_prompts(
            video_id=video_id,
            coarse_keyframe_ordinal=map_row["keyframe_ordinal"],
            default_prompts=default_prompts,
            fused_lookup=fused_lookup,
            query_processor=query_processor,
        )
        text_features = self._get_text_features(prompts)

        center_frame = int(map_row["frame_idx"])
        center_time = float(map_row["pts_time"])
        coarse_window = max(
            0.0,
            float(self.settings.get("coarse_window_seconds", 5.0)),
        )
        coarse_sample_fps = _finite_positive(
            self.settings.get("coarse_sample_fps", 4.0)
        )
        if coarse_sample_fps is None:
            coarse_sample_fps = effective_fps

        max_frame = (
            video_meta["frame_count"] - 1
            if video_meta["frame_count"] is not None
            else center_frame + int(round(coarse_window * effective_fps))
        )
        coarse_start = max(0, center_frame - int(round(coarse_window * effective_fps)))
        coarse_end = min(
            max_frame,
            center_frame + int(round(coarse_window * effective_fps)),
        )
        coarse_step = max(1, int(round(effective_fps / coarse_sample_fps)))
        coarse_ranked = self._scan_interval(
            video_path=video_path,
            start_frame=coarse_start,
            end_frame=coarse_end,
            step_frames=coarse_step,
            text_features=text_features,
            force_frames={center_frame},
        )
        if not coarse_ranked:
            raise TemporalRefinementError(
                "Decoder returned no coarse samples for %s" % video_id
            )

        coarse_best = int(coarse_ranked[0]["frame_idx"])
        fine_window = max(
            0.0,
            float(self.settings.get("fine_window_seconds", 0.75)),
        )
        fine_sample_fps = _finite_positive(
            self.settings.get("fine_sample_fps", 0.0)
        )
        # 0/null -> native frame-by-frame.
        fine_step = (
            1
            if fine_sample_fps is None
            else max(1, int(round(effective_fps / fine_sample_fps)))
        )
        fine_start = max(0, coarse_best - int(round(fine_window * effective_fps)))
        fine_end = min(
            max_frame,
            coarse_best + int(round(fine_window * effective_fps)),
        )
        fine_ranked = self._scan_interval(
            video_path=video_path,
            start_frame=fine_start,
            end_frame=fine_end,
            step_frames=fine_step,
            text_features=text_features,
            force_frames={center_frame, coarse_best},
        )
        if not fine_ranked:
            raise TemporalRefinementError(
                "Decoder returned no fine samples for %s" % video_id
            )

        return {
            "status": "refined",
            "prediction_rank": int(prediction_rank),
            "video_id": video_id,
            "coarse_keyframe_ordinal": int(map_row["keyframe_ordinal"]),
            "map_n": int(map_row["n"]),
            "coarse_mapped_frame": center_frame,
            "coarse_pts_time": center_time,
            "map_fps": map_fps,
            "decoder_fps": video_fps,
            "effective_fps": float(effective_fps),
            "fps_source": "map_keyframes" if map_fps is not None else "decoder",
            "decoder_backend": video_meta["backend"],
            "raw_video_path": str(video_path),
            "semantic_anchor": semantic_anchor,
            "refine_interval": {
                "coarse": {
                    "start_frame": int(coarse_start),
                    "end_frame": int(coarse_end),
                    "start_time": max(0.0, center_time - coarse_window),
                    "end_time": center_time + coarse_window,
                    "sample_fps": float(coarse_sample_fps),
                    "sample_count": len(coarse_ranked),
                },
                "fine": {
                    "start_frame": int(fine_start),
                    "end_frame": int(fine_end),
                    "sample_fps": (
                        "native_frame_by_frame"
                        if fine_sample_fps is None
                        else float(fine_sample_fps)
                    ),
                    "sample_count": len(fine_ranked),
                },
            },
            "coarse_stage_best_actual_frame": coarse_best,
            "coarse_stage_score": float(coarse_ranked[0]["score"]),
            "best_actual_frame": int(fine_ranked[0]["frame_idx"]),
            "score": float(fine_ranked[0]["score"]),
            # Giu mot danh sach ngan de xu ly duplicate giua cac slot Top-100.
            "ranked_actual_frames": fine_ranked[:20],
        }

    def _write_trace(self, trace, query_id):
        if not self.settings.get("log_evidence", True):
            return
        output_dir = Path(
            self.settings.get("log_dir", "output/temporal_refinement")
        )
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(query_id or "query"))
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            with (output_dir / (safe_id + ".json")).open(
                "w",
                encoding="utf-8",
            ) as file_obj:
                json.dump(trace, file_obj, ensure_ascii=False, indent=2)
        except OSError as exc:
            print("TemporalRefiner: Canh bao khong ghi duoc log: %s" % exc)

    def refine_kis_predictions(
        self,
        query_id,
        query_text,
        prompt_ensemble,
        coarse_predictions,
        fused_candidates=None,
        query_processor=None,
    ):
        """
        Refine mot so slot dau cua KIS Top-100.

        Output predictions van chi co video_id/frame_id. frame_id da la actual
        raw-video frame ordinal va KHONG duoc map them lan nua.
        """
        predictions = [dict(item) for item in (coarse_predictions or [])]
        trace = {
            "query_id": str(query_id or ""),
            "query": str(query_text or ""),
            "enabled": self.enabled,
            "coordinate_system": "actual_raw_video_frame_ordinal_0_based",
            "refinements": [],
        }
        if not self.enabled or not predictions:
            trace["status"] = "disabled_or_empty"
            self._write_trace(trace, query_id)
            return predictions, trace

        max_predictions = min(
            len(predictions),
            max(0, int(self.settings.get("top_predictions", 5))),
        )
        if max_predictions <= 0:
            trace["status"] = "no_requested_slots"
            self._write_trace(trace, query_id)
            return predictions, trace

        fused_lookup = {
            str(item.get("video_id")): item
            for item in (fused_candidates or [])
        }
        reserved_by_other_slots = {
            (str(item.get("video_id")), str(item.get("frame_id")))
            for item in predictions
        }
        emitted = set()

        for index in range(max_predictions):
            prediction = predictions[index]
            original_pair = (
                str(prediction.get("video_id")),
                str(prediction.get("frame_id")),
            )
            reserved_by_other_slots.discard(original_pair)
            try:
                result = self._refine_one(
                    prediction_rank=index + 1,
                    prediction=prediction,
                    default_prompts=prompt_ensemble,
                    fused_lookup=fused_lookup,
                    query_processor=query_processor,
                )
                chosen = None
                for raw_candidate in result["ranked_actual_frames"]:
                    pair = (
                        str(prediction.get("video_id")),
                        str(raw_candidate["frame_idx"]),
                    )
                    if pair not in emitted and pair not in reserved_by_other_slots:
                        chosen = raw_candidate
                        break

                if chosen is None:
                    result["status"] = "fallback_duplicate_guard"
                    result["best_actual_frame"] = int(prediction["frame_id"])
                    result["score"] = None
                else:
                    prediction["frame_id"] = str(int(chosen["frame_idx"]))
                    result["best_actual_frame"] = int(chosen["frame_idx"])
                    result["score"] = float(chosen["score"])
                result.pop("ranked_actual_frames", None)
            except Exception as exc:
                result = {
                    "status": "fallback_coarse",
                    "prediction_rank": index + 1,
                    "video_id": str(prediction.get("video_id", "")),
                    "coarse_mapped_frame": prediction.get("frame_id"),
                    "best_actual_frame": prediction.get("frame_id"),
                    "score": None,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }

            final_pair = (
                str(prediction.get("video_id")),
                str(prediction.get("frame_id")),
            )
            emitted.add(final_pair)
            reserved_by_other_slots.add(final_pair)
            trace["refinements"].append(result)

        trace["status"] = "completed"
        self._write_trace(trace, query_id)
        return predictions, trace


    def refine_trake_event(
        self,
        video_id,
        coarse_actual_frame,
        prompts,
        lower_bound_frame,
        upper_bound_frame,
        window_seconds=None,
        sample_fps=None,
    ):
        """
        Tinh chinh actual frame cho MOT event TRAKE, GIOI HAN trong
        [lower_bound_frame, upper_bound_frame] de khong pha vo tinh don
        dieu E1 < E2 < ... < EN (Prompt 8).

        window_seconds/sample_fps cho phep TRAKE dung cau hinh rieng
        (search.trake_alignment.raw_refine) thay vi bat buoc dung chung
        settings cua KIS (search.temporal_refinement); None -> fallback
        ve settings KIS mac dinh.

        Tra ve dict {"actual_frame", "score", "search_start_frame",
        "search_end_frame", "effective_fps"} hoac None neu khong the refine
        (video khong tim thay, decode loi...). Caller PHAI tu fallback ve
        coarse frame khi nhan None.
        """
        if not self.enabled:
            return None

        video_path = self._find_video_path(video_id)
        if video_path is None:
            return None

        video_meta = self._capture_metadata(video_path)
        effective_fps = video_meta["capture_fps"]
        if effective_fps is None:
            return None

        text_features = self._get_text_features(prompts)

        fine_window = max(
            0.0,
            float(
                self.settings.get("fine_window_seconds", 0.75)
                if window_seconds is None else window_seconds
            ),
        )
        configured_sample_fps = (
            self.settings.get("fine_sample_fps", 0.0)
            if sample_fps is None else sample_fps
        )
        fine_sample_fps = _finite_positive(configured_sample_fps)
        fine_step = (
            1 if fine_sample_fps is None
            else max(1, int(round(effective_fps / fine_sample_fps)))
        )

        window_frames = int(round(fine_window * effective_fps))
        start_frame = max(int(lower_bound_frame), int(coarse_actual_frame) - window_frames)
        end_frame = min(int(upper_bound_frame), int(coarse_actual_frame) + window_frames)
        if start_frame > end_frame:
            clamped = max(
                int(lower_bound_frame),
                min(int(upper_bound_frame), int(coarse_actual_frame)),
            )
            start_frame = end_frame = clamped

        force_frames = (
            {int(coarse_actual_frame)}
            if start_frame <= int(coarse_actual_frame) <= end_frame
            else set()
        )

        try:
            ranked = self._scan_interval(
                video_path=video_path,
                start_frame=start_frame,
                end_frame=end_frame,
                step_frames=fine_step,
                text_features=text_features,
                force_frames=force_frames,
            )
        except TemporalRefinementError as exc:
            print("TemporalRefiner: TRAKE refine warning video=%s: %s" % (video_id, exc))
            return None

        if not ranked:
            return None

        best = ranked[0]
        return {
            "actual_frame": int(best["frame_idx"]),
            "score": float(best["score"]),
            "search_start_frame": start_frame,
            "search_end_frame": end_frame,
            "effective_fps": float(effective_fps),
        }