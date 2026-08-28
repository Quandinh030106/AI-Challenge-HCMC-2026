from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def natural_key(path: Path):
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"(\d+)", path.stem)
    ]


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def video_id_from_path(path: Path) -> str:
    return path.stem


def collect_videos(root: Path) -> dict[str, Path]:
    result = {}
    if not root.exists():
        return result
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            result[video_id_from_path(p)] = p
    return result


def collect_keyframe_dirs(root: Path) -> dict[str, Path]:
    """
    Không giả định keyframe là 0000.jpg/001.jpg.
    Một thư mục được coi là video keyframe folder nếu chứa ảnh trực tiếp bên trong.
    """
    result = {}
    if not root.exists():
        return result

    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        try:
            has_image = any(
                p.is_file() and p.suffix.lower() in IMAGE_EXTS
                for p in d.iterdir()
            )
        except OSError:
            continue
        if has_image:
            result[d.name] = d
    return result


def collect_named_files(root: Path, suffix: str) -> dict[str, Path]:
    result = {}
    if not root.exists():
        return result
    for p in root.rglob(f"*{suffix}"):
        if p.is_file():
            result[p.stem] = p
    return result


def collect_object_dirs(root: Path) -> dict[str, Path]:
    result = {}
    if not root.exists():
        return result
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        try:
            has_json = any(p.is_file() and p.suffix.lower() == ".json" for p in d.iterdir())
        except OSError:
            continue
        if has_json:
            result[d.name] = d
    return result


def keyframe_files(folder: Path) -> list[Path]:
    files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return sorted(files, key=natural_key)


def object_files(folder: Path) -> list[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".json"],
        key=natural_key
    )


def read_map_csv(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []

    normalized = {str(c).strip().lower(): c for c in fields}
    required = ["n", "pts_time", "fps", "frame_idx"]
    missing = [c for c in required if c not in normalized]

    info = {
        "count": len(rows),
        "columns": fields,
        "missing_columns": missing,
        "frame_idx_monotonic": False,
        "n_monotonic": False,
        "fps_values": [],
        "first": None,
        "last": None,
    }

    if missing or not rows:
        return info

    try:
        ns = [int(float(r[normalized["n"]])) for r in rows]
        pts = [float(r[normalized["pts_time"]]) for r in rows]
        fps = [float(r[normalized["fps"]]) for r in rows]
        frame_idx = [int(float(r[normalized["frame_idx"]])) for r in rows]

        info["frame_idx_monotonic"] = all(
            frame_idx[i] <= frame_idx[i + 1] for i in range(len(frame_idx) - 1)
        )
        info["n_monotonic"] = all(
            ns[i] < ns[i + 1] for i in range(len(ns) - 1)
        )
        info["fps_values"] = sorted(set(fps))[:20]
        info["first"] = {
            "n": ns[0], "pts_time": pts[0], "fps": fps[0], "frame_idx": frame_idx[0]
        }
        info["last"] = {
            "n": ns[-1], "pts_time": pts[-1], "fps": fps[-1], "frame_idx": frame_idx[-1]
        }
    except (ValueError, TypeError, KeyError):
        pass

    return info


def inspect_video(
    video_id: str,
    videos: dict[str, Path],
    kf_dirs: dict[str, Path],
    features: dict[str, Path],
    maps: dict[str, Path],
    object_dirs: dict[str, Path],
    metadata: dict[str, Path],
) -> dict:
    errors = []
    warnings = []

    kf_dir = kf_dirs.get(video_id)
    kfs = keyframe_files(kf_dir) if kf_dir else []

    feat_path = features.get(video_id)
    feature_rows = None
    feature_dim = None
    if feat_path:
        try:
            arr = np.load(feat_path, mmap_mode="r")
            feature_rows = int(arr.shape[0]) if arr.ndim >= 1 else 0
            feature_dim = int(arr.shape[1]) if arr.ndim >= 2 else None
        except Exception as exc:
            errors.append(f"feature_load_error: {exc}")

    map_path = maps.get(video_id)
    map_info = read_map_csv(map_path) if map_path else None

    obj_dir = object_dirs.get(video_id)
    objs = object_files(obj_dir) if obj_dir else []

    if not videos.get(video_id):
        warnings.append("missing_video")
    if not kf_dir:
        errors.append("missing_keyframes")
    if not feat_path:
        errors.append("missing_features")
    if not map_path:
        errors.append("missing_map_keyframes")
    if not obj_dir:
        warnings.append("missing_objects")
    if not metadata.get(video_id):
        warnings.append("missing_metadata")

    if kfs and feature_rows is not None and len(kfs) != feature_rows:
        errors.append(
            f"feature_rows_mismatch: keyframes={len(kfs)}, features={feature_rows}"
        )

    if kfs and map_info and len(kfs) != map_info["count"]:
        errors.append(
            f"map_rows_mismatch: keyframes={len(kfs)}, map={map_info['count']}"
        )

    if kfs and objs and len(kfs) != len(objs):
        errors.append(
            f"object_count_mismatch: keyframes={len(kfs)}, objects={len(objs)}"
        )

    if map_info:
        if map_info["missing_columns"]:
            errors.append(
                "map_missing_columns: " + ",".join(map_info["missing_columns"])
            )
        if map_info["count"] > 1 and not map_info["frame_idx_monotonic"]:
            errors.append("map_frame_idx_not_monotonic")
        if map_info["count"] > 1 and not map_info["n_monotonic"]:
            warnings.append("map_n_not_strictly_increasing")

    # BTC nói object JSON có cùng tên với keyframe tương ứng.
    filename_alignment_ok = None
    if kfs and objs and len(kfs) == len(objs):
        kf_stems = [p.stem for p in kfs]
        obj_stems = [p.stem for p in objs]
        filename_alignment_ok = (kf_stems == obj_stems)
        if not filename_alignment_ok:
            errors.append("keyframe_object_filename_alignment_mismatch")

    return {
        "video_id": video_id,
        "status": "error" if errors else ("warning" if warnings else "ok"),
        "errors": errors,
        "warnings": warnings,
        "video": str(videos[video_id]) if video_id in videos else None,
        "keyframes": {
            "directory": str(kf_dir) if kf_dir else None,
            "count": len(kfs),
            "first": kfs[0].name if kfs else None,
            "last": kfs[-1].name if kfs else None,
        },
        "features": {
            "path": str(feat_path) if feat_path else None,
            "rows": feature_rows,
            "dimension": feature_dim,
        },
        "mapping": {
            "path": str(map_path) if map_path else None,
            **(map_info or {})
        },
        "objects": {
            "directory": str(obj_dir) if obj_dir else None,
            "count": len(objs),
            "filename_alignment_ok": filename_alignment_ok,
        },
        "metadata": {
            "path": str(metadata[video_id]) if video_id in metadata else None,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Audit toàn bộ dataset AIC 2026 mà không thay đổi inference pipeline."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--output",
        default="output/dataset_audit_full.json"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 = audit toàn bộ; >0 = chỉ audit N video đầu."
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    data = cfg.get("data", {})

    raw_dir = Path(data.get("raw_dir", ""))
    keyframes_dir = Path(data.get("keyframes_dir", ""))
    features_dir = Path(data.get("features_dir", ""))
    map_dir = Path(data.get("map_keyframes_dir", ""))
    objects_dir = Path(data.get("objects_dir", ""))
    metadata_dir = Path(data.get("metadata_dir", ""))

    print("Discovering dataset...")
    videos = collect_videos(raw_dir)
    kf_dirs = collect_keyframe_dirs(keyframes_dir)
    features = collect_named_files(features_dir, ".npy")
    maps = collect_named_files(map_dir, ".csv")
    object_dirs = collect_object_dirs(objects_dir)
    metadata = collect_named_files(metadata_dir, ".json")

    all_ids = sorted(
        set(videos) | set(kf_dirs) | set(features) | set(maps) | set(object_dirs) | set(metadata)
    )

    if args.limit > 0:
        all_ids = all_ids[:args.limit]

    rows = []
    for i, video_id in enumerate(all_ids, start=1):
        if i % 50 == 0 or i == 1 or i == len(all_ids):
            print(f"[{i}/{len(all_ids)}] {video_id}")
        rows.append(
            inspect_video(
                video_id, videos, kf_dirs, features, maps, object_dirs, metadata
            )
        )

    counts = defaultdict(int)
    for r in rows:
        counts[r["status"]] += 1

    report = {
        "config": str(config_path),
        "paths": {
            "raw_dir": str(raw_dir),
            "keyframes_dir": str(keyframes_dir),
            "features_dir": str(features_dir),
            "map_keyframes_dir": str(map_dir),
            "objects_dir": str(objects_dir),
            "metadata_dir": str(metadata_dir),
        },
        "discovered": {
            "videos": len(videos),
            "keyframe_video_dirs": len(kf_dirs),
            "feature_files": len(features),
            "map_files": len(maps),
            "object_video_dirs": len(object_dirs),
            "metadata_files": len(metadata),
            "union_video_ids": len(set(videos) | set(kf_dirs) | set(features) | set(maps) | set(object_dirs)),
        },
        "audited_count": len(rows),
        "summary": dict(counts),
        "videos": rows,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("=" * 72)
    print("AUDIT COMPLETE")
    print(json.dumps(report["discovered"], ensure_ascii=False, indent=2))
    print("Status:", dict(counts))
    print("Report:", output_path.resolve())
    print("=" * 72)

    if counts["error"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
