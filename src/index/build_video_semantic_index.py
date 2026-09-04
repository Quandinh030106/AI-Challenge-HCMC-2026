"""
Build video semantic index for AI Challenge HCMC 2026.

Purpose:
    Create lightweight video-level semantic information
    from metadata and object detection outputs.

Input:
    metadata_dir:
        media-info-aic25-b1/media-info/*.json

    objects_dir:
        objects-aic25-b1/objects/<video_id>/*.json

Output:
    data/index/video_semantic_index.json


Important:
    This module DOES NOT:
    - modify retrieval pipeline
    - touch frame mapping
    - touch submission format

It only creates offline index for coarse filtering.
"""

import os
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import yaml
from tqdm import tqdm


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def normalize_text(value):
    if value is None:
        return ""

    if isinstance(value, list):
        return " ".join(
            str(x)
            for x in value
            if x
        )

    return str(value)


# ---------------------------------------------------------
# Metadata processing
# ---------------------------------------------------------

def extract_metadata(meta):
    """
    Extract searchable metadata fields.
    """

    keywords = meta.get(
        "keywords",
        []
    )

    if not isinstance(keywords, list):
        keywords = [keywords]


    keyword_text = " ".join(
        str(k)
        for k in keywords
        if k
    )


    return {
        "title": normalize_text(
            meta.get("title")
        ),

        "description": normalize_text(
            meta.get("description")
        ),

        "keywords": keywords,

        "keyword_text": keyword_text,

        "author": normalize_text(
            meta.get("author")
        ),

        "publish_date": normalize_text(
            meta.get("publish_date")
        ),

        "length": meta.get("length"),
    }



# ---------------------------------------------------------
# Object processing
# ---------------------------------------------------------

def extract_objects(object_dir):
    """
    Aggregate object detections from all keyframes.

    Example:
        001.json
        002.json
        ...
        307.json


    Output:

    {
       "person": 120,
       "vehicle": 30
    }

    """

    object_counter = Counter()

    frame_count = 0


    if not os.path.isdir(object_dir):
        return {
            "object_counts": {},
            "object_list": [],
            "object_frame_count": 0,
        }


    json_files = sorted(
        Path(object_dir).glob("*.json")
    )


    for json_file in json_files:

        try:
            data = load_json(json_file)

        except Exception:
            continue


        frame_count += 1


        entities = data.get(
            "detection_class_entities",
            []
        )


        names = data.get(
            "detection_class_names",
            []
        )


        scores = data.get(
            "detection_scores",
            []
        )


        for idx, entity in enumerate(entities):

            score = 1.0

            if idx < len(scores):
                try:
                    score = float(scores[idx])
                except Exception:
                    score = 1.0


            # bỏ detection yếu
            if score < 0.3:
                continue


            label = str(entity).strip().lower()


            if label:
                object_counter[label] += 1



        # fallback nếu entity không có
        if not entities:

            for idx, name in enumerate(names):

                score = 1.0

                if idx < len(scores):
                    try:
                        score = float(scores[idx])
                    except Exception:
                        pass


                if score < 0.3:
                    continue


                label = str(name).strip().lower()

                if label:
                    object_counter[label] += 1



    return {

        "object_counts":
            dict(object_counter),

        "object_list":
            list(object_counter.keys()),

        "object_frame_count":
            frame_count,

    }



# ---------------------------------------------------------
# Build index
# ---------------------------------------------------------

def build_index(
    metadata_dir,
    objects_dir,
):

    metadata_dir = Path(metadata_dir)
    objects_dir = Path(objects_dir)


    index = {}


    metadata_files = sorted(
        metadata_dir.glob("*.json")
    )


    print(
        "Found metadata files:",
        len(metadata_files)
    )


    for meta_file in tqdm(
        metadata_files,
        desc="Building semantic index"
    ):

        video_id = meta_file.stem


        try:
            metadata = load_json(meta_file)

        except Exception:
            continue



        item = {

            "video_id":
                video_id,

            "metadata":
                extract_metadata(
                    metadata
                ),

            "objects":
                extract_objects(
                    objects_dir / video_id
                ),

        }



        index[video_id] = item



    return index



# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def load_config(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return yaml.safe_load(f)



def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--config",
        default="configs/local_windows.yaml"
    )


    parser.add_argument(
        "--output",
        default="data/index/video_semantic_index.json"
    )


    args = parser.parse_args()



    config = load_config(
        args.config
    )


    metadata_dir = (
        config["data"]
        .get("metadata_dir")
    )


    objects_dir = (
        config["data"]
        .get("objects_dir")
    )


    print("="*60)
    print(
        "BUILD VIDEO SEMANTIC INDEX"
    )
    print("="*60)


    print(
        "Metadata:",
        metadata_dir
    )

    print(
        "Objects:",
        objects_dir
    )


    index = build_index(
        metadata_dir,
        objects_dir,
    )


    save_json(
        {
            "project":
                "AI Challenge HCMC 2026",

            "video_count":
                len(index),

            "videos":
                index,

        },
        args.output,
    )


    print()
    print(
        "Completed."
    )

    print(
        "Indexed videos:",
        len(index)
    )

    print(
        "Output:",
        args.output
    )



if __name__ == "__main__":
    main()