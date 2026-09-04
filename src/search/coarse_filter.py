"""
Coarse Video Filter
AI Challenge HCMC 2026

Stage 0:
    Fast video-level filtering before expensive retrieval.

Input:
    query_info from QueryProcessor

Output:
    ranked video_ids

Uses:
    - semantic_query
    - metadata
    - objects
    - actions
    - keywords

Does NOT:
    - retrieve frames
    - modify mapping
    - modify submission
"""


import json
from pathlib import Path
from collections import defaultdict



class CoarseFilter:


    def __init__(
        self,
        index_path="data/index/video_semantic_index.json",
        top_k=100
    ):

        self.index_path = Path(index_path)

        self.top_k = top_k

        self.video_index = {}

        self.loaded = False


        # synonym expansion
        self.synonyms = {

            "squid":
            [
                "squid",
                "cuttlefish",
                "seafood",
                "fish"
            ],

            "green peas":
            [
                "green pea",
                "pea",
                "vegetable"
            ],

            "car":
            [
                "car",
                "vehicle",
                "automobile"
            ],

            "bicycle":
            [
                "bicycle",
                "bike",
                "cycle"
            ],

            "person":
            [
                "person",
                "man",
                "woman",
                "people"
            ],

            "cooking":
            [
                "cook",
                "cooking",
                "food",
                "kitchen"
            ],

            "stir frying":
            [
                "stir",
                "frying",
                "cook",
                "cooking"
            ],

            "cutting":
            [
                "cut",
                "cutting",
                "knife"
            ]

        }


        self.load_index()



    # ==================================================
    # LOAD INDEX
    # ==================================================

    def load_index(self):

        if not self.index_path.exists():

            raise FileNotFoundError(
                f"Cannot find semantic index: "
                f"{self.index_path}"
            )


        with open(
            self.index_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)



        self.video_index = (
            data.get(
                "videos",
                {}
            )
        )


        self.loaded = True


        print(
            "CoarseFilter: loaded "
            f"{len(self.video_index)} videos"
        )



    # ==================================================
    # TEXT NORMALIZATION
    # ==================================================

    def normalize(
        self,
        text
    ):

        if text is None:

            return ""


        return (
            str(text)
            .lower()
            .replace("_", " ")
            .strip()
        )



    # ==================================================
    # EXPAND TERM
    # ==================================================

    def expand_term(
        self,
        term
    ):

        term = self.normalize(
            term
        )


        result = {
            term
        }


        for key, values in self.synonyms.items():

            if (
                term == key
                or
                term in values
            ):

                result.update(
                    values
                )


        return result



    # ==================================================
    # EXTRACT QUERY INFORMATION
    # ==================================================

    def extract_query_terms(
        self,
        query_info
    ):


        semantic = (
            query_info.get(
                "semantic_query",
                {}
            )
        )


        result = {

            "objects": set(),

            "actions": set(),

            "scene": set(),

            "keywords": set(),

            "text": ""

        }



        # Objects

        for obj in semantic.get(
            "objects",
            []
        ):

            result["objects"].update(
                self.expand_term(obj)
            )



        # Actions

        for action in semantic.get(
            "actions",
            []
        ):

            result["actions"].update(
                self.expand_term(action)
            )



        # Scene

        scene = semantic.get(
            "scene",
            ""
        )


        if scene:

            result["scene"].update(
                self.expand_term(scene)
            )



        # Literal translation

        result["text"] = self.normalize(
            query_info.get(
                "query_en",
                ""
            )
        )


        # Add visual description

        result["text"] += " " + self.normalize(
            query_info.get(
                "visual_description",
                ""
            )
        )


        return result



    # ==================================================
    # MATCH OBJECT
    # ==================================================

    def object_score(
        self,
        query_objects,
        video_objects
    ):

        if not query_objects:

            return 0


        score = 0


        for q in query_objects:


            matched = False


            for obj in video_objects:


                if (
                    q in obj
                    or
                    obj in q
                ):

                    score += 1

                    matched = True

                    break



            if matched:

                continue



        return score / len(query_objects)



    # ==================================================
    # MATCH ACTION
    # ==================================================

    def action_score(
        self,
        actions,
        metadata_text
    ):

        if not actions:

            return 0


        score = 0


        for action in actions:


            if action in metadata_text:

                score += 1



        return score / len(actions)



    # ==================================================
    # MATCH METADATA
    # ==================================================

    def metadata_score(
        self,
        query_text,
        metadata
    ):


        if not query_text:

            return 0



        corpus = " ".join(
            [

                metadata.get(
                    "title",
                    ""
                ),

                metadata.get(
                    "description",
                    ""
                ),

                metadata.get(
                    "keyword_text",
                    ""
                )

            ]
        )


        corpus = self.normalize(
            corpus
        )


        words = [
            w
            for w in query_text.split()
            if len(w) >= 3
        ]


        if not words:

            return 0



        hit = 0


        for w in words:

            if w in corpus:

                hit += 1



        return hit / len(words)



    # ==================================================
    # VIDEO SCORE
    # ==================================================

    def score_video(
        self,
        video_data,
        query_terms
    ):


        metadata = (
            video_data.get(
                "metadata",
                {}
            )
        )


        objects = (
            video_data
            .get(
                "objects",
                {}
            )
        )


        object_list = [

            self.normalize(x)

            for x in objects.get(
                "object_list",
                []
            )

        ]


        metadata_text = self.normalize(
            " ".join(
                [
                    metadata.get(
                        "title",
                        ""
                    ),

                    metadata.get(
                        "description",
                        ""
                    ),

                    metadata.get(
                        "keyword_text",
                        ""
                    )
                ]
            )
        )



        obj_score = self.object_score(
            query_terms["objects"],
            object_list
        )


        action_score = self.action_score(
            query_terms["actions"],
            metadata_text
        )


        meta_score = self.metadata_score(
            query_terms["text"],
            metadata
        )



        total = (

            0.50 * obj_score

            +

            0.20 * action_score

            +

            0.20 * meta_score

            +

            0.10 * (
                1
                if query_terms["scene"]
                and any(
                    s in metadata_text
                    for s in query_terms["scene"]
                )
                else 0
            )

        )


        return total



    # ==================================================
    # FILTER
    # ==================================================

    def filter(
        self,
        query_info,
        top_k=None
    ):


        if top_k is None:

            top_k = self.top_k



        query_terms = (
            self.extract_query_terms(
                query_info
            )
        )


        results = []


        for video_id, video_data in (
            self.video_index.items()
        ):


            score = self.score_video(
                video_data,
                query_terms
            )


            results.append(
                (
                    video_id,
                    score
                )
            )



        results.sort(
            key=lambda x:x[1],
            reverse=True
        )



        candidates = [

            item[0]

            for item in results[:top_k]

        ]



        # Safety fallback

        if not candidates:

            candidates = list(
                self.video_index.keys()
            )[:top_k]



        return candidates