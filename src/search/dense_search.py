import os
import hashlib
import numpy as np
import torch

from transformers import (
    CLIPModel,
    CLIPProcessor,
    SiglipModel,
    SiglipProcessor,
    AutoModel,
    AutoProcessor
)


class DenseSearcher:

    def __init__(self, config):

        self.config = config

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )


        self.model_name = (
            config["models"]["clip_model"]
        )


        self.features_dir = (
            config["data"]["features_dir"]
        )

        self.similarity_batch_size = int(
            config.get("search", {}).get(
                "dense_similarity_batch_size",
                8192,
            )
        )


        print(
            f"DenseSearcher: Loading {self.model_name} "
            f"on {self.device}"
        )


        # ==================================================
        # Load CLIP / SigLIP model
        # ==================================================

        if "siglip" in self.model_name.lower():

            self.processor = (
                SiglipProcessor
                .from_pretrained(
                    self.model_name
                )
            )

            self.model = (
                SiglipModel
                .from_pretrained(
                    self.model_name
                )
                .to(self.device)
            )


        elif "clip" in self.model_name.lower():

            self.processor = (
                CLIPProcessor
                .from_pretrained(
                    self.model_name
                )
            )

            self.model = (
                CLIPModel
                .from_pretrained(
                    self.model_name
                )
                .to(self.device)
            )


        else:

            self.processor = (
                AutoProcessor
                .from_pretrained(
                    self.model_name
                )
            )

            self.model = (
                AutoModel
                .from_pretrained(
                    self.model_name
                )
                .to(self.device)
            )


        self.model.eval()



        # ==================================================
        # Feature storage
        # ==================================================

        self.global_tensor = None


        self.video_metadata_list = []


        self.video_metadata_dict = {}


        self.all_video_ids = []


        self.video_features_dict = {}



        # ==================================================
        # Query embedding cache
        # ==================================================

        self.query_embedding_cache = {}



        # ==================================================
        # Load feature matrix
        # ==================================================

        self.load_and_build_global_matrix()



    # ======================================================
    # LOAD FEATURE MATRIX
    # ======================================================

    def load_and_build_global_matrix(self):


        print(
            "DenseSearcher: Loading feature files..."
        )


        feature_files = []


        if (
            self.features_dir
            and os.path.exists(
                self.features_dir
            )
        ):

            for root, _, files in os.walk(
                self.features_dir
            ):

                for file in files:

                    if file.endswith(".npy"):

                        feature_files.append(
                            os.path.join(
                                root,
                                file
                            )
                        )


        feature_files = sorted(
            feature_files
        )


        if not feature_files:

            print(
                "DenseSearcher: No feature files found."
            )

            return



        print(
            f"DenseSearcher: Found "
            f"{len(feature_files)} feature files"
        )


        all_vectors = []


        current_idx = 0



        for file_path in feature_files:


            video_id = (
                os.path.splitext(
                    os.path.basename(
                        file_path
                    )
                )[0]
            )


            try:

                feats = np.load(
                    file_path
                )


                # L2 normalize

                norms = np.linalg.norm(
                    feats,
                    axis=-1,
                    keepdims=True
                )


                norms[
                    norms == 0
                ] = 1e-10


                feats_norm = (
                    feats /
                    norms
                )



                n_frames = (
                    feats_norm.shape[0]
                )


                self.video_features_dict[
                    video_id
                ] = feats_norm



                all_vectors.append(
                    feats_norm
                )


                metadata = {

                    "video_id":
                        video_id,

                    "start_idx":
                        current_idx,

                    "end_idx":
                        current_idx + n_frames,

                    "n_frames":
                        n_frames
                }


                self.video_metadata_list.append(
                    metadata
                )


                self.video_metadata_dict[
                    video_id
                ] = metadata



                self.all_video_ids.append(
                    video_id
                )


                current_idx += n_frames



            except Exception as exc:

                print(
                    "DenseSearcher: skip",
                    file_path,
                    exc
                )



        if all_vectors:


            concat_matrix = np.vstack(
                all_vectors
            )


            tensor = torch.from_numpy(
                concat_matrix
            )


            if self.device == "cuda":

                self.global_tensor = (
                    tensor
                    .half()
                    .to(
                        self.device
                    )
                )

            else:

                self.global_tensor = (
                    tensor
                    .float()
                )



            print(
                "DenseSearcher: Global tensor",
                self.global_tensor.shape
            )


            print(
                "Videos:",
                len(
                    self.all_video_ids
                )
            )



    # ======================================================
    # QUERY CACHE KEY
    # ======================================================

    def _cache_key(
        self,
        texts
    ):

        if isinstance(
            texts,
            str
        ):

            texts = [
                texts
            ]


        raw = "|||".join(
            texts
        )


        return hashlib.md5(
            raw.encode(
                "utf-8"
            )
        ).hexdigest()



    # ======================================================
    # ENCODE TEXT MATRIX
    # ======================================================

    def encode_text_matrix(
        self,
        text_or_list
    ):


        if isinstance(
            text_or_list,
            str
        ):

            text_inputs = [
                text_or_list
            ]

        else:

            text_inputs = list(
                text_or_list
            )



        cache_key = self._cache_key(
            text_inputs
        )



        if cache_key in self.query_embedding_cache:

            return (
                self.query_embedding_cache[
                    cache_key
                ]
            )



        inputs = self.processor(
            text=text_inputs,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(
            self.device
        )



        with torch.no_grad():


            if hasattr(
                self.model,
                "get_text_features"
            ):

                outputs = (
                    self.model
                    .get_text_features(
                        **inputs
                    )
                )


            else:

                outputs = (
                    self.model(
                        **inputs
                    )
                )



            if isinstance(
                outputs,
                torch.Tensor
            ):

                text_features = outputs


            elif hasattr(
                outputs,
                "text_embeds"
            ):

                text_features = (
                    outputs.text_embeds
                )


            elif hasattr(
                outputs,
                "pooler_output"
            ):

                text_features = (
                    outputs.pooler_output
                )


            else:

                text_features = outputs[0]



            text_features = (
                text_features /
                text_features.norm(
                    p=2,
                    dim=-1,
                    keepdim=True
                )
            )



            if self.device == "cuda":

                text_features = (
                    text_features
                    .half()
                )


            else:

                text_features = (
                    text_features
                    .float()
                )



        self.query_embedding_cache[
            cache_key
        ] = text_features



        return text_features



    # ======================================================
    # SINGLE / MEAN ENCODE
    # ======================================================

    def encode_text(
        self,
        text_or_list
    ):


        matrix = (
            self.encode_text_matrix(
                text_or_list
            )
        )


        if matrix.shape[0] > 1:


            vec = (
                matrix.mean(
                    dim=0,
                    keepdim=True
                )
            )


            return (
                vec /
                vec.norm(
                    p=2,
                    dim=-1,
                    keepdim=True
                )
            )



        return matrix



    # ======================================================
    # SEARCH
    # ======================================================

    def search(
        self,
        query_input,
        top_k_videos=100,
        candidate_video_ids=None,
        coarse_filtering=True
    ):

        """
        Multi semantic retrieval.

        Input:

            query_text_or_ensemble:
                [
                    literal,
                    scene,
                    objects,
                    actions
                ]


            candidate_video_ids:
                from CoarseFilter


        Output:

            [
              {
                video_id,
                max_score,
                best_frame_idx,
                all_scores
              }
            ]
        """



        if self.global_tensor is None:

            return []



        # -----------------------------------------
        # Build semantic views
        # -----------------------------------------

        semantic_views = (
            self.normalize_query_views(
                query_input
            )
        )


        q_matrix = (
            self.encode_dynamic_query(
                semantic_views
            )
        )



        # -----------------------------------------
        # Candidate video filtering
        # -----------------------------------------

        candidate_metadata = (
            self.video_metadata_list
        )



        if candidate_video_ids is not None:


            candidate_set = set(
                str(v)
                for v in candidate_video_ids
            )


            candidate_metadata = [

                meta

                for meta in self.video_metadata_list

                if meta["video_id"]
                in candidate_set

            ]



            if not candidate_metadata:


                print(
                    "DenseSearcher:"
                    " no valid coarse candidates."
                )


                # fallback baseline

                candidate_metadata = (
                    self.video_metadata_list
                )



        # -----------------------------------------
        # Build search tensor
        # -----------------------------------------

        if (
            candidate_video_ids is None
            or
            not coarse_filtering
        ):


            search_tensor = (
                self.global_tensor
            )


        else:


            candidate_vectors = []


            for meta in candidate_metadata:


                candidate_vectors.append(

                    self.global_tensor[
                        meta["start_idx"]:
                        meta["end_idx"]
                    ]

                )


            search_tensor = torch.cat(
                candidate_vectors,
                dim=0
            )



        # -----------------------------------------
        # Similarity
        # -----------------------------------------

        with torch.no_grad():

            score_chunks = []
            batch_size = max(1, self.similarity_batch_size)
            for start in range(0, search_tensor.shape[0], batch_size):
                end = start + batch_size
                sim_matrix = torch.matmul(
                    search_tensor[start:end],
                    q_matrix.T,
                )
                score_chunks.append(
                    self.weighted_similarity(sim_matrix)
                )

            sim_scores = torch.cat(score_chunks, dim=0)


            sim_scores_np = (
                sim_scores
                .float()
                .cpu()
                .numpy()
            )



        # -----------------------------------------
        # Video aggregation
        # -----------------------------------------

        results = []

        current_offset = 0



        for meta in candidate_metadata:


            video_id = (
                meta["video_id"]
            )



            if (
                candidate_video_ids is not None
                and
                coarse_filtering
            ):


                start_i = current_offset


                end_i = (
                    current_offset
                    +
                    meta["n_frames"]
                )


                v_scores = (
                    sim_scores_np[
                        start_i:end_i
                    ]
                )


                current_offset = end_i



            else:


                start_i = (
                    meta["start_idx"]
                )


                end_i = (
                    meta["end_idx"]
                )


                v_scores = (
                    sim_scores_np[
                        start_i:end_i
                    ]
                )



            if len(v_scores) == 0:

                continue



            # ---------------------------------
            # Temporal smoothing
            # ---------------------------------

            if len(v_scores) >= 5:


                kernel = np.array(
                    [
                        0.1,
                        0.2,
                        0.4,
                        0.2,
                        0.1
                    ],
                    dtype=np.float32
                )


                kernel = (
                    kernel /
                    kernel.sum()
                )


                smooth_scores = np.convolve(
                    v_scores,
                    kernel,
                    mode="same"
                )


                best_idx = int(
                    np.argmax(
                        smooth_scores
                    )
                )


                max_score = float(
                    0.6 *
                    smooth_scores[best_idx]
                    +
                    0.4 *
                    v_scores[best_idx]
                )


            else:


                best_idx = int(
                    np.argmax(
                        v_scores
                    )
                )


                max_score = float(
                    v_scores[best_idx]
                )



            results.append(

                {

                    "video_id":
                        video_id,


                    "max_score":
                        max_score,


                    "best_frame_idx":
                        best_idx,


                    "all_scores":
                        v_scores

                }

            )



        # -----------------------------------------
        # Ranking
        # -----------------------------------------

        results.sort(
            key=lambda x:
                x["max_score"],
            reverse=True
        )



        self.last_dense_dict = {

            item["video_id"]:
                item

            for item in results

        }



        if top_k_videos is None:

            return results



        return results[
            :top_k_videos
        ]

        # ======================================================
    # NORMALIZE QUERY VIEWS
    # ======================================================

    def normalize_query_views(
        self,
        query_input
    ):

        """
        Convert old prompt format
        or new semantic_views format
        into unified format.

        Output:

        [
          {
            text:"",
            weight:0.5
          }
        ]
        """

        views = []


        # --------------------------------
        # New Dynamic Semantic Views
        # --------------------------------

        if isinstance(
            query_input,
            list
        ):

            if (
                len(query_input) > 0
                and
                isinstance(
                    query_input[0],
                    dict
                )
            ):

                for item in query_input:

                    text = item.get(
                        "text",
                        ""
                    )


                    weight = item.get(
                        "importance",
                        1.0
                    )


                    if text:

                        views.append(
                            {
                                "text":
                                    text,

                                "weight":
                                    float(weight)
                            }
                        )


                return views



        # --------------------------------
        # Backward compatibility
        # --------------------------------

        if isinstance(
            query_input,
            str
        ):

            query_input = [
                query_input
            ]


        if isinstance(
            query_input,
            list
        ):


            n = len(
                query_input
            )


            if n == 0:

                return []



            default_weight = (
                1.0 / n
            )


            for text in query_input:

                if text:

                    views.append(
                        {
                            "text":
                                text,

                            "weight":
                                default_weight
                        }
                    )



        return views


        # ======================================================
    # DYNAMIC SEMANTIC FUSION
    # ======================================================

    def encode_dynamic_query(self, semantic_views):
        """
        Fuse multiple semantic views.

        Formula:
        Q = sum(importance_i * embedding_i)
        """
        if not semantic_views:
            return None

        texts = [v["text"] for v in semantic_views]

        weights = torch.tensor(
            [v["weight"] for v in semantic_views],
            dtype=torch.float32,
            device=self.device,
        )

        # Normalize o float32 de cong tong on dinh (khong mat precision).
        weights = weights / weights.sum()

        embeddings = self.encode_text_matrix(texts)

        # QUAN TRONG: tren cuda, embeddings la float16 (half) trong khi
        # weights dang float32. Phep nhan half * float32 se bi PyTorch
        # tu dong "type promote" len float32, lam fused_embedding thanh
        # float32 du dang chay GPU -> vo dtype dong nhat voi global_tensor
        # (half) -> RuntimeError o torch.matmul() trong search().
        # Phai ep weights ve dung dtype cua embeddings TRUOC khi nhan.
        weights = weights.to(dtype=embeddings.dtype)

        weights = weights.unsqueeze(1)

        fused_embedding = torch.sum(embeddings * weights, dim=0, keepdim=True)

        fused_embedding = fused_embedding / fused_embedding.norm(
            p=2, dim=-1, keepdim=True
        )

        return fused_embedding

    # ======================================================
    # WEIGHTED SIMILARITY (chuan hoa shape sau matmul)
    # ======================================================

    def weighted_similarity(self, sim_matrix):
        """
        Chuyen sim_matrix (N_frames, n_cols) thanh vector diem 1 chieu
        (N_frames,) de dung cho argmax/convolve o buoc sau.

        encode_dynamic_query() da fuse toan bo semantic view thanh MOT
        embedding duy nhat (weighted sum theo importance), nen q_matrix
        luon co dung 1 cot -> sim_matrix luon co shape (N_frames, 1).
        Ham nay chi can squeeze chieu do.

        Giu nhanh fallback (mean theo cot) cho truong hop hiem gap q_matrix
        co nhieu cot (vi du neu sau nay co caller truyen nhieu query vector
        chua fuse) de khong crash, khong phai vi day la hanh vi "weighted"
        moi - trong pipeline hien tai nhanh nay khong duoc dung toi.
        """
        if sim_matrix.dim() == 1:
            return sim_matrix
        if sim_matrix.shape[-1] == 1:
            return sim_matrix.squeeze(-1)
        return sim_matrix.mean(dim=-1)