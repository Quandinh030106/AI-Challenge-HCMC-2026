import re
import json
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM
)


class QueryProcessor:
    """
    Semantic Query Understanding Engine

    Pipeline:

    Vietnamese Query
            |
            v
    Semantic Parser (Qwen2.5 - lazy loading)
            |
            v
    Structured Visual Understanding

    + fallback NLLB translation

    Output sẽ giữ tương thích với pipeline cũ:
        - query_en
        - prompt_ensemble
        - intent_info
    """

    def __init__(self):

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # ==================================================
        # Translation model (giữ lại từ pipeline cũ)
        # ==================================================

        self.translation_model_name = (
            "facebook/nllb-200-distilled-600M"
        )

        self.tokenizer = None
        self.model = None
        self.eng_token_id = None

        self.translator_available = False


        # ==================================================
        # Semantic Query Engine
        # ==================================================

        self.semantic_enabled = True

        self.semantic_model_name = (
            "Qwen/Qwen2.5-7B-Instruct"
        )

        self.semantic_tokenizer = None
        self.semantic_model = None

        # tránh load Qwen ngay khi khởi động
        self.semantic_loaded = False


        # ==================================================
        # Knowledge map
        # ==================================================

        self.visual_knowledge_map = {}


        print(
            "QueryProcessor: Initializing..."
        )


        # Load translator ngay
        self.load_translation_model()



    # ======================================================
    # LOAD TRANSLATION MODEL
    # ======================================================

    def load_translation_model(self):

        print(
            f"QueryProcessor: Loading translation model "
            f"{self.translation_model_name}"
        )

        try:

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.translation_model_name,
                src_lang="vie_Latn"
            )

            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.translation_model_name
            ).to(self.device)


            self.model.eval()


            self.eng_token_id = (
                self.tokenizer
                .convert_tokens_to_ids("eng_Latn")
            )


            self.translator_available = True


            print(
                "QueryProcessor: NLLB translation loaded."
            )


        except Exception as e:

            print(
                "QueryProcessor: NLLB failed."
            )

            print(e)


            try:

                fallback_model = (
                    "Helsinki-NLP/opus-mt-vi-en"
                )


                self.translation_model_name = (
                    fallback_model
                )


                self.tokenizer = (
                    AutoTokenizer
                    .from_pretrained(
                        fallback_model
                    )
                )


                self.model = (
                    AutoModelForSeq2SeqLM
                    .from_pretrained(
                        fallback_model
                    )
                    .to(self.device)
                )


                self.model.eval()

                self.eng_token_id = None

                self.translator_available = True


                print(
                    "QueryProcessor: MarianMT fallback loaded."
                )


            except Exception as e2:

                print(
                    "QueryProcessor:"
                    " Translation unavailable."
                )

                print(e2)

                self.translator_available = False




    # ======================================================
    # LOAD SEMANTIC MODEL (LAZY)
    # ======================================================

    def load_semantic_model(self):

        if self.semantic_loaded:
            return


        if not self.semantic_enabled:
            return


        print(
            "QueryProcessor: Loading semantic model "
            f"{self.semantic_model_name}"
        )


        try:

            self.semantic_tokenizer = (
                AutoTokenizer
                .from_pretrained(
                    self.semantic_model_name
                )
            )


            self.semantic_model = (
                AutoModelForCausalLM
                .from_pretrained(
                    self.semantic_model_name,
                    torch_dtype=(
                        torch.float16
                        if self.device == "cuda"
                        else torch.float32
                    ),
                    device_map="auto"
                )
            )


            self.semantic_model.eval()


            self.semantic_loaded = True


            print(
                "QueryProcessor: Semantic model loaded."
            )


        except Exception as e:

            print(
                "QueryProcessor:"
                " Cannot load semantic model."
            )

            print(e)


            self.semantic_enabled = False




    # ======================================================
    # SEMANTIC QUERY PARSER
    # ======================================================

    def semantic_parse(self, query_vi):

        """
        Convert Vietnamese query into structured
        visual understanding.

        Output:

        {
            scene,
            objects,
            actions,
            attributes,
            relationships,
            temporal_order,
            environment,
            domain
        }
        """

        default_result = {

            "scene": "",

            "objects": [],

            "actions": [],

            "attributes": [],

            "relationships": [],

            "temporal_order": [],

            "environment": [],

            "domain": ""

        }


        if (
            not self.semantic_enabled
            or not query_vi.strip()
        ):
            return default_result



        self.load_semantic_model()


        if not self.semantic_loaded:
            return default_result



        prompt = f"""
Bạn là chuyên gia phân tích truy vấn video.

Hãy phân tích câu tiếng Việt sau thành JSON.

Chỉ trả về JSON hợp lệ.

Cấu trúc:

{{
"scene":"",
"objects":[],
"actions":[],
"attributes":[],
"relationships":[],
"temporal_order":[],
"environment":[],
"domain":""
}}

Query:

{query_vi}
"""


        try:

            inputs = (
                self.semantic_tokenizer(
                    prompt,
                    return_tensors="pt"
                )
                .to(self.semantic_model.device)
            )


            with torch.no_grad():

                outputs = (
                    self.semantic_model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.1,
                        do_sample=False
                    )
                )


            text = (
                self.semantic_tokenizer
                .decode(
                    outputs[0],
                    skip_special_tokens=True
                )
            )


            # Thay vi regex greedy \{.*\}, dung json.JSONDecoder.raw_decode
            # tu vi tri dau "{" dau tien de chi lay DUY NHAT object JSON
            # hop le dau tien, bo qua phan text/template bi echo them sau do.
            brace_index = text.find("{")

            if brace_index != -1:
                try:
                    decoder = json.JSONDecoder()
                    data, _ = decoder.raw_decode(text[brace_index:])
                except json.JSONDecodeError as parse_exc:
                    print("Semantic parser JSON decode failed:", parse_exc)
                    data = None

                if data is not None:
                    for key in default_result:
                        if key not in data:
                            data[key] = default_result[key]
                    return data


        except Exception as e:

            print(
                "Semantic parser failed:",
                e
            )


        return default_result




    # ======================================================
    # CLEAN TRANSLATION
    # ======================================================

    def clean_translated_text(self, text):

        if not text:
            return ""


        words = text.split()

        cleaned = []

        last = None

        repeat = 0


        for w in words:

            if w.lower() == last:

                repeat += 1

                if repeat < 2:
                    cleaned.append(w)

            else:

                repeat = 0

                last = w.lower()

                cleaned.append(w)


        return " ".join(cleaned)




    # ======================================================
    # PREPROCESS VI QUERY
    # ======================================================

    def preprocess_query_vi(self, text_vi):

        if not text_vi:
            return ""


        cleaned = text_vi


        replacements = [
            (
                r'\bmực\b',
                'squid'
            ),

            (
                r'\bđậu hà lan\b',
                'green peas'
            ),
            
            (
                r'\bmăng tây\b',
                'asparagus'
            ),

            (
                r'\bmúa lân\b',
                'lion dance performance'
            ),

            (
                r'\bcon lân\b',
                'lion dance performer'
            ),

            (
                r'\bbọ cánh cứng\b',
                'beetle insect'
            ),

            (
                r'\bđiêu khắc cát\b',
                'sand sculpture'
            ),

            (
                r'\bngười đàn ông\b',
                'man'
            ),

            (
                r'\bngười phụ nữ\b',
                'woman'
            )
        ]


        for pattern, repl in replacements:

            cleaned = re.sub(
                pattern,
                repl,
                cleaned,
                flags=re.IGNORECASE
            )


        return cleaned




    # ======================================================
    # TRANSLATION
    # ======================================================

    def translate_vi_to_en(self, text_vi):

        if not text_vi.strip():
            return ""


        text_vi_clean = (
            self.preprocess_query_vi(
                text_vi
            )
        )


        translated = ""


        if self.translator_available:

            try:

                inputs = (
                    self.tokenizer(
                        text_vi_clean,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=150
                    )
                    .to(self.device)
                )


                kwargs = {

                    "max_length":150,

                    "num_beams":2,

                    "no_repeat_ngram_size":3

                }


                if self.eng_token_id:

                    kwargs[
                        "forced_bos_token_id"
                    ] = self.eng_token_id



                with torch.no_grad():

                    tokens = (
                        self.model.generate(
                            **inputs,
                            **kwargs
                        )
                    )


                translated = (
                    self.tokenizer
                    .batch_decode(
                        tokens,
                        skip_special_tokens=True
                    )[0]
                )


                translated = (
                    self.clean_translated_text(
                        translated
                    )
                )


            except Exception as e:

                print(
                    "Translation error:",
                    e
                )



        if (
            not translated
            or len(translated.split()) < 2
        ):

            translated = text_vi_clean



        return translated
    

        # ======================================================
    # FALLBACK SEMANTIC PARSER
    # ======================================================

    def fallback_semantic_parse(self, query_vi):

        """
        Fallback khi Qwen semantic model không chạy.

        Không thay thế LLM.
        Chỉ giữ một số entity/action quan trọng.
        """

        result = {

            "scene": "",

            "objects": [],

            "actions": [],

            "attributes": [],

            "relationships": [],

            "temporal_order": [],

            "environment": [],

            "domain": ""

        }


        text = query_vi.lower()


        # ----------------------------
        # Domain detection
        # ----------------------------

        cooking_words = [

            "nấu",
            "xào",
            "chiên",
            "luộc",
            "chảo",
            "bếp",
            "món ăn",
            "nguyên liệu"

        ]


        if any(
            w in text
            for w in cooking_words
        ):

            result["domain"] = "cooking"

            result["scene"] = (
                "cooking scene"
            )

            result["environment"].extend(
                [
                    "kitchen",
                    "cooking area"
                ]
            )


        # ----------------------------
        # Common objects
        # ----------------------------

        object_map = {

            "mực":
                "squid",

            "đậu hà lan":
                "green peas",

            "hành tây":
                "onion",

            "ớt đỏ":
                "red chili",

            "chảo":
                "pan",

            "bếp":
                "stove",

            "xe đạp":
                "bicycle",

            "con lân":
                "lion dance performer",

            "nho":
                "grape",

            "kéo":
                "scissors"

        }


        for vn, en in object_map.items():

            if vn in text:

                result["objects"].append(
                    en
                )


        # ----------------------------
        # Action extraction
        # ----------------------------

        action_map = {

            "xào":
                "stir frying",

            "cho vào":
                "adding ingredients",

            "bỏ vào":
                "putting ingredients",

            "cắt":
                "cutting",

            "rót":
                "pouring",

            "đuổi theo":
                "chasing",

            "đi":
                "walking"

        }


        for vn, en in action_map.items():

            if vn in text:

                result["actions"].append(
                    en
                )


        return result




    # ======================================================
    # BUILD SEMANTIC PROMPTS
    # ======================================================

    def build_semantic_prompts(
        self,
        semantic_query
    ):

        """
        Convert structured semantic information
        into CLIP-friendly prompts.
        """


        objects = semantic_query.get(
            "objects",
            []
        )


        actions = semantic_query.get(
            "actions",
            []
        )


        scene = semantic_query.get(
            "scene",
            ""
        )


        environment = semantic_query.get(
            "environment",
            []
        )


        object_text = ", ".join(
            objects
        )


        action_text = ", ".join(
            actions
        )


        env_text = ", ".join(
            environment
        )


        prompts = []


        if object_text:

            prompts.append(
                (
                    f"{scene}, "
                    f"showing {object_text}"
                )
            )


        if action_text:

            prompts.append(
                (
                    f"a video scene of "
                    f"{action_text}"
                )
            )


        if env_text:

            prompts.append(
                (
                    f"{scene} in "
                    f"{env_text}"
                )
            )


        return prompts




    # ======================================================
    # QUERY INTENT DETECTION
    # ======================================================

    def detect_query_intent(self, text_vi):

        text_lower = text_vi.lower()


        ocr_keywords = [

            "chữ",

            "biển",

            "bảng",

            "logo",

            "số",

            "tên",

            "ghi là",

            "đọc",

            "poster",

            "banner",

            "slide",

            "tiêu đề"

        ]


        if any(
            k in text_lower
            for k in ocr_keywords
        ):

            return {

                "intent":
                    "OCR_TEXT",

                "dense_weight":
                    0.4,

                "sparse_weight":
                    0.6

            }



        return {

            "intent":
                "VISUAL_SCENE",

            "dense_weight":
                0.75,

            "sparse_weight":
                0.25

        }




    # ======================================================
    # PROMPT ENSEMBLE
    # ======================================================

    def generate_prompt_ensemble(
        self,
        query_en,
        query_vi="",
        semantic_query=None
    ):

        prompts = []


        # --------------------------------
        # 1. Semantic prompts
        # --------------------------------

        if semantic_query:

            prompts.extend(
                self.build_semantic_prompts(
                    semantic_query
                )
            )


        # --------------------------------
        # 2. Literal translation
        # --------------------------------

        clean = (
            query_en
            .strip()
            .rstrip(".")
        )


        if clean:

            prompts.extend(
                [

                    clean,

                    f"a photo of {clean}",

                    f"a video scene showing {clean}",

                    f"a close-up view of {clean}"

                ]
            )



        # --------------------------------
        # Remove duplicate
        # --------------------------------

        unique = []


        for p in prompts:

            if (
                p
                and p not in unique
                and len(p) > 3
            ):

                unique.append(p)



        return unique


        # ======================================================
    # BUILD DYNAMIC SEMANTIC VIEWS
    # ======================================================

    def build_dynamic_semantic_views(
        self,
        query_en,
        semantic_query
    ):

        """
        Convert structured semantic query
        into dynamic retrieval views.

        Không cố định:
            object
            action
            scene

        View nào có dữ liệu mới sinh.

        Output:

        [
          {
            type,
            text,
            importance
          }
        ]
        """


        views = []


        if semantic_query is None:

            semantic_query = {}



        objects = semantic_query.get(
            "objects",
            []
        )


        actions = semantic_query.get(
            "actions",
            []
        )


        attributes = semantic_query.get(
            "attributes",
            []
        )


        environment = semantic_query.get(
            "environment",
            []
        )


        scene = semantic_query.get(
            "scene",
            ""
        )


        relationships = semantic_query.get(
            "relationships",
            []
        )



        # ---------------------------------
        # Object view
        # ---------------------------------

        if objects:

            views.append(

                {
                    "type":
                        "object",

                    "text":
                        ", ".join(
                            objects
                        ),

                    "importance":
                        0.35

                }

            )



        # ---------------------------------
        # Action view
        # ---------------------------------

        if actions:

            views.append(

                {
                    "type":
                        "action",

                    "text":
                        ", ".join(
                            actions
                        ),

                    "importance":
                        0.30

                }

            )



        # ---------------------------------
        # Scene + environment
        # ---------------------------------

        scene_text = []


        if scene:

            scene_text.append(
                scene
            )


        if environment:

            scene_text.extend(
                environment
            )



        if scene_text:

            views.append(

                {
                    "type":
                        "scene",

                    "text":
                        ", ".join(
                            scene_text
                        ),

                    "importance":
                        0.20

                }

            )



        # ---------------------------------
        # Attribute
        # ---------------------------------

        if attributes:

            views.append(

                {
                    "type":
                        "attribute",

                    "text":
                        ", ".join(
                            attributes
                        ),

                    "importance":
                        0.10

                }

            )



        # ---------------------------------
        # Relationship
        # ---------------------------------

        if relationships:

            views.append(

                {
                    "type":
                        "relationship",

                    "text":
                        ", ".join(
                            relationships
                        ),

                    "importance":
                        0.10

                }

            )



        # ---------------------------------
        # Literal query
        # ---------------------------------

        if query_en:

            views.append(

                {
                    "type":
                        "literal",

                    "text":
                        query_en,

                    "importance":
                        0.10

                }

            )



        # ---------------------------------
        # Normalize weights
        # ---------------------------------

        total = sum(
            v["importance"]
            for v in views
        )


        if total > 0:

            for v in views:

                v["importance"] = (
                    v["importance"]
                    /
                    total
                )



        return views

    # ======================================================
    # MAIN PROCESS
    # ======================================================

    def process(self, query_vi):

        """
        Main API.

        Backward compatible.
        """


        # 1. Translation fallback

        query_en = (
            self.translate_vi_to_en(
                query_vi
            )
        )


        # 2. Semantic parsing

        semantic_query = (
            self.semantic_parse(
                query_vi
            )
        )


        # nếu Qwen fail

        if not any(
            semantic_query.values()
        ):

            semantic_query = (
                self.fallback_semantic_parse(
                    query_vi
                )
            )



        # 3. Generate prompts

        semantic_views = (
            self.build_dynamic_semantic_views(
                query_en,
                semantic_query
            )
        )


        prompt_ensemble = [

            view["text"]

            for view in semantic_views

        ]


        # 4. Intent

        intent_info = (
            self.detect_query_intent(
                query_vi
            )
        )



        # 5. Visual description

        visual_description = " ".join(
            prompt_ensemble[:3]
        )


        return {

            "query_vi":
                query_vi,


            "query_en":
                query_en,


            "english_query":
                query_en,


            "semantic_query":
                semantic_query,

            "semantic_views":
                semantic_views,

            "literal_query":
                query_en,


            "visual_description":
                visual_description,


            "prompt_ensemble":
                prompt_ensemble,


            "intent_info":
                intent_info

        }