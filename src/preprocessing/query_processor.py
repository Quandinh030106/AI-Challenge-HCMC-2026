import gc
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

    def __init__(self, config=None):

        config = config or {}
        semantic_config = config.get("semantic_query", {})

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

        self.semantic_enabled = bool(
            semantic_config.get("enabled", False)
        )

        self.semantic_model_name = semantic_config.get(
            "model", {}
        ).get(
            "name",
            "Qwen/Qwen2.5-7B-Instruct",
        )

        self.semantic_quantization_4bit = bool(
            semantic_config.get("quantization_4bit", False)
        )

        self.semantic_max_new_tokens = int(
            semantic_config.get("model", {}).get(
                "max_new_tokens",
                250,
            )
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

            # Dual-GPU: cuda:1 danh rieng cho VLM (Qwen2-VL) trong
            # VisualReRanker/solve_task2. Neu dung device_map="auto",
            # accelerate co the tran layer cua model 7B nay sang cuda:1
            # khi cuda:0 khong du cho, lam VLM OOM ngay tu lenh goi dau
            # tien (da xac nhan qua log: cuda:1 da 14.35/14.56 GiB TRUOC
            # khi VisualReRanker xu ly bat ky anh nao). Ep model nay CHI
            # dung cuda:0 de bao ve VRAM danh cho VLM.
            device_map = (
                {"": "cuda:0"}
                if torch.cuda.device_count() >= 2
                else "auto"
            )

            quantization_config = None
            if self.semantic_quantization_4bit and self.device == "cuda":
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
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
                    device_map=device_map,
                    quantization_config=quantization_config,
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

            if self.semantic_model is not None:
                del self.semantic_model
                self.semantic_model = None

            if self.semantic_tokenizer is not None:
                del self.semantic_tokenizer
                self.semantic_tokenizer = None

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            self.semantic_enabled = False
            self.semantic_loaded = False


    # ======================================================
    # PARTIAL JSON RECOVERY (Prompt 10)
    # ======================================================

    @staticmethod
    def _partial_recover_semantic_json(text, default_result):
        """
        Khi json.JSONDecoder.raw_decode() that bai (thuong do Qwen output
        bi CAT CUT vi vuot max_new_tokens - da xac nhan qua log thuc te:
        "Unterminated string starting at: line 1 column 1133"), hanh vi cu
        vut bo TOAN BO du lieu da sinh ra, ke ca cac truong da hoan chinh
        truoc diem cat.

        Ham nay co gang cuu lai TUNG TRUONG da hoan chinh (dung format
        JSON that su cho truong do: co du dau ngoac dong/mo), bo qua
        truong bi cat cut giua chung (KHONG doan bua du lieu). Vi thu tu
        key trong prompt la scene -> objects -> actions -> attributes ->
        relationships -> temporal_order -> environment -> domain, cac
        truong quan trong nhat cho CLIP retrieval (scene/objects/actions)
        thuong nam o DAU nen co xac suat con nguyen ven cao nhat.

        Tra ve dict day du (default cho truong khong cuu duoc) neu cuu
        duoc IT NHAT 1 truong, hoac None neu khong cuu duoc gi (giu
        nguyen hanh vi cu: roll ve fallback_semantic_parse).
        """
        recovered = dict(default_result)
        found_any = False

        list_keys = (
            "objects", "actions", "attributes",
            "relationships", "temporal_order", "environment",
        )
        for key in list_keys:
            match = re.search(
                r'"%s"\s*:\s*\[(.*?)\]' % re.escape(key),
                text,
                flags=re.DOTALL,
            )
            if not match:
                continue
            items = re.findall(r'"((?:[^"\\]|\\.)*)"', match.group(1))
            if items:
                recovered[key] = items
                found_any = True

        for key in ("scene", "domain"):
            match = re.search(
                r'"%s"\s*:\s*"((?:[^"\\]|\\.)*)"' % re.escape(key),
                text,
            )
            if match:
                recovered[key] = match.group(1)
                found_any = True

        return recovered if found_any else None


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
Hãy phân tích câu tiếng Việt sau và trả về JSON MÔ TẢ ĐÚNG NỘI DUNG câu đó.
QUAN TRỌNG: mọi giá trị PHẢI dịch sang TIẾNG ANH và PHẢN ÁNH ĐÚNG nội dung câu, KHÔNG để trống nếu câu có thông tin liên quan.

Ví dụ định dạng (chỉ minh họa, không phải nội dung cần trả lời):
Query: "Một người đàn ông đang chiên trứng trong chảo trên bếp gas."
{{"scene": "cooking scene in a kitchen", "objects": ["man", "pan", "egg", "gas stove"], "actions": ["frying"], "attributes": [], "relationships": [], "temporal_order": [], "environment": ["kitchen"], "domain": "cooking"}}

Việc chỉ tìm các từ riêng lẻ có thể trả về rất nhiều kết quả sai; hệ thống cần hiểu đồng thời object, action và context.
Yêu cầu: liệt kê ĐẦY ĐỦ các danh từ, động từ, tính từ, mối quan hệ, thứ tự thời gian, môi trường và bối cảnh CÓ THẬT trong câu, không bỏ sót chi tiết nào và không bịa thêm chi tiết không có trong câu.
Mỗi mục trong danh sách viết đầy đủ, tránh lặp lại ý đã có ở mục khác, chú ý để toàn bộ JSON sinh ra không bị vượt quá giới hạn độ dài cho phép.

Bây giờ hãy phân tích câu sau, CHỈ trả về JSON hợp lệ (đúng cú pháp, đóng đủ dấu ngoặc), không thêm giải thích:

Query:

{query_vi}
"""


        try:
            # Prompt 11: don GPU truoc generate(), giong pattern da dung o
            # visual_reranker.py::verify_single_image() va
            # task2_vqa.py::solve_single_video_vqa(). semantic_parse() la ham
            # generate() DUY NHAT trong code truoc day KHONG lam viec nay.
            # Voi prompt hien tai (khong con gioi han do dai muc liet ke),
            # Qwen thuong sinh gan sat max_new_tokens MOI LAN goi, va ham nay
            # co the bi goi NHIEU LAN cho 1 cau hoi (moi semantic event trong
            # sequence_search.py goi lai process() -> semantic_parse() rieng),
            # nen rui ro tich luy phan manh VRAM qua nhieu lan generate() lien
            # tiep la co that.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
                        max_new_tokens=self.semantic_max_new_tokens,
                        temperature=0.1,
                        do_sample=False
                    )
                )

            # CHI lay phan token MOI duoc sinh ra, khong lay lai prompt dau vao.
            # Neu decode ca outputs[0] (gom ca prompt), text.find("{") se bat nham
            # dau "{" cua vi du cau truc RONG trong chinh prompt, khong bao gio
            # doc toi JSON that su model sinh ra o cuoi chuoi.
            generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]

            text = (
                self.semantic_tokenizer
                .decode(
                    generated_ids,
                    skip_special_tokens=True
                )
            )

            print("[DEBUG] Qwen raw output:", text[:500])


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
                    # Prompt 10: KHONG vut bo toan bo output khi JSON bi cat
                    # cut (thuong do vuot max_new_tokens voi cau phuc tap -
                    # da xac nhan qua log query-p1-23-kis). Cac truong o DAU
                    # JSON (scene/objects/actions) van thuong con nguyen ven
                    # va co gia tri retrieval cao hon nhieu so voi roll het
                    # ve fallback_semantic_parse (chi vai tu khoa hardcode).
                    data = self._partial_recover_semantic_json(
                        text[brace_index:], default_result,
                    )
                    if data is not None:
                        print(
                            "Semantic parser: da cuu duoc mot phan JSON "
                            "bi cat cut (partial recovery)."
                        )

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

        finally:
            # Luon giai phong CUDA cache sau generate(), bat ke thanh cong
            # hay that bai, vi ham nay bi goi lap lai nhieu lan trong cung
            # 1 query (sequence-aware KIS goi rieng cho tung event).
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
                r'\bcon cá\b',
                'fish'
            ),

            (
                r'\b(?<!cân bằng)(?<!cân nặng)(?<!cân đo)(?<!cân nhắc)(?<!cân đối)cân\b',
                'scale'
            ),

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

    @staticmethod
    def build_dynamic_semantic_views(
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

        Prompt 10: THEM view "context" ghep object+action+attribute+scene
        thanh MOT cau tu nhien (thay vi cac cum tu roi rac), vi cac view
        rieng le duoi day duoc CLIP encode DOC LAP roi cong trong so lai
        (weighted sum trong encode_dynamic_query), nen mat het lien ket
        "ai lam gi voi gi" (vi du "nguoi cam keo mau den" bi tach thanh
        hai vector rieng "person" va "black scissors" khong con gan voi
        nhau). View "context" bo sung mot cau gan voi phong cach caption
        CLIP duoc huan luyen, giup giu duoc quan he giua cac thanh phan.
        Cac view cu KHONG bi xoa, chi them view moi -> an toan rollback
        bang cach chinh importance ve 0.

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
        # Context view (Prompt 10 - relational)
        # ---------------------------------

        context_parts = []

        if actions and objects:
            context_parts.append(
                "%s %s" % (", ".join(actions), ", ".join(objects))
            )
        elif objects:
            context_parts.append(", ".join(objects))
        elif actions:
            context_parts.append(", ".join(actions))

        if attributes:
            context_parts.append(", ".join(attributes))

        if scene_text:
            context_parts.append(", ".join(scene_text))

        if context_parts:
            views.append(
                {
                    "type": "context",
                    "text": ". ".join(context_parts),
                    "importance": 0.25,
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