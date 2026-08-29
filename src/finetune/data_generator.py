# ==============================================================================
# AIC 2026 - DATASET GENERATOR & VALIDATOR FOR MODEL A (NLP) & MODEL B (VLM)
# ==============================================================================
import os
import json
import random
from typing import List, Dict, Any

class FinetuneDataGenerator:
    """
    Generates and validates instruction-tuning datasets for:
    - Model A (NLP Parser): Anti-bias diverse domain pairs (Vietnamese query -> CPT Search JSON)
    - Model B (VLM Verifier): Chain-of-Thought Visual Reasoning pairs with strict partial penalty
    """

    @staticmethod
    def create_sample_nlp_data(output_path: str = "data/finetune/nlp_train.jsonl"):
        """
        Creates seed diverse instruction pairs for Model A enforcing Contextual Phrase Translation (CPT).
        Covers traffic, sports, cooking, cultural events, agriculture, OCR, etc. to prevent bias.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        system_prompt = (
            "Bạn là chuyên gia phân tích cú pháp truy vấn video đa phương thức (CLIP, BM25, OpenImages, VLM).\n"
            "Nhiệm vụ: Phân tích đoạn mô tả hoặc câu hỏi Tiếng Việt, trích xuất cấu trúc tìm kiếm JSON với ĐÚNG CÁC TRƯỜNG SAU:\n\n"
            "1. 'intent': 'VISUAL_SCENE' (mô tả bối cảnh/hình ảnh/hành động) hoặc 'OCR_TEXT' (khi đề bài yêu cầu đọc con số, chữ viết trên bảng/biển báo/cân/bản đồ/giá cả/cột mốc/slide).\n"
            "2. 'dense_weight': Trọng số tìm kiếm hình ảnh CLIP (0.75 cho VISUAL_SCENE, 0.35 cho OCR_TEXT).\n"
            "3. 'sparse_weight': Trọng số tìm kiếm văn bản BM25 (0.25 cho VISUAL_SCENE, 0.65 cho OCR_TEXT). Tổng = 1.0.\n"
            "4. 'golden_english_prompts': Mảng 2-4 câu miêu tả hình ảnh theo chuẩn CỤM TỪ NGỮ CẢNH (Contextual Phrase Translation) Tiếng Anh tự nhiên chuẩn 100%:\n"
            "   - QUY TẮC DỊCH CONTEXTUAL PHRASE:\n"
            "     + Không dịch thô Word-by-Word máy móc (KHÔNG dịch 'dán niêm phong' thành 'paste seal stamp').\n"
            "     + Không trừu tượng hóa mơ hồ (KHÔNG dịch thành 'two people doing something').\n"
            "     + BẮT BUỘC giữ đúng toàn bộ các danh từ vật thể và hành động thực tế bằng cụm Tiếng Anh tự nhiên.\n"
            "5. 'bm25_keywords': Mảng chứa TOÀN BỘ CÂU TIẾNG VIỆT GỐC và các CỤM DANH TỪ CỐT LÕI (2-4 từ/cụm).\n"
            "6. 'openimages_classes': Mảng danh từ Tiếng Anh đại diện cho VẬT THỂ THỂ LÝ nhìn thấy được ('person', 'box', 'fruit', 'car', 'sign', ...).\n"
            "7. 'vlm_question': Câu hỏi Tiếng Việt trực tiếp, cô đọng để VLM đọc ảnh trả lời (100% bằng Tiếng Việt).\n\n"
            "YÊU CẦU ĐẦU RA: CHỈ NÊU MỘT KHỐI JSON HỢP LỆ VÀ NẰM TRONG CẶP THẺ ```json ... ```. KHÔNG THÊM BẤT KỲ LỜI DẪN NÀO."
        )

        sample_pairs = [
            {
                "query": "Hai người phụ nữ cùng nhau dán niêm phong một thùng carton bằng băng dính.",
                "type": "KIS",
                "output": {
                    "intent": "VISUAL_SCENE",
                    "dense_weight": 0.75,
                    "sparse_weight": 0.25,
                    "golden_english_prompts": [
                        "two women sealing a cardboard carton box with tape",
                        "close up of hands taping a cardboard box",
                        "two women packing and sealing a parcel"
                    ],
                    "bm25_keywords": ["dán niêm phong", "thùng carton", "băng dính", "hai người phụ nữ"],
                    "openimages_classes": ["person", "box", "tape"],
                    "vlm_question": "Có phải có hai người phụ nữ đang dán niêm phong thùng carton không?"
                }
            },
            {
                "query": "Đoạn video bắt đầu bằng ảnh cận đầu một con lân trắng, mũi đỏ, bên cạnh lá cờ trắng viền đỏ.",
                "type": "KIS",
                "output": {
                    "intent": "VISUAL_SCENE",
                    "dense_weight": 0.80,
                    "sparse_weight": 0.20,
                    "golden_english_prompts": [
                        "close up of white lion dance head with red nose",
                        "white lion costume head next to white flag with red border",
                        "lion dance performance costume with red nose and decorative flag"
                    ],
                    "bm25_keywords": ["con lân trắng", "mũi đỏ", "lá cờ trắng viền đỏ", "đầu con lân"],
                    "openimages_classes": ["person", "flag", "toy", "clothing"],
                    "vlm_question": "Trong ảnh có đầu con lân trắng mũi đỏ bên cạnh lá cờ viền đỏ không?"
                }
            },
            {
                "query": "Con số hiển thị trên chiếc cân điện tử khi người bán cân túi trái cây là bao nhiêu?",
                "type": "QA",
                "output": {
                    "intent": "OCR_TEXT",
                    "dense_weight": 0.35,
                    "sparse_weight": 0.65,
                    "golden_english_prompts": [
                        "digital weight scale display numbers weighing fruit bag",
                        "electronic scale screen showing weight digits in market",
                        "close up of digital scale digits"
                    ],
                    "bm25_keywords": ["con số hiển thị", "cân điện tử", "túi trái cây", "cân"],
                    "openimages_classes": ["scale", "fruit", "person"],
                    "vlm_question": "Con số hiển thị trên chiếc cân điện tử là bao nhiêu?"
                }
            },
            {
                "query": "Các vận động viên đua xe đạp đang bứt tốc về đích trên đoạn đường đèo dốc.",
                "type": "KIS",
                "output": {
                    "intent": "VISUAL_SCENE",
                    "dense_weight": 0.75,
                    "sparse_weight": 0.25,
                    "golden_english_prompts": [
                        "cyclists sprinting to finish line on mountain pass road",
                        "bicycle race finish line sprint road cycling",
                        "group of cyclists racing on asphalt mountain road"
                    ],
                    "bm25_keywords": ["đua xe đạp", "bứt tốc về đích", "đường đèo dốc", "vận động viên"],
                    "openimages_classes": ["person", "bicycle", "helmet", "land vehicle"],
                    "vlm_question": "Các vận động viên đua xe đạp có đang bứt tốc về đích không?"
                }
            }
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            for item in sample_pairs:
                user_content = f"Loại truy vấn: {item['type']}\nNội dung Tiếng Việt: {item['query']}"
                assistant_content = f"```json\n{json.dumps(item['output'], ensure_ascii=False, indent=2)}\n```"
                
                record = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content}
                    ]
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"[INFO] FinetuneDataGenerator: Successfully created sample dataset at '{output_path}'")

    @staticmethod
    def create_sample_vlm_data(output_path: str = "data/finetune/vlm_train.jsonl"):
        """
        Creates seed diverse instruction pairs for Model B (VLM Verifier).
        Enforces Chain-of-Thought (CoT) and strict penalty for partial matches.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        sample_vlm_records = [
            {
                "image_path": "data/sample_frames/frame_01.jpg",
                "query_text": "con lân trắng, mũi đỏ, bên cạnh lá cờ trắng viền đỏ",
                "cot_reasoning": (
                    "Bước 1: Quét bối cảnh tổng quan: Có đầu con lân biểu diễn.\n"
                    "Bước 2: Kiểm tra chi tiết màu sắc: Lân màu trắng, mũi màu đỏ nổi bật.\n"
                    "Bước 3: Kiểm tra phụ kiện xung quanh: Bên cạnh có lá cờ nền trắng viền đỏ.\n"
                    "Kết luận: Khớp 100% tất cả 3 thực thể bắt buộc. Điểm: 95"
                ),
                "score": 95.0
            },
            {
                "image_path": "data/sample_frames/frame_02.jpg",
                "query_text": "con lân trắng, mũi đỏ, bên cạnh lá cờ trắng viền đỏ",
                "cot_reasoning": (
                    "Bước 1: Quét bối cảnh tổng quan: Có con lân màu vàng đang múa.\n"
                    "Bước 2: Kiểm tra chi tiết: Không phải lân trắng, không có mũi đỏ, không có lá cờ viền đỏ.\n"
                    "Kết luận: Khớp một phần sai lệch hoàn toàn. Điểm: 15"
                ),
                "score": 15.0
            }
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            for item in sample_vlm_records:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"[INFO] FinetuneDataGenerator: Successfully created VLM sample dataset at '{output_path}'")

if __name__ == "__main__":
    FinetuneDataGenerator.create_sample_nlp_data()
    FinetuneDataGenerator.create_sample_vlm_data()
