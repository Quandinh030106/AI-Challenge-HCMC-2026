import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import glob
import re
from src.preprocessing.query_processor import QueryProcessor
from src.search.object_search import ObjectSearcher
from src.utils import load_config

# Danh sach cac cau hoi mau dac trung tu bo de thi AIC 2026 de test truc tiep
SAMPLE_QUERIES = [
    {
        "id": "query-p1-01 (Tàu vũ trụ)",
        "text": "Đoạn video về một công ty tư nhân phóng tàu vũ trụ chở 4 phi hành gia trong trang phục màu đen, ngoài không gian có thể nhìn thấy cực quang rực rỡ."
    },
    {
        "id": "query-p1-02 (Đàn hổ)",
        "text": "Đoạn video quay cảnh một đàn hổ, trong đó có một con hổ con nhảy chồm lên tảng đá."
    },
    {
        "id": "query-p1-03 (Vệ sinh máy ảnh)",
        "text": "Tìm phân cảnh một người dùng tăm bông để vệ sinh ống kính máy ảnh, xung quanh có đặt một chiếc khăn màu tím."
    },
    {
        "id": "query-p1-05 (Thu hoạch dứa)",
        "text": "Cảnh một người phụ nữ lớn tuổi đang chèo ghe chở đầy dứa vừa thu hoạch trên sông nước miền Tây."
    },
    {
        "id": "query-p1-06 (Gỏi cuốn hoa pansy)",
        "text": "Tìm đoạn video hướng dẫn làm món gỏi cuốn chay với bánh tráng màu vàng, màu tím và có trang trí cánh hoa pansy."
    },
    {
        "id": "query-p1-07 (Cho dê ăn)",
        "text": "Cảnh hai người phụ nữ với nụ cười rạng rỡ đang cầm cỏ cho đàn dê ăn trong một chuồng dê bằng gỗ."
    },
    {
        "id": "query-p1-08 (Điêu khắc cát)",
        "text": "Lễ hội triển lãm các bức tượng điêu khắc bằng cát khổng lồ, bên cạnh có các bạn trẻ đang trượt ván patin."
    },
    {
        "id": "query-p1-09 (Robot bọ Lausanne)",
        "text": "Các nhà khoa học tại Đại học Lausanne giới thiệu robot bọ cánh cứng bay mô phỏng cơ học cánh côn trùng."
    },
    {
        "id": "query-p1-10 (Cá mập Steven Spielberg)",
        "text": "Đoạn phóng sự giới thiệu về bộ phim về loài cá mập trắng khổng lồ năm 1975 của đạo diễn Steven Spielberg."
    },
    {
        "id": "query-p1-11 (Bánh rán dâu chuối)",
        "text": "Trang trí những chiếc bánh rán chiên vàng với sốt socola, dâu tây cắt lát và những lát chuối chín."
    },
    {
        "id": "query-p1-12 (Panna cotta)",
        "text": "Ly tráng miệng panna cotta mát lạnh được trang trí vài quả nho xanh và một nhánh lá bạc hà."
    },
    {
        "id": "query-p1-13 (Đua xe đạp flycam)",
        "text": "Đoàn vận động viên đua xe đạp đang bứt tốc trên đường nhựa, góc quay từ trên cao flycam bao quát đoàn đua."
    },
    {
        "id": "query-p1-14 (QA: CLB FANA Khánh Hòa)",
        "text": "Chương trình thiện nguyện của câu lạc bộ FANA hỗ trợ các trẻ em có hoàn cảnh khó khăn tại một xã ở tỉnh Khánh Hòa.\nCâu hỏi: Tên của xã đó là gì?"
    },
    {
        "id": "query-p1-15 (QA: Đền thờ Nguyễn Trung Trực)",
        "text": "Lễ hội tưởng niệm vị anh hùng dân tộc Nguyễn Trung Trực tại một ngôi đền ở Kiên Giang.\nCâu hỏi: Hai câu thơ được khắc trên đền thờ là gì?"
    }
]

def run_keyword_test(input_dir=None, config_path="configs/default.yaml"):
    """Chay kiem tra bieu dien tu khoa, thuc the va cau dich tren cac cau hoi."""
    config = load_config(config_path) if os.path.exists(config_path) else {"data": {}, "models": {}}
    
    print("================================================================")
    print("KHOI CHAY SCRIPT KIEM TRA TRICH XUAT TU KHOA & NGU NGHIA (TEST)")
    print("================================================================")
    
    query_processor = QueryProcessor()
    object_searcher = ObjectSearcher(config)
    
    queries_to_test = []
    if input_dir and os.path.exists(input_dir):
        txt_files = sorted(glob.glob(os.path.join(input_dir, "*.txt")) + glob.glob(os.path.join(input_dir, "**", "*.txt"), recursive=True))
        for f in txt_files:
            with open(f, "r", encoding="utf-8") as fp:
                content = fp.read().strip()
                queries_to_test.append({"id": os.path.basename(f), "text": content})
    else:
        queries_to_test = SAMPLE_QUERIES
        
    print(f"\nTong so cau hoi kiem tra: {len(queries_to_test)}\n")
    
    for idx, item in enumerate(queries_to_test, 1):
        q_id = item["id"]
        q_text = item["text"]
        
        print("----------------------------------------------------------------")
        print(f"[{idx}/{len(queries_to_test)}] ID: {q_id}")
        print(f"📝 Câu hỏi Tiếng Việt  : {q_text}")
        
        # 1. Xu ly dich & Prompt Ensemble
        proc_res = query_processor.process(q_text)
        trans_en = proc_res.get("query_en", "")
        intent_info = proc_res.get("intent_info", {})
        prompts = proc_res.get("prompt_ensemble", [])
        
        print(f"🌐 Dịch Tiếng Anh (CLIP): {trans_en}")
        print(f"🎯 Phân loại Intent     : {intent_info.get('intent')} (Dense weight: {intent_info.get('dense_weight')}, Sparse weight: {intent_info.get('sparse_weight')})")
        
        # 2. Trich xuat thuc the & Named Entities
        named_entities = re.findall(r'\b[A-ZĐÁÀẢÃẠÂẤẦẨẪẬĂẮẰẲẴẶÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ][a-zđáàảãạâấầẩẫậăắằẳẵặéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ]+\b', q_text)
        rare_keywords = [kw for kw in ["fana", "khánh hòa", "nguyễn trung trực", "kiên giang", "lausanne", "spielberg", "covid", "panna cotta", "múa lân", "gỏi cuốn", "dê", "xe đạp", "bánh rán", "cá mập", "tàu vũ trụ", "ống kính", "máy ảnh"] if kw in q_text.lower()]
        
        print(f"🏷️  Thực thể viết hoa    : {named_entities if named_entities else 'Không có'}")
        print(f"🔑 Từ khóa đặc biệt     : {rare_keywords if rare_keywords else 'Không có'}")
        
        # 3. Anh xa sang lop vat the Object Detection
        mapped_objects = object_searcher.extract_target_entities(q_text)
        print(f"📦 Objects tương ứng    : {mapped_objects if mapped_objects else 'Không có'}")
        
        # 4. Cac bien the Prompt Ensemble gui cho CLIP
        print(f"🚀 Prompt Ensemble ({len(prompts)} biến thể):")
        for p_idx, p in enumerate(prompts[:4], 1):
            print(f"    {p_idx}. \"{p}\"")
        if len(prompts) > 4:
            print(f"    ... và {len(prompts) - 4} biến thể khác")
            
    print("\n================================================================")
    print("✅ HOAN TAT KIEM TRA! Ban co the xem xet ket qua o tren de danh gia do chinh xac.")
    print("================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=None, help="Thu muc chua cac file .txt neu muon test truc tiep de thi")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    
    run_keyword_test(args.input_dir, args.config)
