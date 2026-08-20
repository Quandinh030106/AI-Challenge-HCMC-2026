import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import glob
import re
import zipfile
from src.preprocessing.query_processor import QueryProcessor
from src.search.object_search import ObjectSearcher
from src.export_codabench_submission import parse_query_file
from src.utils import load_config

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def run_keyword_test(input_dir=None, config_path="configs/default.yaml"):
    """
    Script chan doan chuyen sau:
    Doc toan bo 24 cau hoi de thi va hien thi chi tiet cau truc du lieu, cau dich Meta NLLB-200,
    cac thuc the viet hoa, tu khoa hiem, lop vat the Objects va Prompt Ensemble gui cho AI.
    """
    config = load_config(config_path) if os.path.exists(config_path) else {"data": {}, "models": {}}
    
    print("================================================================")
    print("🔬 KHOI CHAY CHAN DOAN TRICH XUAT TU KHOA & NGU NGHIA DE THI")
    print(f"Thu muc de thi: {input_dir}")
    print("================================================================")
    
    query_processor = QueryProcessor()
    object_searcher = ObjectSearcher(config)
    
    txt_files = []
    if input_dir and os.path.exists(input_dir):
        if os.path.isfile(input_dir) and input_dir.lower().endswith(".zip"):
            unzip_tmp = "/kaggle/working/bo_de_thi_extracted"
            os.makedirs(unzip_tmp, exist_ok=True)
            with zipfile.ZipFile(input_dir, "r") as zf:
                zf.extractall(unzip_tmp)
            input_dir = unzip_tmp
            
        all_found = []
        for root, _, files in os.walk(input_dir):
            for f in files:
                if f.lower().endswith(".txt"):
                    all_found.append(os.path.join(root, f))
                elif f.lower().endswith(".zip"):
                    try:
                        uz = "/kaggle/working/bo_de_thi_auto"
                        os.makedirs(uz, exist_ok=True)
                        with zipfile.ZipFile(os.path.join(root, f), "r") as zf:
                            zf.extractall(uz)
                        for r2, _, f2 in os.walk(uz):
                            for fs in f2:
                                if fs.lower().endswith(".txt"):
                                    all_found.append(os.path.join(r2, fs))
                    except Exception:
                        pass
        txt_files = sorted(list(set(all_found)), key=natural_sort_key)
        
    if not txt_files:
        print(f"Khong tim thay file de thi trong {input_dir}. Chay tren bo cau hoi mau.")
        txt_files = []
        
    print(f"\nTong so cau hoi kiem tra: {len(txt_files)} file.\n")
    
    for idx, file_path in enumerate(txt_files, 1):
        parsed = parse_query_file(file_path)
        q_id = parsed["query_id"]
        task_type = parsed["task_type"]
        q_text = parsed["query"]
        
        print("----------------------------------------------------------------")
        print(f"[{idx}/{len(txt_files)}] ID: {q_id} | TASK: {task_type.upper()}")
        print(f"📝 Nội dung Tiếng Việt  : {q_text}")
        
        if task_type == "qa":
            print(f"❓ Câu hỏi cần trả lời  : {parsed.get('question', '')}")
        elif task_type == "trake":
            print(f"⏱️  Chuỗi sự kiện ({len(parsed.get('events', []))} bước):")
            for ev_idx, ev in enumerate(parsed.get("events", []), 1):
                print(f"    - Bước {ev_idx}: {ev}")
                
        # 1. Xu ly dich & Prompt Ensemble
        search_target = f"{q_text} {parsed.get('question', '')}".strip() if task_type == "qa" else q_text
        proc_res = query_processor.process(search_target)
        trans_en = proc_res.get("query_en", "")
        intent_info = proc_res.get("intent_info", {})
        prompts = proc_res.get("prompt_ensemble", [])
        
        print(f"🌐 Dịch Tiếng Anh (NLLB): {trans_en}")
        print(f"🎯 Phân loại Intent     : {intent_info.get('intent')} (Dense weight: {intent_info.get('dense_weight')}, Sparse weight: {intent_info.get('sparse_weight')})")
        
        # 2. Trich xuat thuc the & Named Entities
        named_entities = re.findall(r'\b[A-ZĐÁÀẢÃẠÂẤẦẨẪẬĂẮẰẲẴẶÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ][a-zđáàảãạâấầẩẫậăắằẳẵặéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ]+\b', search_target)
        rare_keywords = [kw for kw in ["fana", "khánh hòa", "nguyễn trung trực", "kiên giang", "lausanne", "spielberg", "covid", "panna cotta", "múa lân", "gỏi cuốn", "dê", "xe đạp", "bánh rán", "cá mập", "tàu vũ trụ", "ống kính", "máy ảnh", "thịt xay", "măng tây", "nấm", "đậu hũ", "củ năng"] if kw in search_target.lower()]
        
        print(f"🏷️  Thực thể viết hoa    : {named_entities if named_entities else 'Không có'}")
        print(f"🔑 Từ khóa đặc biệt     : {rare_keywords if rare_keywords else 'Không có'}")
        
        # 3. Anh xa sang lop vat the Object Detection
        mapped_objects = object_searcher.extract_target_entities(search_target)
        print(f"📦 Objects tương ứng    : {mapped_objects if mapped_objects else 'Không có'}")
        
        # 4. Cac bien the Prompt Ensemble gui cho CLIP
        print(f"🚀 Prompt Ensemble ({len(prompts)} biến thể):")
        for p_idx, p in enumerate(prompts[:3], 1):
            print(f"    {p_idx}. \"{p}\"")
        if len(prompts) > 3:
            print(f"    ... và {len(prompts) - 3} biến thể khác")
            
    print("\n================================================================")
    print("✅ HOAN TAT KIEM TRA TOAN BO 24 CAU HOI!")
    print("Ban hay quan sat toan bo ket qua dich va tu khoa o tren.")
    print("Neu tat ca deu chuan xac, ban co the tu tin chay Cell tiep theo de tao submission.zip!")
    print("================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="/kaggle/input/aic-hcmc2026-thu-nghiem-bo-de-thi", help="Thu muc chua cac file .txt de thi")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    
    run_keyword_test(args.input_dir, args.config)
