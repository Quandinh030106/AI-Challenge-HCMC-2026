import streamlit as st
import os
import numpy as np
import pandas as pd
from PIL import Image

from src.utils import load_config, get_keyframe_path_by_index

from src.search.dense_search import DenseSearcher
from src.search.sparse_search import SparseSearcher
from src.search.fusion import reciprocal_rank_fusion
from src.search.sequence_search import rerank_sequence_aware_kis
from src.preprocessing.query_processor import QueryProcessor
from src.tasks.task1_kis import solve_task1, get_frame_id_from_idx
from src.tasks.task2_vqa import solve_task2
from src.tasks.task3_trake import solve_task3

st.set_page_config(
    page_title="AI Challenge HCMC 2026 - Video Search System",
    page_icon="🎬",
    layout="wide"
)

@st.cache_resource(show_spinner="Dang khoi tao he thong tim kiem...")
def init_engine():
    config = load_config("configs/default.yaml")
    dense = DenseSearcher(config)
    sparse = SparseSearcher(config)
    qp = QueryProcessor(config)
    return config, dense, sparse, qp

config, dense_searcher, sparse_searcher, query_processor = init_engine()
keyframes_dir = config["data"]["keyframes_dir"]
metadata_dir = config["data"].get("metadata_dir")
map_keyframes_dir = (
    config["data"].get("map_keyframes_dir")
    or metadata_dir
)

st.title("🎬 AI Challenge HCMC 2026 - He Thong Truy Xuat Video")
st.markdown("*He thong tim kiem video da phuong thuc sieu toc danh cho Vong So Tuyen AIC 2026*")

# Sidebar
st.sidebar.header("Cai Dat & Chuc Nang")
task_mode = st.sidebar.radio(
    "Chon Dạng Bai Thi:",
    ["Task 1: Textual KIS (Tim kiem su kien)", "Task 2: Visual Q&A (Hoi - Dap)", "Task 3: TRAKE (Chuoi su kien)"]
)
top_k = st.sidebar.slider("So luong ket qua hien thi (Top K):", 1, 20, 5)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Device:** `{dense_searcher.device}`")
st.sidebar.markdown(f"**So video da index:** `{len(dense_searcher.all_video_ids)} videos`")

_FRAME_MAP_CACHE = {}

def get_keyframe_index_from_frame_id(video_id, frame_id):
    """
    Reverse mapping:
        actual video frame_id
        -> keyframe/vector ordinal (0-based)

    Chỉ trả về ordinal khi frame_id tồn tại chính xác trong Map-Keyframes.
    Không đoán nearest frame.
    """
    try:
        actual_frame_id = int(frame_id)
    except (TypeError, ValueError):
        return None

    if video_id not in _FRAME_MAP_CACHE:
        if not map_keyframes_dir:
            return None

        csv_path = os.path.join(
            map_keyframes_dir,
            f"{video_id}.csv",
        )

        if not os.path.isfile(csv_path):
            return None

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return None

        normalized_columns = {
            str(col).strip().lower(): col
            for col in df.columns
        }

        if "frame_idx" not in normalized_columns:
            return None

        values = pd.to_numeric(
            df[normalized_columns["frame_idx"]],
            errors="coerce",
        )

        if values.isna().any():
            return None

        _FRAME_MAP_CACHE[video_id] = values.to_numpy(
            dtype=np.int64
        )

    frame_values = _FRAME_MAP_CACHE[video_id]

    matches = np.flatnonzero(
        frame_values == actual_frame_id
    )

    if matches.size == 0:
        return None

    return int(matches[0])


def find_keyframe_image(video_id, frame_id):
    """
    actual video frame_id
        -> Map-Keyframes
        -> keyframe ordinal
        -> file anh vat ly
    """
    keyframe_idx = get_keyframe_index_from_frame_id(
        video_id,
        frame_id,
    )

    if keyframe_idx is None:
        return None

    try:
        return get_keyframe_path_by_index(
            keyframes_dir,
            video_id,
            keyframe_idx,
        )
    except (IndexError, FileNotFoundError, ValueError):
        return None


# --- TASK 1: TEXTUAL KIS ---
if "Task 1" in task_mode:
    st.subheader("📌 Task 1: Tim Kiem Chinh Xac Theo Van Ban (Textual KIS)")
    query_input = st.text_input("Nhap cau mo ta su kien can tim:", "mot dien gia dang phat bieu truoc may quay")
    
    if st.button("🔍 Tim Kiem Video", key="btn_task1"):
        with st.spinner("Dang tim kiem..."):
            q_info = query_processor.process(query_input)
            intent = q_info["intent_info"]
            
            dense_res = dense_searcher.search(q_info["semantic_views"], top_k_videos=top_k*2)
            sparse_res = sparse_searcher.search(query_input, top_k_videos=top_k*2)
            fused = reciprocal_rank_fusion(
                dense_res, sparse_res, 
                dense_weight=intent["dense_weight"], 
                sparse_weight=intent["sparse_weight"]
            )
            fused, sequence_trace = rerank_sequence_aware_kis(
                query_text=query_input,
                fused_candidates=fused,
                dense_searcher=dense_searcher,
                sparse_searcher=sparse_searcher,
                query_processor=query_processor,
                config=config,
                pre_object_candidates=fused,
                query_id="streamlit_task1",
            )
            
            st.success(f"Da dich sang English: **{q_info['query_en']}** | Y dinh: `{intent['intent']}`")
            if sequence_trace.get("applied"):
                st.info(
                    "Sequence-aware: %d semantic events"
                    % len(sequence_trace.get("events", []))
                )
            
            cols = st.columns(min(top_k, 5))
            for idx, cand in enumerate(fused[:top_k]):
                vid = cand["video_id"]
                dense_info = cand.get("dense_info")
                
                best_frame_idx = dense_info["best_frame_idx"] if dense_info else 0

                frame_id = get_frame_id_from_idx(
                    keyframes_dir,
                    vid,
                    best_frame_idx,
                    metadata_dir=map_keyframes_dir,
                )

                img_path = get_keyframe_path_by_index(
                    keyframes_dir,
                    vid,
                    best_frame_idx,
                )
                
                col = cols[idx % 5]
                with col:
                    st.markdown(f"### Top {idx+1}")
                    st.markdown(f"**Video ID:** `{vid}`")
                    st.markdown(f"**Frame ID:** `{frame_id}`")
                    
                    if img_path and os.path.exists(img_path):
                        img = Image.open(img_path)
                        st.image(img, use_container_width=True)
                    else:
                        st.info("Chua co anh keyframe vat ly")

# --- TASK 2: VISUAL Q&A ---
elif "Task 2" in task_mode:
    st.subheader("❓ Task 2: Truy Van Hoi - Dap Truc Quan (Visual Q&A)")
    query_input = st.text_input("Mo ta boi canh / su kien:", "dien gia phat bieu tai cuoc hop bao")
    question_input = st.text_input("Cau hoi can tra loi:", "Nguoi dien gia mac ao mau gi?")
    
    if st.button("🔍 Tim Kiem & Tra Loi", key="btn_task2"):
        with st.spinner("Dang tim kiem va goi VLM Qwen2-VL..."):
            q_info = query_processor.process(query_input)
            intent = q_info["intent_info"]
            
            dense_res = dense_searcher.search(q_info["semantic_views"], top_k_videos=10)
            sparse_res = sparse_searcher.search(query_input, top_k_videos=10)
            fused = reciprocal_rank_fusion(dense_res, sparse_res)
            
            ans_res = solve_task2(
                query_input,
                question_input,
                fused,
                keyframes_dir,
                model_id=config["models"]["vlm_model"],
                metadata_dir=map_keyframes_dir,
                ocr_dir=config["data"].get("metadata_dir"),
                qa_config=config.get("search", {}).get("qa_evidence", {}),
                query_processor=query_processor,
                query_id="streamlit_task2",
            )
            
            st.success(f"**Dap An Tu VLM:** `{ans_res['answer']}`")
            st.markdown(f"- **Video:** `{ans_res['video_id']}` | **Frame:** `{ans_res['frame_id']}`")
            st.markdown(
                "- **Evidence:** `%s` | **Score:** `%s` | **Confidence:** `%s`"
                % (
                    ans_res.get("evidence_source"),
                    ans_res.get("evidence_score"),
                    ans_res.get("confidence"),
                )
            )
            
            img_path = find_keyframe_image(ans_res['video_id'], ans_res['frame_id'])
            if img_path and os.path.exists(img_path):
                st.image(Image.open(img_path), caption=f"Keyframe {ans_res['frame_id']}", width=400)

# --- TASK 3: TRAKE ---
elif "Task 3" in task_mode:
    st.subheader("⏱️ Task 3: Can Chinh Chuoi Su Kien (TRAKE)")
    query_input = st.text_input("Mo ta tong quat chuoi su kien:", "Hoi thao va phat bieu")
    
    events_str = st.text_area(
        "Danh sach cac su kien con (moi dong 1 su kien):",
        "Dien gia buoc len san khau\nDien gia phat bieu truoc micro\nKhan gia vo tay"
    )
    events_list = [e.strip() for e in events_str.split("\n") if e.strip()]
    
    if st.button("🔍 Can Chinh Chuoi Su Kien", key="btn_task3"):
        with st.spinner("Dang can chinh thoi gian bang Quy hoach dong (DP)..."):
            q_info = query_processor.process(query_input)
            dense_res = dense_searcher.search(q_info["semantic_views"], top_k_videos=10)
            sparse_res = sparse_searcher.search(query_input, top_k_videos=10)
            fused = reciprocal_rank_fusion(dense_res, sparse_res)
            
            align_res = solve_task3(
                events_list, fused, keyframes_dir, dense_searcher,
                metadata_dir=map_keyframes_dir, query_processor=query_processor,
                config=config,
            )
            vid = align_res["video_id"]
            frame_ids = align_res["frame_ids"]
            
            st.success(f"Video khop nhat: **{vid}**")
            st.markdown("### Dòng Thoi Gian Su Kien (Timeline):")
            
            cols = st.columns(len(events_list))
            for i, ev_name in enumerate(events_list):
                fid = frame_ids[i] if i < len(frame_ids) else None

                with cols[i]:
                    st.markdown(f"**Event {i+1}:** {ev_name}")

                    if fid is None:
                        st.warning("Không có frame hợp lệ để hiển thị.")
                        continue

                    st.markdown(f"**Frame:** `{fid}`")
                    img_path = find_keyframe_image(vid, fid)
                    if img_path and os.path.exists(img_path):
                        st.image(Image.open(img_path), use_container_width=True)
