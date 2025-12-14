import time
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from models import get_available_hf_models, load_baseline_model, load_hf_model, predict_proba
from utils import (
    PREVIEW_CHAR_LIMIT,
    chunk_text,
    clean_text,
    detect_language_safe,
    load_text_from_file,
    sample_texts,
    summarize_text,
)


st.set_page_config(
    page_title="AI vs Human Detector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def cached_baseline():
    return load_baseline_model()


@st.cache_resource(show_spinner=False)
def cached_hf(model_name: str):
    return load_hf_model(model_name)


def run_inference(text: str, model_name: str, use_hf: bool):
    chunks = chunk_text(text, max_chars=1200)
    start = time.time()
    if use_hf:
        model = cached_hf(model_name)
    else:
        model = cached_baseline()
    probs = [predict_proba(model, chunk, use_hf=use_hf) for chunk in chunks]
    avg_ai = sum(p["ai"] for p in probs) / len(probs)
    avg_human = sum(p["human"] for p in probs) / len(probs)
    elapsed_ms = int((time.time() - start) * 1000)
    return {"ai": avg_ai, "human": avg_human, "elapsed_ms": elapsed_ms, "chunks": len(chunks)}


def render_gauges(prob_ai: float, prob_human: float):
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.barh(["AI", "Human"], [prob_ai * 100, prob_human * 100], color=["#e4572e", "#4b9cd3"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)")
    for i, v in enumerate([prob_ai, prob_human]):
        ax.text(v * 100 + 1, i, f"{v*100:.1f}%", va="center")
    st.pyplot(fig, use_container_width=True)

    fig2, ax2 = plt.subplots(figsize=(3, 3), subplot_kw=dict(aspect="equal"))
    values = [prob_ai, prob_human]
    labels = ["AI", "Human"]
    colors = ["#e4572e", "#4b9cd3"]
    wedges, _ = ax2.pie(values, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
    centre_circle = plt.Circle((0, 0), 0.55, fc="white")
    fig2.gca().add_artist(centre_circle)
    st.pyplot(fig2, use_container_width=True)


def main():
    st.title("AI vs Human 文章偵測器")
    st.caption("TF-IDF + Logistic Regression baseline，並可切換 Hugging Face transformers 模型。")

    st.sidebar.header("輸入")
    input_mode = st.sidebar.radio("選擇輸入方式", ["文字輸入", "檔案上傳", "範例測試"])

    raw_text = ""
    uploaded_preview = ""
    if input_mode == "文字輸入":
        raw_text = st.text_area("輸入文本（支援中文/英文）", height=220, placeholder="貼上想檢測的文章...")
    elif input_mode == "檔案上傳":
        file = st.file_uploader("上傳 .txt 或 .docx", type=["txt", "docx"])
        if file:
            raw_text = load_text_from_file(file)
            uploaded_preview = summarize_text(raw_text, PREVIEW_CHAR_LIMIT)
    else:
        sample_choice = st.sidebar.selectbox("選擇範例", list(sample_texts.keys()))
        raw_text = sample_texts[sample_choice]
        st.info(f"已載入範例：{sample_choice}")

    st.sidebar.header("模型選擇")
    model_source = st.sidebar.radio("模型類型", ["Baseline (TF-IDF + LR)", "Transformers"])
    use_hf = model_source.startswith("Transformers")
    hf_models = get_available_hf_models()
    hf_model_name = st.sidebar.selectbox("Transformers 模型 (首次載入需等待下載)", hf_models, index=0, disabled=not use_hf)
    max_len = st.sidebar.slider("長文截斷 (chars)", min_value=500, max_value=4000, value=2000, step=100)
    if use_hf:
        st.sidebar.info("Transformers 首次載入需下載模型，請耐心等候；若環境受限請改用 Baseline。")

    st.sidebar.markdown("---")
    st.sidebar.caption("檔案大小限制 2 MB；長文會截斷/分段平均。")

    if uploaded_preview:
        st.sidebar.subheader("檔案預覽")
        st.sidebar.write(uploaded_preview)

    if st.button("立即偵測", type="primary") or (input_mode == "範例測試" and raw_text):
        if not raw_text:
            st.warning("請先輸入或上傳文本")
            return
        lang = detect_language_safe(raw_text)
        if lang not in {"zh", "en"}:
            st.error("目前僅支援中文/英文，請提供相應文本。")
            return
        cleaned = clean_text(raw_text, max_chars=max_len)
        with st.spinner("模型推論中..."):
            result = run_inference(cleaned, hf_model_name, use_hf=use_hf)

        st.success("完成！")
        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.metric("AI 機率", f"{result['ai']*100:.2f}%")
            st.metric("Human 機率", f"{result['human']*100:.2f}%")
            st.caption(f"模型：{'Transformers' if use_hf else 'Baseline'} | {hf_model_name if use_hf else 'TF-IDF + LR'}")
            st.caption(f"耗時：{result['elapsed_ms']} ms | 分段數：{result['chunks']}")
        with col2:
            render_gauges(result["ai"], result["human"])

        st.subheader("詳細資訊")
        st.write(
            f"**語言**：{lang} ｜ **字數**：{len(cleaned)} ｜ **模型**：{'HF/' + hf_model_name if use_hf else 'TF-IDF + LR'}"
        )
        st.text_area("輸入文本 (清理後)", cleaned, height=200)

        if use_hf:
            st.info("Transformers 輸出為每段的 logits/機率平均。若模型標籤與 AI/Human 不符，已嘗試映射。")


if __name__ == "__main__":
    main()
