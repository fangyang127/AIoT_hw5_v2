import io
import re
from pathlib import Path
from typing import Dict, Iterable

import langdetect
import streamlit as st

try:
    import docx  # type: ignore
except Exception:
    docx = None


PREVIEW_CHAR_LIMIT = 600


def clean_text(text: str, max_chars: int = 2000) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text[:max_chars]


def chunk_text(text: str, max_chars: int = 1200) -> Iterable[str]:
    text = clean_text(text)
    for i in range(0, len(text), max_chars):
        yield text[i : i + max_chars]


def detect_language_safe(text: str) -> str:
    try:
        return langdetect.detect(text)
    except Exception:
        return "unknown"


def load_text_from_file(uploaded_file) -> str:
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    if file_size_mb > 2:
        st.error("檔案過大 (>2MB)")
        return ""

    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".txt":
        return uploaded_file.getvalue().decode("utf-8", errors="ignore")
    if suffix == ".docx":
        if not docx:
            st.error("缺少 python-docx，無法解析 .docx")
            return ""
        document = docx.Document(io.BytesIO(uploaded_file.getvalue()))
        return "\n".join([p.text for p in document.paragraphs])
    st.error("不支援的檔案格式")
    return ""


def summarize_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + " ..."


sample_texts: Dict[str, str] = {
    "AI-生成範例": "In this article, we will comprehensively explore the key factors affecting climate dynamics, providing a structured overview with bullet points and conclusions.",
    "人類撰寫範例": "今天下午去公園散步，看到小朋友在放風箏，風有點大但陽光很好，回家路上還買了杯手搖飲。",
}
