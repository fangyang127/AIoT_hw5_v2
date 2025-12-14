import logging
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


logger = logging.getLogger(__name__)


def _seed_corpus() -> Tuple[List[str], List[int]]:
    # 0 -> human, 1 -> AI
    ai_samples = [
        "In this comprehensive analysis, we will explore the multifaceted dynamics of the topic with structured sections.",
        "The model was trained on a diverse corpus and demonstrates competitive performance across benchmarks.",
        "As an AI language model, I do not possess consciousness or subjective experiences.",
        "Overall, the findings suggest promising directions for future work in the domain.",
    ]
    human_samples = [
        "今天早上買早餐的時候，下起一場大雨，我只好躲在騎樓等。",
        "這篇文章是我花了一整晚寫的，內容可能不夠完美，但是真實的心情。",
        "My friend told me a funny story on the way home, and I couldn't stop laughing.",
        "手寫的字跡有時候潦草，但裡面有很多情感，是機器難以模仿的。",
    ]
    texts = human_samples + ai_samples
    labels = [0] * len(human_samples) + [1] * len(ai_samples)
    return texts, labels


def load_baseline_model():
    texts, labels = _seed_corpus()
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=5000,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(texts)
    clf = LogisticRegression(max_iter=200)
    clf.fit(X, labels)
    logger.info("Baseline TF-IDF + LR model trained on seed corpus.")
    return {"vectorizer": vectorizer, "clf": clf}


def load_hf_model(model_name: str):
    from transformers import pipeline

    logger.info("Loading HF model: %s", model_name)
    # Force CPU to avoid GPU/driver issues on hosted envs; enable low memory usage when possible.
    return pipeline(
        "text-classification",
        model=model_name,
        device_map="cpu",
        model_kwargs={"low_cpu_mem_usage": True},
    )


def map_hf_labels(label: str) -> Dict[str, float]:
    label_lower = label.lower()
    if "fake" in label_lower or "ai" in label_lower or "generated" in label_lower:
        return {"ai": 1.0, "human": 0.0}
    if "real" in label_lower or "human" in label_lower:
        return {"ai": 0.0, "human": 1.0}
    if "positive" in label_lower:
        return {"ai": 0.7, "human": 0.3}
    if "negative" in label_lower:
        return {"ai": 0.3, "human": 0.7}
    return {"ai": 0.5, "human": 0.5}


def predict_proba(model, text: str, use_hf: bool = False) -> Dict[str, float]:
    if use_hf:
        outputs = model(text, truncation=True, max_length=512)
        result = outputs[0] if isinstance(outputs, list) else outputs
        mapped = map_hf_labels(result["label"])
        score = float(result.get("score", 0.5))
        ai_prob = mapped["ai"] * score + (1 - mapped["human"]) * (1 - score)
        human_prob = 1.0 - ai_prob
        return {"ai": ai_prob, "human": human_prob}

    vectorizer = model["vectorizer"]
    clf = model["clf"]
    X = vectorizer.transform([text])
    proba = clf.predict_proba(X)[0]
    # proba ordering is [human(0), ai(1)]
    return {"ai": float(proba[1]), "human": float(proba[0])}


def get_available_hf_models() -> List[str]:
    # Keep list short; user can edit README to add more.
    return [
        "distilbert-base-uncased-finetuned-sst-2-english",  # small, downloads faster
        "distilroberta-base",
        "roberta-base-openai-detector",
    ]
