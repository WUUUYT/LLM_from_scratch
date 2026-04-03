# cs336_data/quality_classifier.py
"""
CS336 Assignment 4 - 2.7 Quality Classifier
使用训练好的fastText模型对文本质量打分
"""

import os
import warnings

import fasttext

_quality_model = None


def _get_quality_model(model_path: str) -> fasttext.FastText._FastText:
    global _quality_model
    if _quality_model is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _quality_model = fasttext.load_model(model_path)
    return _quality_model


def _get_model_path() -> str:
    return os.environ.get("QUALITY_MODEL_PATH", "data/classifiers/quality_classifier.bin")


# cs336_data/quality_classifier.py
def classify_quality(
    text: str,
    model_path: str | None = None,
    threshold: float = 0.5,
) -> tuple[str, float]:
    if model_path is None:
        model_path = _get_model_path()

    model = _get_quality_model(model_path)

    cleaned = text.replace("\n", " ").strip()
    if not cleaned:
        return ("cc", 1.0)

    labels, scores = model.predict(cleaned, k=2)

    label_score = dict(zip([l.replace("__label__", "") for l in labels], [float(s) for s in scores]))

    wiki_score = label_score.get("wiki", 0.0)

    if wiki_score >= threshold:
        return ("wiki", wiki_score)
    else:
        return ("cc", label_score.get("cc", 1.0 - wiki_score))
