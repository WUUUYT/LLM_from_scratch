# cs336_data/harmful_content.py

import re
import warnings

import fasttext

# ── 模型缓存 ──────────────────────────────────────────
_nsfw_model = None
_toxic_model = None


def _load_model(path: str) -> fasttext.FastText._FastText:
    """加载fastText模型，抑制警告"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fasttext.load_model(path)


def _get_nsfw_model(model_path: str) -> fasttext.FastText._FastText:
    global _nsfw_model
    if _nsfw_model is None:
        _nsfw_model = _load_model(model_path)
    return _nsfw_model


def _get_toxic_model(model_path: str) -> fasttext.FastText._FastText:
    global _toxic_model
    if _toxic_model is None:
        _toxic_model = _load_model(model_path)
    return _toxic_model


# ── 工具函数 ──────────────────────────────────────────
def _preprocess(text: str) -> str:
    """
    fastText预处理：
    1. 换行符替换为空格（否则只处理第一行）
    2. 压缩多余空格
    """
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _predict(model, text: str) -> tuple[str, float]:
    """通用预测函数，返回(标签, 置信度)"""
    cleaned = _preprocess(text)
    if not cleaned:
        return ("non-toxic", 0.0)

    labels, scores = model.predict(cleaned, k=1)
    # 去掉 __label__ 前缀
    label = labels[0].replace("__label__", "")
    score = float(scores[0])
    return (label, score)


# ── NSFW 分类器 ────────────────────────────────────────
def classify_nsfw(
    text: str, model_path: str = "data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin"
) -> tuple[str, float]:
    """
    检测文本是否包含NSFW内容。

    Returns:
        ("nsfw", score)     如果是NSFW内容
        ("non-nsfw", score) 如果是正常内容
    """
    model = _get_nsfw_model(model_path)
    return _predict(model, text)


# ── 毒性语言分类器 ─────────────────────────────────────
def classify_toxic_speech(
    text: str, model_path: str = "data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin"
) -> tuple[str, float]:
    """
    检测文本是否包含毒性言论。

    Returns:
        ("toxic", score)     如果是毒性内容
        ("non-toxic", score) 如果是正常内容
    """
    model = _get_toxic_model(model_path)
    return _predict(model, text)
