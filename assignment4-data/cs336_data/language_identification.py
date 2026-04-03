# cs336_data/language_identification.py

import fasttext

# 全局缓存模型，避免重复加载（模型较大）
_model = None
LANG_REMAP = {
    "zho": "zh",
    "cmn": "zh",
    # 其他需要映射的语言可以在这里添加
}


def _get_model(model_path: str = "/data/classifiers/lid.176.bin") -> fasttext.FastText._FastText:
    global _model
    if _model is None:
        # 加载时抑制fasttext的警告输出
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _model = fasttext.load_model(model_path)
    return _model


def identify_language(text: str, model_path: str = "data/classifiers/lid.176.bin") -> tuple[str, float]:
    """
    识别文本的主要语言。

    Args:
        text: 输入的Unicode字符串
        model_path: fastText模型路径
    Returns:
        (language_code, confidence_score) 元组
        language_code: 语言代码字符串，如 "en", "zh"
        confidence_score: 0到1之间的置信度分数
    """
    model = _get_model(model_path)

    # fasttext不能处理换行符，需要替换
    cleaned = text.replace("\n", " ").strip()

    # 空文本处理
    if not cleaned:
        return ("unknown", 0.0)

    # 预测：返回top-1结果
    # labels格式: ['__label__en'], scores格式: [0.98]
    labels, scores = model.predict(cleaned, k=1)

    # 去掉 "__label__" 前缀，得到语言代码
    lang = labels[0].replace("__label__", "")
    score = float(scores[0])

    return (lang, score)
