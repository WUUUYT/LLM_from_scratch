# cs336_data/train_quality_classifier.py
"""
训练fastText质量分类器
"""

import random
from pathlib import Path

import fasttext


def train_quality_classifier(
    train_file: str,
    model_output: str = "data/classifiers/quality_classifier.bin",
    valid_ratio: float = 0.1,
    # fastText超参数
    lr: float = 0.1,
    epoch: int = 20,
    word_ngrams: int = 2,  # bigram，捕捉词组特征
    dim: int = 100,  # 词向量维度
    min_count: int = 1,
) -> dict:
    """
    训练fastText二分类器区分高质量/低质量文本。

    Args:
        train_file: fastText格式的训练数据文件
        model_output: 模型保存路径
        valid_ratio: 验证集比例
        lr: 学习率
        epoch: 训练轮数
        word_ngrams: n-gram大小（2=bigram，能捕捉词组）
        dim: 词向量维度
    Returns:
        包含验证集精度等指标的字典
    """
    # ── 划分训练/验证集 ──────────────────────────────────
    with open(train_file, encoding="utf-8") as f:
        lines = f.readlines()

    random.shuffle(lines)
    n_valid = int(len(lines) * valid_ratio)
    valid_lines = lines[:n_valid]
    train_lines = lines[n_valid:]

    train_split = train_file.replace(".txt", "_train_split.txt")
    valid_split = train_file.replace(".txt", "_valid_split.txt")

    with open(train_split, "w", encoding="utf-8") as f:
        f.writelines(train_lines)
    with open(valid_split, "w", encoding="utf-8") as f:
        f.writelines(valid_lines)

    print(f"训练集: {len(train_lines)} 条")
    print(f"验证集: {len(valid_lines)} 条")

    # ── 训练 ──────────────────────────────────────────────
    print("\n开始训练 fastText 分类器...")
    print(f"  lr={lr}, epoch={epoch}, word_ngrams={word_ngrams}, dim={dim}")

    model = fasttext.train_supervised(
        input=train_split,
        lr=lr,
        epoch=epoch,
        wordNgrams=word_ngrams,
        dim=dim,
        minCount=min_count,
        loss="softmax",  # 二分类用softmax
        verbose=2,
    )

    # ── 评估 ──────────────────────────────────────────────
    n, precision, recall = model.test(valid_split)
    print("\n验证集结果:")
    print(f"  样本数: {n}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {2 * precision * recall / (precision + recall):.4f}")

    # ── 保存模型 ──────────────────────────────────────────
    Path(model_output).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_output)
    print(f"\n模型已保存到 {model_output}")

    return {
        "precision": precision,
        "recall": recall,
        "n_train": len(train_lines),
        "n_valid": n,
    }


if __name__ == "__main__":
    train_quality_classifier(
        train_file="data/quality_classifier_train.txt",
        model_output="data/classifiers/quality_classifier.bin",
    )
