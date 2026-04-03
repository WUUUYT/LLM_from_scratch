# cs336_data/prepare_quality_training_data.py
"""
准备质量分类器的训练数据
正例：Wikipedia引用页面
负例：Common Crawl随机页面
"""

import gzip
import random
from pathlib import Path

from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.extract_text import extract_text_from_html_bytes
from cs336_data.gopher_quality_filter import passes_gopher_quality_filter
from cs336_data.language_identification import LANG_REMAP, identify_language


def extract_texts_from_warc(
    warc_path: str,
    max_docs: int = 10000,
    apply_filters: bool = True,
) -> list[str]:
    texts = []

    with open(warc_path, "rb") as f:
        for record in ArchiveIterator(f, record_types=WarcRecordType.response):
            if len(texts) >= max_docs:
                break

            content_type = record.http_headers.get("Content-Type", "")
            if "text/html" not in content_type:
                continue

            # ── 新增：检查HTTP状态码，跳过重定向/错误页 ──
            status = record.http_headers.get("status_code", "200")
            try:
                if int(status) != 200:
                    continue
            except (ValueError, TypeError):
                pass

            try:
                html_bytes = record.reader.read()
                text = extract_text_from_html_bytes(html_bytes)
            except Exception:
                continue

            if not text.strip():
                continue

            # ── 新增：最少要有100个词才算有效文本 ──
            if len(text.split()) < 100:
                continue

            if apply_filters:
                lang, score = identify_language(text)
                lang = LANG_REMAP.get(lang, lang)
                if lang != "en" or score < 0.5:
                    continue

                if not passes_gopher_quality_filter(text):
                    continue

            texts.append(text)

    return texts


# prepare_quality_training_data.py
# 修改 prepare_training_data 函数


def prepare_training_data(
    positive_warc: str,
    negative_warc: str,
    output_path: str = "data/quality_classifier_train.txt",
    max_per_class: int = 5000,
) -> None:
    print("提取正例（Wikipedia引用页面）...")
    positive_texts = extract_texts_from_warc(
        positive_warc,
        max_docs=max_per_class,
        apply_filters=True,
    )
    print(f"  获得 {len(positive_texts)} 条正例")

    print("提取负例（CC随机页面）...")
    # 负例：不做Gopher过滤，保留更多低质量特征
    negative_texts = extract_texts_from_warc(
        negative_warc,
        max_docs=max_per_class,
        apply_filters=False,  # ← 负例不过滤，保留低质量特征
    )
    # 但负例要求至少有50个词（排除完全空的页面）
    negative_texts = [t for t in negative_texts if len(t.split()) >= 50]
    print(f"  获得 {len(negative_texts)} 条负例")

    # 平衡正负例
    n = min(len(positive_texts), len(negative_texts))
    if n < 100:
        print(f"⚠️  正例只有 {len(positive_texts)} 条，数据太少！")
        print("   建议爬取更多URL后重试")
        return

    positive_texts = random.sample(positive_texts, n)
    negative_texts = random.sample(negative_texts, n)

    print(f"写入训练文件（每类 {n} 条）...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for text in positive_texts:
            clean = text.replace("\n", " ").strip()
            if clean:
                f.write(f"__label__wiki {clean}\n")
        for text in negative_texts:
            clean = text.replace("\n", " ").strip()
            if clean:
                f.write(f"__label__cc {clean}\n")

    print(f"✓ 训练数据已保存：{output_path}")
    print(f"  正例: {n} 条  负例: {n} 条  共: {n * 2} 条")


def subsample_wiki_urls(
    url_file: str,
    output_file: str,
    n: int = 1000,
    seed: int = 42,
) -> None:
    """
    从43.5M Wikipedia URL中随机采样n条。
    """
    random.seed(seed)
    print(f"读取URL文件: {url_file}")

    with gzip.open(url_file, "rt", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"总URL数: {len(urls):,}，采样 {n} 条")
    sampled = random.sample(urls, min(n, len(urls)))

    with open(output_file, "w") as f:
        for url in sampled:
            f.write(url + "\n")

    print(f"采样URL已保存到 {output_file}")
