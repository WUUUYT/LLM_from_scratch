# run_langid_analysis.py
from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.extract_text import extract_text_from_html_bytes
from cs336_data.language_identification import identify_language

results = []

with open("CC-MAIN-20250417135010-20250417165010-00065.warc.gz", "rb") as f:
    for i, record in enumerate(ArchiveIterator(f, record_types=WarcRecordType.response)):
        if i >= 20:
            break
        content_type = record.http_headers.get("Content-Type", "")
        if "text/html" not in content_type:
            continue

        url = str(record.headers.get("WARC-Target-URI", ""))
        html_bytes = record.reader.read()
        text = extract_text_from_html_bytes(html_bytes)

        if text.strip():
            lang, score = identify_language(text)
            print(f"[{i + 1:2d}] {lang} ({score:.2f}) | {url[:60]}")
            print(f"     预览: {text[:80].strip()}")
            print()
            results.append((lang, score))

# 统计
en_count = sum(1 for l, s in results if l == "en")
print(f"\n英文比例: {en_count}/{len(results)} = {en_count / len(results) * 100:.0f}%")
