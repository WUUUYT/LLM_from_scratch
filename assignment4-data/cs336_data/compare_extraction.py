# compare_extraction.py
from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.extract_text import extract_text_from_html_bytes

# 读取WARC，提取第一个网页
with open("CC-MAIN-20250417135010-20250417165010-00065.warc.gz", "rb") as f:
    for record in ArchiveIterator(f, record_types=WarcRecordType.response):
        content_type = record.http_headers.get("Content-Type", "")
        if "text/html" in content_type:
            html_bytes = record.reader.read()
            my_text = extract_text_from_html_bytes(html_bytes)
            print("=== 我的提取结果 ===")
            print(my_text[:1000])
            break

# 读取对应WET文件的第一条
with open("CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz", "rb") as f:
    for record in ArchiveIterator(f, record_types=WarcRecordType.conversion):
        wet_text = record.reader.read().decode("utf-8")
        print("\n=== WET官方提取结果 ===")
        print(wet_text[:1000])
        break
