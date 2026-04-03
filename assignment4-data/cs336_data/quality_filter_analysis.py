# 加到 analysis.py 里，或单独运行
from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.extract_text import extract_text_from_html_bytes
from cs336_data.gopher_quality_filter import passes_gopher_quality_filter


def analyze_gopher_filter(n_total: int = 100):
    """
    遍历WARC文件，统计过滤情况，
    并打印20个典型案例（通过/拒绝各10个）
    """
    passed, rejected = [], []

    with open("CC-MAIN-20250417135010-20250417165010-00065.warc.gz", "rb") as f:
        for record in ArchiveIterator(f, record_types=WarcRecordType.response):
            if len(passed) + len(rejected) >= n_total:
                break
            content_type = record.http_headers.get("Content-Type", "")
            if "text/html" not in content_type:
                continue

            url = str(record.headers.get("WARC-Target-URI", ""))
            html_bytes = record.reader.read()
            text = extract_text_from_html_bytes(html_bytes)

            if not text.strip():
                continue

            words = text.split()
            result = passes_gopher_quality_filter(text)

            entry = {
                "url": url,
                "words": len(words),
                "mean_len": sum(len(w) for w in words) / max(len(words), 1),
                "preview": text[:120].replace("\n", " "),
                "pass": result,
            }

            if result:
                passed.append(entry)
            else:
                rejected.append(entry)

    print(
        f"\n通过: {len(passed)}, 拒绝: {len(rejected)}, "
        f"通过率: {len(passed) / (len(passed) + len(rejected)) * 100:.0f}%\n"
    )

    print("=== 通过的样例 ===")
    for e in passed[:10]:
        print(f"  词数={e['words']} 均词长={e['mean_len']:.1f}")
        print(f"  URL: {e['url'][:55]}")
        print(f"  预览: {e['preview'][:80]}")
        print()

    print("=== 被拒绝的样例 ===")
    for e in rejected[:10]:
        print(f"  词数={e['words']} 均词长={e['mean_len']:.1f}")
        print(f"  URL: {e['url'][:55]}")
        print(f"  预览: {e['preview'][:80]}")
        print()


if __name__ == "__main__":
    analyze_gopher_filter()
