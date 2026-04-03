# run_pii_analysis.py
from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.extract_text import extract_text_from_html_bytes
from cs336_data.mask_pii import mask_emails, mask_ips, mask_phone_numbers

with open("CC-MAIN-20250417135010-20250417165010-00065.warc.gz", "rb") as f:
    count = 0
    for record in ArchiveIterator(f, record_types=WarcRecordType.response):
        if count >= 20:
            break
        content_type = record.http_headers.get("Content-Type", "")
        if "text/html" not in content_type:
            continue

        html_bytes = record.reader.read()
        text = extract_text_from_html_bytes(html_bytes)

        t1, n1 = mask_emails(text)
        t2, n2 = mask_phone_numbers(t1)
        t3, n3 = mask_ips(t2)

        if n1 + n2 + n3 > 0:
            print(f"=== 文档 {count + 1} ===")
            print(f"邮件:{n1} 电话:{n2} IP:{n3}")
            # 显示替换前后对比
            for line in text.splitlines():
                masked = mask_emails(line)[0]
                masked = mask_phone_numbers(masked)[0]
                masked = mask_ips(masked)[0]
                if "|||" in masked:
                    print(f"  原文: {line.strip()}")
                    print(f"  替换: {masked.strip()}")
            print()
            count += 1
