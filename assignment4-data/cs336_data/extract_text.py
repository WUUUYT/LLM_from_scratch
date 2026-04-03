# cs336_data/extract_text.py

from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding


def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    """
    从原始HTML字节中提取纯文本。

    Args:
        html_bytes: 原始HTML的字节串
    Returns:
        提取出的纯文本字符串
    """
    # 第一步：尝试 UTF-8 解码（最常见，98.2%的网页）
    try:
        html_str = html_bytes.decode("utf-8")
    except (UnicodeDecodeError, ValueError):
        # 第二步：UTF-8失败，自动检测编码
        encoding = detect_encoding(html_bytes)
        try:
            html_str = html_bytes.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            # 最后兜底：忽略无法解码的字符
            html_str = html_bytes.decode("utf-8", errors="ignore")

    # 第三步：提取纯文本
    text = extract_plain_text(html_str)
    return text
