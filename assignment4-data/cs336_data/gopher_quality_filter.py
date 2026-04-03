# cs336_data/gopher_quality_filter.py
"""
CS336 Assignment 4 - 2.6 Gopher Quality Filters
基于Gopher论文附录A实现的启发式质量过滤规则
"""


def passes_gopher_quality_filter(text: str) -> bool:
    """
    判断文本是否通过Gopher质量过滤。

    过滤规则（满足任一条件则拒绝）：
    1. 词数 < 50 或 > 100,000
    2. 平均词长不在 [3, 10] 范围内
    3. 超过30%的行以省略号结尾
    4. 少于80%的词含有至少一个字母

    Args:
        text: 提取后的纯文本字符串
    Returns:
        True  → 通过过滤，是高质量文本
        False → 未通过，应当丢弃
    """

    # ── 预处理：分词 ──────────────────────────────────────
    # 用空白分词，速度快，适合大规模处理
    words = text.split()
    num_words = len(words)

    # ── 规则1：词数检查 ────────────────────────────────────
    # 太短 → 登录页/错误页/占位页
    # 太长 → 爬虫错误或垃圾堆砌
    if num_words < 50 or num_words > 100_000:
        return False

    # ── 规则2：平均词长检查 ────────────────────────────────
    # 过短 → 乱码或纯菜单
    # 过长 → URL堆砌或编码错误
    mean_word_length = sum(len(w) for w in words) / num_words
    if mean_word_length < 3 or mean_word_length > 10:
        return False

    # ── 规则3：省略号结尾行比例 ────────────────────────────
    # 付费墙/内容预览页的特征
    lines = [line for line in text.splitlines() if line.strip()]
    if lines:
        ellipsis_lines = sum(1 for line in lines if line.rstrip().endswith("..."))
        if ellipsis_lines / len(lines) > 0.3:
            return False

    # ── 规则4：含字母词的比例 ─────────────────────────────
    # 正常文本中绝大多数词应包含字母
    # 大量纯数字词 → 数据表格/垃圾内容
    alpha_word_count = sum(1 for w in words if any(c.isalpha() for c in w))
    if alpha_word_count / num_words < 0.8:
        return False

    # 通过所有规则
    return True
