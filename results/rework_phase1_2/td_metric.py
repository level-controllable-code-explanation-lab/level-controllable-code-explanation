# -*- coding: utf-8 -*-
"""
Technical Density (TD) —— 唯一权威实现。

背景：原代码库中存在 4 套词表（52/40/38/35 条）和 2 种计数算法
（数所有出现次数 vs 每词最多计 1 次），导致不同脚本产出的 TD 不可比。
本模块统一为：52 条词表 + 最长优先匹配 + 统计所有出现次数。

定义：
    TD(e) = 1000 * N(e) / L(e)
    N(e) = 词表 V 在 e 中的匹配总次数（最长优先，已匹配区间不重复计数）
    L(e) = e 的字符总数（含标点与空白，不做分词）

匹配规则：按词长降序遍历 V；对每个词用正则找出所有出现位置；
若某位置已被更长的词覆盖则跳过（避免"复杂度"重复计入"时间复杂度"）。

词表来源：evaluate_multilevel_v2.py（主评估脚本）中的 TECH_TERMS，原样照搬，未增删。

版本：v1.0-unified（2026-09）
"""
from __future__ import annotations
import re

TD_VERSION = "v1.0-unified"
VOCAB_SOURCE = "evaluate_multilevel_v2.py::TECH_TERMS (52 entries, verbatim)"

TECH_TERMS: list[str] = [
    "时间复杂度", "空间复杂度", "动态规划", "数据结构", "位运算", "哈希表",
    "二分查找", "深度优先", "广度优先", "递归调用", "迭代遍历", "贪心算法",
    "分治策略", "回溯搜索", "类型注解", "装饰器模式", "生成器函数", "异常处理",
    "边界条件", "短路返回", "惰性求值", "依赖注入", "线程安全", "内存管理",
    "缓存策略", "设计模式", "O(log n)", "O(n log n)", "O(n)", "O(1)", "O(n",
    "DFS", "BFS", "API", "DP", "复杂度", "算法", "指针", "递归", "迭代",
    "排序", "栈", "队列", "堆", "树", "图", "哈希", "优化", "封装", "多态",
    "继承", "模块化",
]

assert len(TECH_TERMS) == 52, "原始词表必须是 52 条"
assert len(set(TECH_TERMS)) == 52, "词表不得有重复"

# ── 被排除的条目及理由 ──────────────────────────────────────────────
# 单字条目在复合词内部误命中。最长优先只能挡住词表内的更长词，挡不住词表外的词。
# 实测影响：multilevel beginner 层 TD 2.454 -> 剔除后 0.054，即 98% 来自误命中
# （beginner 解释大量使用日常类比：图片/树立/堆放/栈房）。
SINGLE_CHAR_TERMS: list[str] = [t for t in TECH_TERMS if len(t) == 1]   # 栈 堆 树 图

# "O(n" 是残缺前缀，会重复命中 "O(n)" / "O(n log n)" 之外的任意 O(n… 形式。
FRAGMENT_TERMS: list[str] = ["O(n"]

EXCLUDED: list[str] = SINGLE_CHAR_TERMS + FRAGMENT_TERMS

# ── 权威词表：47 条 ─────────────────────────────────────────────────
# = 52 条主评估词表 − 4 个单字条目 − 1 个残缺前缀
CANONICAL_TERMS: list[str] = [t for t in TECH_TERMS if t not in EXCLUDED]

VOCAB_SIZE = len(CANONICAL_TERMS)

# 最长优先：保证 "时间复杂度" 先于 "复杂度"、"O(n log n)" 先于 "O(n)" 匹配
TECH_TERMS_SORTED: list[str] = sorted(CANONICAL_TERMS, key=len, reverse=True)

# 原始 52 条（含被排除项），仅用于敏感性分析与复现历史数字
TECH_TERMS_RAW52_SORTED: list[str] = sorted(TECH_TERMS, key=len, reverse=True)


def count_matches(text: str, terms: list[str] | None = None) -> int:
    """最长优先、不重叠地统计术语匹配总次数。"""
    if not text:
        return 0
    vocab = TECH_TERMS_SORTED if terms is None else sorted(set(terms), key=len, reverse=True)
    claimed: list[tuple[int, int]] = []
    for term in vocab:
        for m in re.finditer(re.escape(term), text):
            s, e = m.span()
            if any(s >= cs and e <= ce for cs, ce in claimed):
                continue
            claimed.append((s, e))
    return len(claimed)


def td(text: str, terms: list[str] | None = None) -> float:
    """TD 主函数：每 1000 字符的术语匹配次数。"""
    return 1000.0 * count_matches(text, terms) / max(len(text or ""), 1)


def td_raw52(text: str) -> float:
    """敏感性分析：用原始 52 条（含单字与残缺前缀）计算，量化误命中影响。"""
    return td(text, TECH_TERMS)


def td_distinct_legacy(text: str, vocab: list[str] | None = None) -> float:
    """
    旧算法（ctrl_baseline.py 使用）：每个词最多计 1 次，上限被词表大小卡死。
    仅用于复现历史数字与做对比，不作为论文指标。
    """
    if not text:
        return 0.0
    sv = sorted(set(vocab or CANONICAL_TERMS), key=len, reverse=True)
    remaining = text
    count = 0
    for term in sv:
        if term in remaining:
            count += 1
            remaining = remaining.replace(term, " ")
    return 1000.0 * count / max(len(text), 1)


def matched_terms(text: str) -> dict[str, int]:
    """返回每个术语的命中次数，用于人工抽查误命中。"""
    if not text:
        return {}
    claimed: list[tuple[int, int]] = []
    hits: dict[str, int] = {}
    for term in TECH_TERMS_SORTED:
        n = 0
        for m in re.finditer(re.escape(term), text):
            s, e = m.span()
            if any(s >= cs and e <= ce for cs, ce in claimed):
                continue
            claimed.append((s, e))
            n += 1
        if n:
            hits[term] = n
    return hits


if __name__ == "__main__":
    print("TD 版本   :", TD_VERSION)
    print("词表来源  :", VOCAB_SOURCE)
    print("原始条目  :", len(TECH_TERMS))
    print("权威条目  :", VOCAB_SIZE)
    print("已排除    :", EXCLUDED, "(单字误命中 + 残缺前缀)")

    demo = "这个函数用递归实现二分查找，时间复杂度是 O(log n)，需要额外的哈希表做缓存策略。"
    print("\n[正常示例]", demo)
    print("  权威 TD =", round(td(demo), 3), "命中:", matched_terms(demo))

    bad = "这张图片展示了树立目标的过程，栈房里堆放着杂物。"
    print("\n[单字误命中演示]", bad)
    print("  原始52条 TD =", round(td_raw52(bad), 3), "  <- 零技术内容却得高分")
    print("  权威47条 TD =", round(td(bad), 3), "  <- 正确")

    print("\n[算法对比]", demo)
    print("  出现次数法(权威) =", round(td(demo), 3))
    print("  distinct旧算法   =", round(td_distinct_legacy(demo), 3))
