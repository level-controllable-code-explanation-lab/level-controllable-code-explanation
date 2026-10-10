# -*- coding: utf-8 -*-
"""P3定稿：用权威47词表(td_metric.py，td()默认使用CANONICAL_TERMS)对data/train_750.json重算target TD，
产出 beginner 0.4413 / intermediate 3.0446 / expert 34.0260，替换Appendix A1陈旧的
0.185/2.835/29.135（旧值来自results_v3/<model>.json里预存的ref_td字段，该字段用的是
rework前、未排除单字/O(n残缺的52词表，非本论文声明的47词表口径，详见recompute_all.py
的agg("ref_td")实现——它只是对旧字段取均值，不是重新计算）。
"""
import json, statistics as st, importlib.util, os

spec = importlib.util.spec_from_file_location(
    "td_metric", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "td_metric.py")
)
TD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TD)

TRAIN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "train_750.json")

data = json.load(open(TRAIN_PATH, encoding="utf-8"))
by_level = {"beginner": [], "intermediate": [], "expert": []}
for r in data:
    lv = r["meta"]["level"]
    by_level[lv].append(TD.td(r["output"]))

print("total records:", len(data))
for lv in ["beginner", "intermediate", "expert"]:
    vals = by_level[lv]
    print(f"{lv}: n={len(vals)} mean_TD={st.mean(vals):.4f} sd={st.pstdev(vals):.4f}")
