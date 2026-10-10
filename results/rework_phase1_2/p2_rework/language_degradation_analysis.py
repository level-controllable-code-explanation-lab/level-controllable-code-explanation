# -*- coding: utf-8 -*-
"""
Section 4.7 language-degradation analysis and Table 5 reproduction.

Degradation is defined as zero Chinese characters in the generated output
(regex \\u4e00-\\u9fff count == 0). Run from any location; all paths are
resolved relative to this file.
"""
import json
import os
import re

RESULTS_V3 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results_v3")

MODELS = [
    "zero_shot", "few_shot", "ctrl_baseline",
    "specialist_beginner", "specialist_intermediate", "specialist_expert",
    "multilevel", "multilevel_onestep", "multilevel_twostep",
]
LEVELS = ["beginner", "intermediate", "expert"]

SEEDED_MODELS = {"ctrl_baseline", "specialist_beginner", "specialist_intermediate",
                  "specialist_expert", "multilevel"}


def is_zero_chinese(text):
    return len(re.findall(r"[一-鿿]", text)) == 0


def load(model):
    path = os.path.join(RESULTS_V3, f"{model}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def print_27cell_matrix():
    print("=== 27-cell language degradation matrix (model x level) ===")
    print(f"{'model':28s} {'level':13s} {'n':>5s} {'degraded':>9s} {'pct':>7s}")
    for model in MODELS:
        data = load(model)
        for level in LEVELS:
            rows = [r for r in data if r["level"] == level]
            n = len(rows)
            degraded = sum(1 for r in rows if is_zero_chinese(r["pred"]))
            pct = 100.0 * degraded / n if n else 0.0
            print(f"{model:28s} {level:13s} {n:5d} {degraded:9d} {pct:6.1f}%")
    print()


def per_code_breakdown(model, level):
    """For seeded models (150 rows = 50 code x 3 seed): classify each code as
    all-3-seeds-degraded, none-degraded, or mixed."""
    data = load(model)
    rows = [r for r in data if r["level"] == level]
    by_code = {}
    for r in rows:
        by_code.setdefault(r["code"], []).append(is_zero_chinese(r["pred"]))
    all3 = set(c for c, v in by_code.items() if all(v))
    none = set(c for c, v in by_code.items() if not any(v))
    mixed = set(c for c in by_code if c not in all3 and c not in none)
    return all3, none, mixed


def print_code_level_analysis():
    print("=== Code-level breakdown (Table 5 source) ===")
    targets = [
        ("specialist_expert", "intermediate"),
        ("specialist_beginner", "expert"),
        ("specialist_intermediate", "intermediate"),
    ]
    results = {}
    for model, level in targets:
        all3, none, mixed = per_code_breakdown(model, level)
        results[(model, level)] = all3
        print(f"{model}[{level}]: codes=50 all3-degraded={len(all3)} "
              f"none-degraded={len(none)} mixed={len(mixed)}")
    print()

    zs = load("zero_shot")
    zs_rows = [r for r in zs if r["level"] == "intermediate"]
    zs_degraded = set(r["code"] for r in zs_rows if is_zero_chinese(r["pred"]))
    print(f"zero_shot[intermediate] degraded codes: {len(zs_degraded)} / 50")

    se_all3 = results[("specialist_expert", "intermediate")]
    overlap = zs_degraded & se_all3
    print(f"overlap: zero_shot[intermediate]-degraded ({len(zs_degraded)}) vs "
          f"specialist_expert[intermediate]-all3-degraded ({len(se_all3)}): "
          f"{len(overlap)} / {len(zs_degraded)}")


if __name__ == "__main__":
    print_27cell_matrix()
    print_code_level_analysis()
