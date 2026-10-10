# -*- coding: utf-8 -*-
import sys, json, statistics as st, math
from collections import Counter, defaultdict
import importlib.util
import os
spec = importlib.util.spec_from_file_location("td_metric", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "td_metric.py"))
TD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TD)

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results_v3")
MODELS = ["ctrl_baseline", "specialist_beginner", "specialist_intermediate", "specialist_expert", "multilevel"]
LEVELS = ["beginner", "intermediate", "expert"]

def mannwhitneyu(x, y):
    nx, ny = len(x), len(y)
    allv = sorted([(v, 0) for v in x] + [(v, 1) for v in y])
    rank = {0: 0.0, 1: 0.0}
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j+1][0] == allv[i][0]:
            j += 1
        r = (i + 1 + j + 1) / 2.0
        for k in range(i, j+1):
            rank[allv[k][1]] += r
        i = j + 1
    r1 = rank[1]
    u1 = r1 - ny*(ny+1)/2.0
    u = min(u1, nx*ny - u1)
    mu = nx*ny/2.0
    ties = Counter(v for v, _ in allv)  # pooled value only, correct tie grouping
    tie_corr = sum((c**3 - c) for c in ties.values() if c > 1)
    sd = math.sqrt(nx*ny/12.0 * ((nx+ny+1) - tie_corr/((nx+ny)*(nx+ny-1)))) if (nx+ny) > 1 else 0
    z = (u - mu) / sd if sd else 0.0
    p = 2 * (1 - 0.5*(1 + math.erf(abs(z)/math.sqrt(2))))
    return u, p, z

results = {}
for model in MODELS:
    path = os.path.join(BASE, f"{model}.json")
    data = json.load(open(path, encoding="utf-8"))
    is_ctrl = (model == "ctrl_baseline")
    by_level_code = defaultdict(dict)
    for idx, r in enumerate(data):
        lv = r.get("level")
        if lv not in LEVELS:
            continue
        if is_ctrl:
            # no real seed field; these 3 records per code are byte-identical copies, not independent runs
            seed = str(idx)
        else:
            seed = r.get("exp", "").split("_s")[-1]
        by_level_code[lv].setdefault(r["code"], {})[seed] = TD.td(r.get("pred") or "")

    # dedup: for ctrl_baseline (no real repeats) just take one copy per code (n=50, true sample size)
    # for the other 4 models, average TD across the 3 real independent seeds per code (n=50, each value a 3-seed mean)
    by_level_vals = {}
    for lv in LEVELS:
        vals = []
        for code, seedvals in by_level_code[lv].items():
            if is_ctrl:
                vals.append(next(iter(seedvals.values())))
            else:
                vals.append(st.mean(seedvals.values()))
        by_level_vals[lv] = vals

    n_per = {lv: len(by_level_vals[lv]) for lv in LEVELS}
    means = {lv: st.mean(by_level_vals[lv]) for lv in LEVELS}
    sds = {lv: st.pstdev(by_level_vals[lv]) for lv in LEVELS}

    pairwise = {}
    pairs = [("beginner", "intermediate"), ("intermediate", "expert"), ("beginner", "expert")]
    for a, b in pairs:
        U, p, z = mannwhitneyu(by_level_vals[a], by_level_vals[b])
        n1, n2 = len(by_level_vals[a]), len(by_level_vals[b])
        r_eff = abs(z) / math.sqrt(n1 + n2)
        pairwise[f"{a}_vs_{b}"] = {"U": U, "p_raw": p, "r": r_eff, "n1": n1, "n2": n2}

    results[model] = {"n_per_level": n_per, "td_mean": means, "td_sd": sds, "pairwise": pairwise}
    print(f"=== {model} === n={n_per}")
    print(f"  TD mean: beginner={means['beginner']:.4f} intermediate={means['intermediate']:.4f} expert={means['expert']:.4f}")
    for k, v in pairwise.items():
        print(f"  {k}: U={v['U']:.1f} p_raw={v['p_raw']:.4g} r={v['r']:.3f}")

json.dump(results, open(r"C:\Users\86153\AppData\Local\Temp\p2_recompute_result.json", "w"), indent=2)
print("\nDONE")
