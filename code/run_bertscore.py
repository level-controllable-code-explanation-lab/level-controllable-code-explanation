
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json, statistics
from bert_score import score as bert_score_fn

RESULTS_DIR = "/root/autodl-tmp/multilevel_exp/results_v3"
OUT_FILE = "/root/autodl-tmp/multilevel_exp/bertscore_results.json"
CONDITIONS = ["zero_shot", "few_shot", "multilevel",
              "specialist_beginner", "specialist_intermediate", "specialist_expert"]
LEVELS = ["beginner", "intermediate", "expert"]
MODEL = "bert-base-chinese"

results = {}
for cond in CONDITIONS:
    fpath = os.path.join(RESULTS_DIR, cond + ".json")
    data = json.load(open(fpath, encoding="utf-8"))
    print(f"\n=== {cond} (n={len(data)}) ===", flush=True)

    by_level = {lv: [] for lv in LEVELS}
    for item in data:
        lv = item.get("level", "?")
        if lv in by_level:
            by_level[lv].append((item["pred"], item["ref"]))

    cond_result = {}
    all_preds, all_refs = [], []
    for lv in LEVELS:
        pairs = by_level[lv]
        if not pairs:
            continue
        preds = [p for p, r in pairs]
        refs  = [r for p, r in pairs]
        all_preds.extend(preds)
        all_refs.extend(refs)
        P, R, F1 = bert_score_fn(preds, refs, model_type=MODEL,
                                  lang="zh", verbose=False, device="cuda")
        f1_vals = F1.tolist()
        cond_result[lv] = {
            "mean": round(statistics.mean(f1_vals), 4),
            "std":  round(statistics.stdev(f1_vals), 4),
            "n":    len(f1_vals)
        }
        print(f"  {lv}: {cond_result[lv]['mean']:.4f} +- {cond_result[lv]['std']:.4f}", flush=True)

    P, R, F1 = bert_score_fn(all_preds, all_refs, model_type=MODEL,
                              lang="zh", verbose=False, device="cuda")
    f1_all = F1.tolist()
    cond_result["overall"] = {
        "mean": round(statistics.mean(f1_all), 4),
        "std":  round(statistics.stdev(f1_all), 4),
        "n":    len(f1_all)
    }
    print(f"  overall: {cond_result['overall']['mean']:.4f} +- {cond_result['overall']['std']:.4f}", flush=True)
    results[cond] = cond_result

json.dump(results, open(OUT_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"\nSaved: {OUT_FILE}", flush=True)
