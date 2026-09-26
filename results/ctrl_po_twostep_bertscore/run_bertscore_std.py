
import json, statistics
from bert_score import score

with open("/root/autodl-tmp/vclite_v2/data/code_explanation_cn_multilevel_test.json", "r", encoding="utf-8") as f:
    ts = json.load(f)

ref_by_level = {}
for level in ["beginner", "intermediate", "expert"]:
    ref_by_level[level] = [r["output"] for r in ts if r["meta"]["level"] == level]

with open("/root/autodl-tmp/multilevel_exp/results_v3/ctrl_baseline.json", "r", encoding="utf-8") as f:
    cb = json.load(f)
with open("/root/autodl-tmp/multilevel_exp/results_v3/multilevel_twostep.json", "r", encoding="utf-8") as f:
    twostep = json.load(f)

cb_preds = {level: [r["pred"] for r in cb if r["level"] == level][:50] for level in ["beginner", "intermediate", "expert"]}
ts_preds = {level: [r["pred"] for r in twostep if r["level"] == level] for level in ["beginner", "intermediate", "expert"]}

results = {}
for name, preds_by_level in (("ctrl_baseline", cb_preds), ("multilevel_twostep", ts_preds)):
    results[name] = {}
    for level in ["beginner", "intermediate", "expert"]:
        preds = preds_by_level[level]
        refs = ref_by_level[level]
        P, R, F1 = score(preds, refs, lang="zh", device="cuda", verbose=False, batch_size=8)
        f1 = F1.tolist()
        mean = statistics.mean(f1)
        std = statistics.stdev(f1)
        results[name][level] = {"f1_mean": mean, "f1_std": std, "n": len(f1)}
        print(name, level, "mean=", round(mean, 4), "std=", round(std, 4), flush=True)

with open("/root/autodl-tmp/multilevel_exp/bertscore_ctrlpo_twostep_std.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("ALL DONE", flush=True)
