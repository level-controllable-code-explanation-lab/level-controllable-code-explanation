import os, json, statistics as st
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from bert_score import score as bert_score_fn

RESULTS_DIR = "/root/autodl-tmp/multilevel_exp/results_v4_750"
OUT_FILE = "/root/autodl-tmp/multilevel_exp/bertscore_results_750.json"
CONDITIONS = ["specialist_beginner_750", "specialist_intermediate_750", "specialist_expert_750"]
LEVELS = ["beginner", "intermediate", "expert"]
MODEL = "bert-base-chinese"

out = {}
for cond in CONDITIONS:
    path = f"{RESULTS_DIR}/{cond}.json"
    if not os.path.exists(path):
        print(f"MISSING {path}", flush=True)
        continue
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"\n{cond}: {len(data)} records", flush=True)

    cond_result = {"per_level": {}, "overall": {}}
    for lv in LEVELS:
        items = [r for r in data if r["level"] == lv]
        if not items:
            continue
        preds = [r["pred"] for r in items]
        refs = [r["ref"] for r in items]
        P, R, F1 = bert_score_fn(preds, refs, lang="zh", model_type=MODEL, device="cuda", verbose=False)
        f1_list = F1.tolist()
        cond_result["per_level"][lv] = {
            "n": len(items),
            "f1_mean": st.mean(f1_list),
            "f1_std": st.stdev(f1_list) if len(f1_list) > 1 else 0.0,
            "p_mean": P.mean().item(),
            "r_mean": R.mean().item(),
        }
        print(f"  {lv}: n={len(items)} F1={cond_result['per_level'][lv]['f1_mean']:.4f}", flush=True)

    all_preds = [r["pred"] for r in data]
    all_refs = [r["ref"] for r in data]
    P, R, F1 = bert_score_fn(all_preds, all_refs, lang="zh", model_type=MODEL, device="cuda", verbose=False)
    f1_list = F1.tolist()
    cond_result["overall"] = {
        "n": len(data),
        "f1_mean": st.mean(f1_list),
        "f1_std": st.stdev(f1_list) if len(f1_list) > 1 else 0.0,
    }
    print(f"  overall: F1={cond_result['overall']['f1_mean']:.4f}", flush=True)
    out[cond] = cond_result

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"\nSAVED {OUT_FILE}", flush=True)
print("ALL DONE bertscore_750", flush=True)
