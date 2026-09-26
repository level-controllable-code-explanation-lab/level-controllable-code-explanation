"""
Batch evaluation - same results as sequential, 5-6x faster
Uses greedy decoding for deterministic output.
"""
import json, os, gc, time, re
import torch
import numpy as np
from scipy import stats as scistats
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_DIR = "/root/autodl-tmp/multilevel_exp"
MODEL_PATH = "/root/autodl-tmp/gemma-2-9b-it"
DATA_DIR = "/root/autodl-tmp/vclite_v2/data"
RESULT_DIR = f"{BASE_DIR}/results_v3"
os.makedirs(RESULT_DIR, exist_ok=True)

LEVEL_INSTRUCTIONS = {
    "beginner": "你是一位面向编程初学者的教师。请用通俗易懂的语言解释以下代码，避免使用专业术语。\n\n代码：\n```python\n{code}\n```\n\n请给出初学者友好的解释：",
    "intermediate": "你是一位编程教育者。请用标准编程术语解释以下代码。\n\n代码：\n```python\n{code}\n```\n\n请给出解释：",
    "expert": "你是一位资深软件工程师。请用最精炼的技术语言解释以下代码，突出算法设计和复杂度分析。\n\n代码：\n```python\n{code}\n```\n\n请给出专家级解释：",
}

TECH_TERMS = sorted([
    "时间复杂度", "空间复杂度", "动态规划", "数据结构", "位运算", "哈希表", "二分查找",
    "深度优先", "广度优先", "递归调用", "迭代遍历", "贪心算法", "分治策略", "回溯搜索",
    "短路返回", "惰性求值", "依赖注入", "内存管理", "缓存策略", "设计模式", "异常处理",
    "策略模式", "工厂模式", "O(log n)", "O(n log n)", "O(N", "O(1)", "O(log", "O(n", "BFS", "DFS", "DP",
    "算法", "指针", "递归", "迭代", "排序", "优化", "封装", "模块化",
], key=len, reverse=True)

def term_density(text):
    if not text: return 0.0
    matched = []
    for term in TECH_TERMS:
        for m in re.finditer(re.escape(term), text):
            s, e = m.span()
            if not any(s >= ms and e <= me for ms, me in matched):
                matched.append((s, e))
    return len(matched) / max(len(text), 1) * 1000

# Load test data
with open(f"{DATA_DIR}/code_explanation_cn_multilevel_test.json", encoding="utf-8") as f:
    test_raw = json.load(f)
test_data = {"beginner": [], "intermediate": [], "expert": []}
for item in test_raw:
    lv = item.get("meta", {}).get("level", "")
    if lv in test_data: test_data[lv].append(item)
print(f"Test: B={len(test_data['beginner'])} I={len(test_data['intermediate'])} E={len(test_data['expert'])}")

def load_model(adapter_path=None):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, padding_side="left")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb,
        device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="eager")
    if adapter_path and os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tok

def batch_generate(model, tok, instructions, max_new=256, batch_size=8):
    """Batch inference in small chunks to avoid OOM"""
    all_preds = []
    for chunk_start in range(0, len(instructions), batch_size):
        chunk = instructions[chunk_start:chunk_start + batch_size]
        texts = []
        for inst in chunk:
            if hasattr(tok, 'apply_chat_template'):
                t = tok.apply_chat_template([{"role": "user", "content": inst}],
                    tokenize=False, add_generation_prompt=True)
            else:
                t = inst
            texts.append(t)

        inputs = tok(texts, return_tensors="pt", truncation=True, max_length=1024, padding=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                pad_token_id=tok.eos_token_id)

        for i in range(len(chunk)):
            pred = tok.decode(outputs[i][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            all_preds.append(pred)
    return all_preds

def run_experiment(name, adapter_path, test_data):
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    model, tok = load_model(adapter_path)
    results = []
    for lv in ["beginner", "intermediate", "expert"]:
        items = test_data[lv]
        # Build all instructions for this level
        instructions = []
        refs = []
        codes = []
        for item in items:
            code = item.get("raw_code", item.get("code", ""))
            inst = LEVEL_INSTRUCTIONS[lv].format(code=code)
            instructions.append(inst)
            refs.append(item.get("output", ""))
            codes.append(code[:200])

        print(f"  {lv}: {len(instructions)} samples...", end=" ", flush=True)
        t0 = time.time()
        preds = batch_generate(model, tok, instructions)
        elapsed = time.time() - t0
        print(f"OK ({elapsed:.1f}s)")

        for code, ref, pred in zip(codes, refs, preds):
            results.append({
                "exp": name, "level": lv, "code": code, "pred": pred, "ref": ref,
                "pred_len": len(pred), "td": term_density(pred), "ref_td": term_density(ref)
            })
    del model, tok; gc.collect(); torch.cuda.empty_cache(); time.sleep(3)
    return results

# === RUN ===
all_results = {}

# Load existing
for name in ["zero_shot", "specialist_beginner"]:
    save_path = f"{RESULT_DIR}/{name}.json"
    if os.path.exists(save_path):
        print(f"{name}: loading saved")
        with open(save_path) as f:
            all_results[name] = json.load(f)

# Specialist intermediate
exp = "specialist_intermediate"
save_path = f"{RESULT_DIR}/{exp}.json"
if not os.path.exists(save_path):
    all_seed = []
    for seed in [42, 123, 456]:
        ap = f"{BASE_DIR}/models/{exp}/seed_{seed}"
        if os.path.exists(f"{ap}/adapter_model.safetensors"):
            sr = run_experiment(f"{exp}_s{seed}", ap, test_data)
            all_seed.extend(sr)
    all_results[exp] = all_seed
    with open(save_path, "w") as f: json.dump(all_seed, f, ensure_ascii=False, indent=2)
else:
    with open(save_path) as f: all_results[exp] = json.load(f)

# Specialist expert (newly trained)
exp = "specialist_expert"
save_path = f"{RESULT_DIR}/{exp}.json"
if not os.path.exists(save_path):
    all_seed = []
    for seed in [42, 123, 456]:
        ap = f"{BASE_DIR}/models/{exp}/seed_{seed}"
        if os.path.exists(f"{ap}/adapter_model.safetensors"):
            sr = run_experiment(f"{exp}_s{seed}", ap, test_data)
            all_seed.extend(sr)
    all_results[exp] = all_seed
    with open(save_path, "w") as f: json.dump(all_seed, f, ensure_ascii=False, indent=2)
else:
    with open(save_path) as f: all_results[exp] = json.load(f)

# Multilevel (newly trained)
exp = "multilevel"
save_path = f"{RESULT_DIR}/{exp}.json"
if not os.path.exists(save_path):
    all_multi = []
    for seed in [42, 123, 456]:
        ap = f"{BASE_DIR}/models/{exp}/seed_{seed}"
        if os.path.exists(f"{ap}/adapter_model.safetensors"):
            sr = run_experiment(f"{exp}_s{seed}", ap, test_data)
            all_multi.extend(sr)
    all_results[exp] = all_multi
    with open(save_path, "w") as f: json.dump(all_multi, f, ensure_ascii=False, indent=2)
else:
    with open(save_path) as f: all_results[exp] = json.load(f)

# === SUMMARY ===
print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
print(f"{'Experiment':<25} {'Beginner TD':>12} {'Intermed TD':>12} {'Expert TD':>12} {'ANOVA':>10}")
print("-" * 70)
for name in ["zero_shot", "specialist_beginner", "specialist_intermediate", "specialist_expert", "multilevel"]:
    if name not in all_results: continue
    res = all_results[name]
    by = {"beginner": [], "intermediate": [], "expert": []}
    for r in res:
        if r["level"] in by: by[r["level"]].append(r["td"])
    b, i, e = np.mean(by["beginner"]), np.mean(by["intermediate"]), np.mean(by["expert"])
    if by["beginner"] and by["intermediate"] and by["expert"]:
        _, p = scistats.f_oneway(by["beginner"], by["intermediate"], by["expert"])
        stt = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "ns"
    else:
        stt = "N/A"
    print(f"{name:<25} {b:>12.2f} {i:>12.2f} {e:>12.2f} {stt:>10}")

print(f"\nDONE. Results: {RESULT_DIR}")
