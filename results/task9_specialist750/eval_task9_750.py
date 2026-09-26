"""
评估 task9 新训练的 specialist_*_750 (9个adapter: 3层级 x 3种子)。
每个adapter跑全部三层测试集(50题/层)，用统一的47条TD词表(td_metric.py)。
输出格式对齐 results_v3 (exp/level/code/pred/ref/pred_len/td/ref_td)，
存到独立目录 results_v4_750，避免与旧的212条specialist结果混淆。
"""
import json, os, gc, time, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.path.insert(0, "/root/autodl-tmp/rework")
from td_metric import td

BASE_DIR = "/root/autodl-tmp/multilevel_exp"
MODEL_PATH = "/root/autodl-tmp/gemma-2-9b-it"
DATA_DIR = "/root/autodl-tmp/vclite_v2/data"
RESULT_DIR = f"{BASE_DIR}/results_v4_750"
os.makedirs(RESULT_DIR, exist_ok=True)

LEVEL_INSTRUCTIONS = {
    "beginner": "你是一位面向编程初学者的教师。请用通俗易懂的语言解释以下代码，避免使用专业术语。\n\n代码：\n```python\n{code}\n```\n\n请给出初学者友好的解释：",
    "intermediate": "你是一位编程教育者。请用标准编程术语解释以下代码。\n\n代码：\n```python\n{code}\n```\n\n请给出解释：",
    "expert": "你是一位资深软件工程师。请用最精炼的技术语言解释以下代码，突出算法设计和复杂度分析。\n\n代码：\n```python\n{code}\n```\n\n请给出专家级解释：",
}

with open(f"{DATA_DIR}/code_explanation_cn_multilevel_test.json", encoding="utf-8") as f:
    test_raw = json.load(f)
test_data = {"beginner": [], "intermediate": [], "expert": []}
for item in test_raw:
    lv = item.get("meta", {}).get("level", "")
    if lv in test_data:
        test_data[lv].append(item)
print(f"Test: B={len(test_data['beginner'])} I={len(test_data['intermediate'])} E={len(test_data['expert'])}", flush=True)


def load_model(adapter_path=None):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb,
        device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="eager")
    if adapter_path and os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tok


def batch_generate(model, tok, instructions, max_new=256, batch_size=8):
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
    print(f"\n{'='*60}\n{name}\n{'='*60}", flush=True)
    model, tok = load_model(adapter_path)
    results = []
    for lv in ["beginner", "intermediate", "expert"]:
        items = test_data[lv]
        instructions, refs, codes = [], [], []
        for item in items:
            code = item.get("raw_code", item.get("code", ""))
            inst = LEVEL_INSTRUCTIONS[lv].format(code=code)
            instructions.append(inst)
            refs.append(item.get("output", ""))
            codes.append(code[:200])
        print(f"  {lv}: {len(instructions)} samples...", end=" ", flush=True)
        t0 = time.time()
        preds = batch_generate(model, tok, instructions)
        print(f"OK ({time.time()-t0:.1f}s)", flush=True)
        for code, ref, pred in zip(codes, refs, preds):
            results.append({
                "exp": name, "level": lv, "code": code, "pred": pred, "ref": ref,
                "pred_len": len(pred), "td": td(pred), "ref_td": td(ref)
            })
    del model, tok; gc.collect(); torch.cuda.empty_cache(); time.sleep(3)
    return results


LEVELS = ["beginner", "intermediate", "expert"]
SEEDS = [42, 123, 456]

for level in LEVELS:
    exp = f"specialist_{level}_750"
    save_path = f"{RESULT_DIR}/{exp}.json"
    if os.path.exists(save_path):
        print(f"SKIP {exp} (already evaluated)", flush=True)
        continue
    all_seed = []
    for seed in SEEDS:
        ap = f"{BASE_DIR}/models/{exp}/seed_{seed}"
        if not os.path.exists(f"{ap}/adapter_model.safetensors"):
            print(f"  MISSING adapter: {ap}", flush=True)
            continue
        sr = run_experiment(f"{exp}_s{seed}", ap, test_data)
        all_seed.extend(sr)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_seed, f, ensure_ascii=False, indent=2)
    print(f"SAVED {save_path} ({len(all_seed)} records)", flush=True)

print("\nALL DONE task9_eval", flush=True)
