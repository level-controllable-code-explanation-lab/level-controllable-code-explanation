"""
Multi-Level Evaluation Script V2 - FIXED TERM DENSITY
======================================================
FIX: Longest-match-first to avoid double-counting overlapping terms.
"""
import json, random, os, gc, time, re, statistics, sys
import torch
import numpy as np
from scipy import stats as scistats
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ===== FIXED CONFIG =====
BASE_DIR = "/root/autodl-tmp/multilevel_exp"
DATA_DIR = "/root/autodl-tmp/vclite_v2/data"
MODEL_PATH = "/root/autodl-tmp/gemma-2-9b-it"
TEST_FILE = "code_explanation_cn_multilevel_test.json"
RESULT_DIR = f"{BASE_DIR}/results_v2"

SEEDS = [42, 123, 456]
MAX_NEW_TOKENS = 512

LEVEL_INSTRUCTIONS = {
    "beginner": "你是一位面向编程初学者的教师。请用通俗易懂的语言解释以下代码，避免使用专业术语。\n\n代码：\n```python\n{code}\n```\n\n请给出初学者友好的解释：",
    "intermediate": "你是一位编程教育者。请用标准编程术语解释以下代码。\n\n代码：\n```python\n{code}\n```\n\n请给出解释：",
    "expert": "你是一位资深软件工程师。请用最精炼的技术语言解释以下代码，突出算法设计和复杂度分析。\n\n代码：\n```python\n{code}\n```\n\n请给出专家级解释：",
}

FEWSHOT_EXAMPLES = {
    "beginner": [
        {"code": "def add(a, b):\n    return a + b",
         "explanation": "把两个数字加起来。\n\n步骤分解：\n1. 接收两个数字\n2. 把它们相加\n3. 返回结果\n\n核心思想：实现最基本的加法运算"},
        {"code": "def is_even(n):\n    return n % 2 == 0",
         "explanation": "判断一个数是不是偶数。\n\n步骤分解：\n1. 用这个数除以2看余数\n2. 如果余数是0就是偶数\n3. 返回判断结果\n\n核心思想：用取余数判断奇偶"},
    ],
    "intermediate": [
        {"code": "def add(a, b):\n    return a + b",
         "explanation": "实现两个数的加法运算。\n\n算法模式：过程式逻辑\n\n关键代码段：\n1. 参数接收：a和b\n2. 返回a+b的结果\n\n注意事项：参数类型需支持+运算符"},
        {"code": "def is_even(n):\n    return n % 2 == 0",
         "explanation": "通过取模运算判断整数奇偶性。\n\n算法模式：过程式逻辑\n\n关键代码段：\n1. n % 2计算余数\n2. 与0比较返回布尔值\n\n注意事项：负数取模结果仍正确"},
    ],
    "expert": [
        {"code": "def add(a, b):\n    return a + b",
         "explanation": "二元加法封装。复杂度：O(1)。设计考量：直接委托Python内置+运算符，无类型约束依赖duck typing"},
        {"code": "def is_even(n):\n    return n % 2 == 0",
         "explanation": "模2判偶，O(1)操作。设计考量：取模运算适用于任意整数，位运算版n&1==0在性能敏感场景更优"},
    ],
}

os.makedirs(RESULT_DIR, exist_ok=True)

# ===== FIXED TECHNICAL TERMS (no overlapping shorter terms) =====
TECH_TERMS = [
    # Use longest terms first, no substring overlaps
    "时间复杂度", "空间复杂度", "动态规划", "数据结构",
    "位运算", "哈希表", "二分查找", "深度优先", "广度优先",
    "递归调用", "迭代遍历", "贪心算法", "分治策略", "回溯搜索",
    "类型注解", "装饰器模式", "生成器函数", "异常处理",
    "边界条件", "短路返回", "惰性求值", "依赖注入",
    "线程安全", "内存管理", "缓存策略", "设计模式",
    "O(log n)", "O(n log n)", "O(n)", "O(1)", "O(n",
    "DFS", "BFS", "API", "DP",
    "复杂度", "算法", "指针", "递归", "迭代", "排序",
    "栈", "队列", "堆", "树", "图", "哈希",
    "优化", "封装", "多态", "继承", "模块化",
]

# Sort by length descending for longest-match-first
TECH_TERMS_SORTED = sorted(TECH_TERMS, key=len, reverse=True)


def compute_term_density(text):
    """
    FIXED: Longest-match-first greedy algorithm.
    "时间复杂度" matches first, blocking later "复杂度" match on same span.
    """
    if not text:
        return 0.0
    matched_spans = []
    for term in TECH_TERMS_SORTED:
        for match in re.finditer(re.escape(term), text):
            start, end = match.span()
            # Check overlap with already-matched longer terms
            overlaps = False
            for ms, me in matched_spans:
                if start >= ms and end <= me:
                    overlaps = True
                    break
            if not overlaps:
                matched_spans.append((start, end))

    count = len(matched_spans)
    return count / max(len(text), 1) * 1000


def compute_sentence_length(text):
    sentences = re.split(r'[。！？\n]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sum(len(s) for s in sentences) / max(len(sentences), 1)


# ===== INFERENCE =====
def load_model(adapter_path=None):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="eager",
    )
    if adapter_path and os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, instruction, fewshot_exs=None):
    if fewshot_exs:
        fs_text = ""
        for ex in fewshot_exs:
            fs_text += f"示例代码：\n```python\n{ex['code']}\n```\n\n示例解释：\n{ex['explanation']}\n\n---\n\n"
        instruction = fs_text + instruction
    if hasattr(tokenizer, 'apply_chat_template'):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}], tokenize=False, add_generation_prompt=True)
    else:
        text = instruction
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()


def run_inference(name, adapter_path, test_data, fewshot=False):
    print(f"\n{'='*60}\nInference: {name}\n{'='*60}")
    results = []
    for lv in ["beginner", "intermediate", "expert"]:
        items = test_data[lv]
        model, tokenizer = load_model(adapter_path)
        for item in items:
            code = item.get("raw_code", item.get("code", ""))
            ref = item.get("output", "")
            inst = LEVEL_INSTRUCTIONS[lv].format(code=code)
            fs = FEWSHOT_EXAMPLES[lv][:2] if fewshot else None
            pred = generate(model, tokenizer, inst, fs)
            results.append({
                "experiment": name, "level": lv,
                "code": code[:200], "reference": ref, "prediction": pred,
                "pred_len": len(pred), "ref_len": len(ref),
                "term_density_pred": compute_term_density(pred),
                "term_density_ref": compute_term_density(ref),
                "sent_len_pred": compute_sentence_length(pred),
            })
        del model, tokenizer; gc.collect(); torch.cuda.empty_cache()
    return results


# ===== STATISTICS =====
def run_stats(results, exp_name):
    print(f"\n=== Stats: {exp_name} ===")
    by_lv = {"beginner": [], "intermediate": [], "expert": []}
    for r in results:
        if r["level"] in by_lv:
            by_lv[r["level"]].append(r["term_density_pred"])
    f_stat, p_val = scistats.f_oneway(by_lv["beginner"], by_lv["intermediate"], by_lv["expert"])
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    print(f"  ANOVA: F={f_stat:.2f}, p={p_val:.6f} {sig}")
    for lv1, lv2 in [("beginner", "intermediate"), ("intermediate", "expert"), ("beginner", "expert")]:
        t, p = scistats.ttest_ind(by_lv[lv1], by_lv[lv2])
        print(f"  {lv1} vs {lv2}: t={t:.2f}, p={p:.6f}")
    return {"anova_F": float(f_stat), "anova_p": float(p_val)}


# ===== MAIN =====
def main():
    print("=" * 60)
    print("EVALUATION V2 (Fixed Term Density)")
    print("=" * 60)

    with open(f"{DATA_DIR}/{TEST_FILE}", encoding="utf-8") as f:
        test_raw = json.load(f)
    test_data = {"beginner": [], "intermediate": [], "expert": []}
    for item in test_raw:
        lv = item.get("meta", {}).get("level", "")
        if lv in test_data:
            test_data[lv].append(item)
    print(f"Test: B={len(test_data['beginner'])} I={len(test_data['intermediate'])} E={len(test_data['expert'])}")

    all_results = {}

    # Zero-shot
    zs = run_inference("zero_shot", None, test_data, fewshot=False)
    all_results["zero_shot"] = zs
    with open(f"{RESULT_DIR}/zero_shot.json", "w", encoding="utf-8") as f:
        json.dump(zs, f, ensure_ascii=False, indent=2)

    # Few-shot
    fs = run_inference("few_shot", None, test_data, fewshot=True)
    all_results["few_shot"] = fs
    with open(f"{RESULT_DIR}/few_shot.json", "w", encoding="utf-8") as f:
        json.dump(fs, f, ensure_ascii=False, indent=2)

    # Specialists + Multi-Level
    for exp_name in ["specialist_beginner", "specialist_intermediate",
                      "specialist_expert", "multilevel"]:
        all_seed = []
        for seed in SEEDS:
            ap = f"{BASE_DIR}/models/{exp_name}/seed_{seed}"
            if not os.path.exists(f"{ap}/adapter_model.safetensors"):
                print(f"  SKIP {exp_name}/seed_{seed}")
                continue
            sr = run_inference(f"{exp_name}_seed{seed}", ap, test_data)
            all_seed.extend(sr)
        all_results[exp_name] = all_seed
        with open(f"{RESULT_DIR}/{exp_name}.json", "w", encoding="utf-8") as f:
            json.dump(all_seed, f, ensure_ascii=False, indent=2)

    # Stats & Summary
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"{'Experiment':<25} {'Beginner TD':>12} {'Intermed TD':>12} {'Expert TD':>12} {'ANOVA sig':>10}")
    print("-" * 75)
    for exp_name, res in all_results.items():
        if not res: continue
        by_lv = {"beginner": [], "intermediate": [], "expert": []}
        for r in res:
            if r["level"] in by_lv:
                by_lv[r["level"]].append(r["term_density_pred"])
        b = np.mean(by_lv["beginner"]) if by_lv["beginner"] else 0
        i = np.mean(by_lv["intermediate"]) if by_lv["intermediate"] else 0
        e = np.mean(by_lv["expert"]) if by_lv["expert"] else 0
        if by_lv["beginner"] and by_lv["intermediate"] and by_lv["expert"]:
            _, p = scistats.f_oneway(by_lv["beginner"], by_lv["intermediate"], by_lv["expert"])
            p_str = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
        else:
            p_str = "N/A"
        print(f"{exp_name:<25} {b:>12.2f} {i:>12.2f} {e:>12.2f} {p_str:>10}")

    print(f"\nDONE. Results: {RESULT_DIR}")


if __name__ == "__main__":
    main()
