"""
Multi-seed robustness experiment for Gemma-2-9B two_turn.
Trains with seeds 0, 1, 2 and evaluates each checkpoint.
seed=42 already done; this adds 0,1,2 to give 4-seed statistics.

Usage:
  python3 step_multi_seed.py
"""
import os, sys, json, time, math, re, difflib, argparse
from datetime import datetime

import torch
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

MODEL_PATH   = "/root/autodl-tmp/gemma-2-9b-it"
DATA_DIR     = "/root/autodl-tmp/twostep_exp/sft_data"
TEST_PATH    = "/root/autodl-tmp/twostep_exp/data/step2_test.json"
OUTPUT_ROOT  = "/root/autodl-tmp/twostep_exp/checkpoints_multi_seed"
RESULTS_DIR  = "/root/autodl-tmp/twostep_exp/eval_results"

SEEDS = [0, 1, 2]   # seed=42 already exists

HPARAMS = {
    "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj","v_proj","k_proj","o_proj",
                            "gate_proj","up_proj","down_proj"],
    "learning_rate": 2e-5, "num_epochs": 5,
    "per_device_train_batch_size": 2, "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 8, "max_seq_length": 1024,
    "warmup_ratio": 0.1, "weight_decay": 0.01,
    "eval_steps": 3, "save_steps": 3, "logging_steps": 1,
    "early_stopping_patience": 3,
}

OV = r"概述|核心功能|一句话"
ST = r"步骤分解|分步解析|步骤"
CO = r"核心思想|总结|关键要点"

# ── helpers ──────────────────────────────────────────────────────────────────

def rouge_l(ref, cand):
    rc, cc = list(ref), list(cand)
    if not rc or not cc: return 0.0
    m = difflib.SequenceMatcher(None, rc, cc)
    lcs = sum(b.size for b in m.get_matching_blocks())
    r = lcs/len(rc); p = lcs/len(cc)
    return 2*r*p/(r+p) if (r+p) > 0 else 0.0

def check_struct(text):
    return bool(re.search(OV,text)), bool(re.search(ST,text)), bool(re.search(CO,text))

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def format_conversation(messages, tokenizer):
    system_prompt = ""
    clean = []
    for m in messages:
        if m["role"] == "system":
            system_prompt = m["content"] + "\n\n"
        else:
            clean.append(m)
    if system_prompt and clean and clean[0]["role"] == "user":
        clean[0] = dict(clean[0])
        clean[0]["content"] = system_prompt + clean[0]["content"]
    return tokenizer.apply_chat_template(clean, tokenize=False, add_generation_prompt=False)

def prepare_dataset(data, tokenizer, max_len):
    texts = [format_conversation(s["messages"], tokenizer) for s in data]
    ds = Dataset.from_dict({"text": texts})
    ds = ds.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=max_len, padding=False),
        batched=True, remove_columns=["text"],
    )
    return ds

# ── training ─────────────────────────────────────────────────────────────────

def train_one(seed):
    torch.manual_seed(seed)
    ts = datetime.now().strftime("%m%d_%H%M")
    out_dir = f"{OUTPUT_ROOT}/two_turn_seed{seed}_{ts}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*55}\nTraining two_turn seed={seed}\n{'='*55}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=HPARAMS["lora_r"], lora_alpha=HPARAMS["lora_alpha"],
        lora_dropout=HPARAMS["lora_dropout"],
        target_modules=HPARAMS["lora_target_modules"],
        bias="none", task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    train_ds = prepare_dataset(load_json(f"{DATA_DIR}/two_turn_train.json"), tokenizer, HPARAMS["max_seq_length"])
    val_ds   = prepare_dataset(load_json(f"{DATA_DIR}/two_turn_val.json"),   tokenizer, HPARAMS["max_seq_length"])

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=HPARAMS["num_epochs"],
        per_device_train_batch_size=HPARAMS["per_device_train_batch_size"],
        per_device_eval_batch_size=HPARAMS["per_device_eval_batch_size"],
        gradient_accumulation_steps=HPARAMS["gradient_accumulation_steps"],
        learning_rate=HPARAMS["learning_rate"],
        warmup_ratio=HPARAMS["warmup_ratio"], weight_decay=HPARAMS["weight_decay"],
        optim="paged_adamw_32bit", bf16=True, fp16=False,
        logging_steps=HPARAMS["logging_steps"],
        eval_strategy="steps", eval_steps=HPARAMS["eval_steps"],
        save_strategy="steps", save_steps=HPARAMS["save_steps"],
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, seed=seed, report_to="none",
        dataloader_pin_memory=False,
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=HPARAMS["early_stopping_patience"])],
    )
    t0 = time.time()
    trainer.train()
    elapsed = int(time.time() - t0)

    final = f"{out_dir}/final"
    trainer.save_model(final); tokenizer.save_pretrained(final)

    best_loss = min(e["eval_loss"] for e in trainer.state.log_history if "eval_loss" in e)
    print(f"Done: best_eval_loss={best_loss:.4f}, elapsed={elapsed}s")
    return final, best_loss, elapsed

# ── evaluation ───────────────────────────────────────────────────────────────

def generate(model, tokenizer, prompt, max_new=512):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new,
                             do_sample=False, temperature=1.0,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def eval_one(ckpt_path, seed):
    print(f"\nEvaluating seed={seed} from {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    test_data = load_json(TEST_PATH)
    results = []
    for i, s in enumerate(test_data):
        # Turn1: plan
        p1_msgs = [{"role":"user","content":f"请分析以下代码的执行逻辑，生成步骤规划：\n```python\n{s['code']}\n```"}]
        p1 = tokenizer.apply_chat_template(p1_msgs, tokenize=False, add_generation_prompt=True)
        plan = generate(model, tokenizer, p1, 256)

        # Turn2: explanation
        p2_msgs = [
            {"role":"user","content":f"请分析以下代码的执行逻辑，生成步骤规划：\n```python\n{s['code']}\n```"},
            {"role":"assistant","content":plan},
            {"role":"user","content":"请基于以上规划，生成完整的代码解释，包含一句话概述、步骤分解和核心思想三个部分："},
        ]
        p2 = tokenizer.apply_chat_template(p2_msgs, tokenize=False, add_generation_prompt=True)
        explanation = generate(model, tokenizer, p2, 512)

        ov, st, co = check_struct(explanation)
        rl = rouge_l(s["explanation"], explanation)
        results.append({
            "task_id": s["task_id"], "seed": seed,
            "generated_explanation": explanation,
            "struct": {"overview":ov,"steps":st,"core":co,"full":int(ov and st and co)},
            "rouge_l": round(rl, 4),
        })
        print(f"  [{i+1:02d}/30] struct={int(ov and st and co)} rouge={rl:.4f}")

    rouges = np.array([r["rouge_l"] for r in results])
    structs = np.array([r["struct"]["full"] for r in results])
    summary = {
        "seed": seed,
        "struct_full": int(structs.sum()),
        "rouge_mean": round(float(rouges.mean()), 4),
        "rouge_std":  round(float(rouges.std(ddof=0)), 4),
    }
    print(f"  => struct={summary['struct_full']}/30, rouge={summary['rouge_mean']:.4f}+/-{summary['rouge_std']:.4f}")

    ts = datetime.now().strftime("%m%d_%H%M")
    out_path = f"{RESULTS_DIR}/multi_seed_seed{seed}_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    return summary

# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_summaries = []

    # Load existing seed=42 result
    import glob
    existing = glob.glob(f"{RESULTS_DIR}/two_turn_results_*.json")
    if existing:
        d = load_json(sorted(existing)[-1])
        rouges_42 = np.array([float(r["rouge_l"]) if not isinstance(r["rouge_l"], dict)
                               else 0.0 for r in d])
        def get_full(r):
            s = r.get("structure", r.get("struct", {}))
            if "full" in s: return s["full"]
            if "score" in s: return int(float(s["score"]) >= 1.0)
            return int(s.get("has_overview",False) and s.get("has_steps",False) and s.get("has_core",False))
        structs_42 = np.array([get_full(r) for r in d])
        s42 = {
            "seed": 42,
            "struct_full": int(structs_42.sum()),
            "rouge_mean": round(float(rouges_42.mean()), 4),
            "rouge_std":  round(float(rouges_42.std(ddof=0)), 4),
        }
        print(f"Loaded seed=42 existing result: struct={s42['struct_full']}/30, rouge={s42['rouge_mean']:.4f}")
        all_summaries.append(s42)
    else:
        print("WARNING: seed=42 result not found, skipping")

    for seed in SEEDS:
        ckpt, best_loss, elapsed = train_one(seed)
        summary = eval_one(ckpt, seed)
        summary["best_eval_loss"] = best_loss
        summary["elapsed_s"] = elapsed
        all_summaries.append(summary)

    # Aggregate
    print(f"\n{'='*55}")
    print("MULTI-SEED SUMMARY")
    print(f"{'='*55}")
    struct_vals = [s["struct_full"] for s in all_summaries]
    rouge_means = [s["rouge_mean"] for s in all_summaries]
    for s in all_summaries:
        print(f"  seed={s['seed']}: struct={s['struct_full']}/30, rouge={s['rouge_mean']:.4f}+/-{s['rouge_std']:.4f}")
    print(f"\nAcross {len(all_summaries)} seeds:")
    print(f"  Structure: mean={np.mean(struct_vals):.2f}/30, min={min(struct_vals)}, max={max(struct_vals)}")
    print(f"  ROUGE-L:   mean={np.mean(rouge_means):.4f}, std={np.std(rouge_means, ddof=0):.4f}")

    out = f"{RESULTS_DIR}/multi_seed_aggregate.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"seeds": all_summaries,
                   "aggregate": {
                       "struct_mean": round(np.mean(struct_vals), 3),
                       "struct_min": min(struct_vals),
                       "struct_max": max(struct_vals),
                       "rouge_mean": round(np.mean(rouge_means), 4),
                       "rouge_std":  round(np.std(rouge_means, ddof=0), 4),
                   }}, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out}")
