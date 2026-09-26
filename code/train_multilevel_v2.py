"""
Multi-Level Training Script V2 - FIXED DATA SPLITTING
======================================================
FIX: Split by unique raw_code first, THEN assign levels.
Ensures same code snippet's 3 levels stay together.
"""
import json, random, os, gc, time, argparse, hashlib
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ===== FIXED CONFIG =====
BASE_DIR = "/root/autodl-tmp/multilevel_exp"
DATA_DIR = "/root/autodl-tmp/vclite_v2/data"
MODEL_PATH = "/root/autodl-tmp/gemma-2-9b-it"
TRAIN_FILE = "code_explanation_cn_multilevel.json"
REGENERATED_EXPERT = f"{BASE_DIR}/expert_regenerated.json"

SEEDS = [42, 123, 456]
MAX_LENGTH = 768
BATCH_SIZE = 1
GRAD_ACCUM = 8
LR = 5e-5
EPOCHS = 2
LORA_R = 8
LORA_ALPHA = 16
LORA_TARGETS = ["q_proj", "v_proj", "k_proj", "o_proj"]

LEVEL_INSTRUCTIONS = {
    "beginner": "你是一位面向编程初学者的教师。请用通俗易懂的语言解释以下代码，避免使用专业术语。\n\n代码：\n```python\n{code}\n```\n\n请给出初学者友好的解释：",
    "intermediate": "你是一位编程教育者。请用标准编程术语解释以下代码。\n\n代码：\n```python\n{code}\n```\n\n请给出解释：",
    "expert": "你是一位资深软件工程师。请用最精炼的技术语言解释以下代码，突出算法设计和复杂度分析。\n\n代码：\n```python\n{code}\n```\n\n请给出专家级解释：",
}

os.makedirs(BASE_DIR, exist_ok=True)


# ===== FIXED DATA SPLITTING =====
def split_by_code_id(data, val_ratio=0.15, seed=42):
    """
    FIX: Group by unique raw_code first, then split.
    All 3 levels of the same code stay in the same split.
    Returns: train_items, val_items (lists of items)
    """
    # Step 1: Group items by raw_code
    by_code = {}
    for item in data:
        code = item.get("raw_code", "")
        if not code:
            continue
        code_hash = hashlib.md5(code.encode()).hexdigest()
        if code_hash not in by_code:
            by_code[code_hash] = []
        by_code[code_hash].append(item)

    unique_codes = list(by_code.keys())
    print(f"  Unique code snippets: {len(unique_codes)}")
    print(f"  Total entries: {len(data)}")
    print(f"  Avg entries per code: {len(data)/len(unique_codes):.1f}")

    # Step 2: Shuffle codes with fixed seed
    random.seed(seed)
    random.shuffle(unique_codes)

    # Step 3: Split codes
    split_idx = int(len(unique_codes) * (1 - val_ratio))
    train_codes = set(unique_codes[:split_idx])
    val_codes = set(unique_codes[split_idx:])

    # Step 4: Assign items by their code
    train_items = []
    val_items = []
    for code_hash, items in by_code.items():
        if code_hash in train_codes:
            train_items.extend(items)
        else:
            val_items.extend(items)

    print(f"  Train: {len(train_items)} items ({len(train_codes)} unique codes)")
    print(f"  Val: {len(val_items)} items ({len(val_codes)} unique codes)")

    # Verify no leakage
    train_code_set = set(hashlib.md5(i["raw_code"].encode()).hexdigest() for i in train_items if i.get("raw_code"))
    val_code_set = set(hashlib.md5(i["raw_code"].encode()).hexdigest() for i in val_items if i.get("raw_code"))
    overlap = train_code_set & val_code_set
    assert len(overlap) == 0, f"DATA LEAK: {len(overlap)} codes in both train and val!"
    print(f"  Overlap check: OK (0 codes shared)")

    return train_items, val_items


def apply_regenerated_expert(data, regenerated_map):
    """Replace expert outputs with regenerated versions."""
    updated = 0
    for item in data:
        if item.get("meta", {}).get("level") == "expert":
            code = item.get("raw_code", "")
            if code in regenerated_map:
                item["output"] = regenerated_map[code]
                updated += 1
    print(f"  Applied regenerated expert: {updated}/{sum(1 for i in data if i.get('meta',{}).get('level')=='expert')}")
    return data


# ===== TRAINING DATA CONSTRUCTION =====
def build_samples(items, level):
    samples = []
    for item in items:
        code = item.get("raw_code", item.get("code", ""))
        output = item.get("output", "")
        if not code or not output:
            continue
        # Only include items matching the target level
        item_level = item.get("meta", {}).get("level", "")
        if item_level != level:
            continue
        instruction = LEVEL_INSTRUCTIONS[level].format(code=code)
        samples.append({"instruction": instruction, "output": output})
    return samples


def build_multilevel_samples(items):
    """Include ALL levels for the multi-level model."""
    samples = []
    for item in items:
        code = item.get("raw_code", item.get("code", ""))
        output = item.get("output", "")
        if not code or not output:
            continue
        lv = item.get("meta", {}).get("level", "")
        if lv not in LEVEL_INSTRUCTIONS:
            continue
        instruction = LEVEL_INSTRUCTIONS[lv].format(code=code)
        samples.append({"instruction": instruction, "output": output})
    random.seed(42)
    random.shuffle(samples)
    return samples


# ===== MODEL =====
def load_model():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, attn_implementation="eager", torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
                       target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, tokenizer


class CustomDataCollator:
    def __init__(self, tokenizer, max_length=MAX_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __call__(self, batch):
        instructions = [i["instruction"] for i in batch]
        outputs = [i["output"] for i in batch]
        full_texts = [inst + "\n\n" + out + self.tokenizer.eos_token for inst, out in zip(instructions, outputs)]
        tok = self.tokenizer(full_texts, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt")
        labels = tok["input_ids"].clone()
        for idx, inst in enumerate(instructions):
            ilen = len(self.tokenizer(inst + "\n\n", truncation=True, max_length=self.max_length, return_tensors="pt")["input_ids"][0])
            labels[idx, :min(ilen, self.max_length)] = -100
        tok["labels"] = labels
        return tok


def train(name, samples, out_dir, seed):
    print(f"\n{'='*60}\nTraining: {name} (seed={seed}, n={len(samples)})\n{'='*60}")
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    random.seed(seed)
    with open(f"{out_dir}/metadata.json", "w", encoding="utf-8") as f:
        json.dump({"name": name, "seed": seed, "n": len(samples),
                    "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lr": LR, "epochs": EPOCHS}, f, ensure_ascii=False, indent=2)
    model, tokenizer = load_model()
    ds = Dataset.from_list(samples)
    args = TrainingArguments(output_dir=out_dir, num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR, warmup_ratio=0.03, weight_decay=0.01,
        logging_steps=10, save_strategy="epoch", bf16=True, report_to="none", remove_unused_columns=False)
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=CustomDataCollator(tokenizer))
    t0 = time.time()
    trainer.train()
    print(f"  Done in {(time.time()-t0)/60:.1f} min")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["specialist_beginner", "specialist_intermediate", "specialist_expert", "multilevel", "all"])
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("MULTI-LEVEL TRAINING V2 (Fixed Data Splitting)")
    print("=" * 60)

    # Load data
    with open(f"{DATA_DIR}/{TRAIN_FILE}", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Apply regenerated expert if available
    if os.path.exists(REGENERATED_EXPERT):
        print("\nApplying regenerated expert explanations...")
        with open(REGENERATED_EXPERT, encoding="utf-8") as f:
            regen = json.load(f)
        regen_map = {r["raw_code"]: r["new_expert_output"] for r in regen if not r["issues"]}
        raw_data = apply_regenerated_expert(raw_data, regen_map)

    # Split by code ID (THE FIX)
    print(f"\nSplitting {len(raw_data)} entries by unique code...")
    train_items, val_items = split_by_code_id(raw_data, val_ratio=0.15, seed=42)

    # Build samples
    b_samples = build_samples(train_items, "beginner")
    i_samples = build_samples(train_items, "intermediate")
    e_samples = build_samples(train_items, "expert")
    m_samples = build_multilevel_samples(train_items)

    print(f"\nTraining samples:")
    print(f"  Specialist Beginner: {len(b_samples)}")
    print(f"  Specialist Intermediate: {len(i_samples)}")
    print(f"  Specialist Expert: {len(e_samples)}")
    print(f"  Multi-Level (Ours): {len(m_samples)}")
    print(f"  (All share same {len(set(hashlib.md5(i['raw_code'].encode()).hexdigest() for i in train_items if i.get('raw_code')))} unique code snippets)")

    # Experiments
    exp_map = {
        "specialist_beginner": b_samples,
        "specialist_intermediate": i_samples,
        "specialist_expert": e_samples,
        "multilevel": m_samples,
    }

    exps_to_run = list(exp_map.items()) if args.experiment == "all" else [(args.experiment, exp_map[args.experiment])]
    seeds = [args.seed] if args.seed else SEEDS

    for exp_name, samples in exps_to_run:
        for seed in seeds:
            out_dir = f"{BASE_DIR}/models/{exp_name}/seed_{seed}"
            if os.path.exists(f"{out_dir}/adapter_model.safetensors"):
                print(f"\n  SKIP {exp_name}/seed_{seed} (already trained)")
                continue
            train(exp_name, samples, out_dir, seed)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
