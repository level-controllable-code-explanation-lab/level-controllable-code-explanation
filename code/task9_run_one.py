"""跑单个(level, seed)组合，重复采样到750条。用法: python3 task9_run_one.py <level> <seed>"""
import sys, json, random, os
sys.path.insert(0, "/root/autodl-tmp")
import train_multilevel_v2 as base

level = sys.argv[1]
seed = int(sys.argv[2])
TARGET_N = 750

with open(f"{base.DATA_DIR}/{base.TRAIN_FILE}", encoding="utf-8") as f:
    raw_data = json.load(f)

if os.path.exists(base.REGENERATED_EXPERT):
    with open(base.REGENERATED_EXPERT, encoding="utf-8") as f:
        regen = json.load(f)
    regen_map = {r["raw_code"]: r["new_expert_output"] for r in regen if not r["issues"]}
    raw_data = base.apply_regenerated_expert(raw_data, regen_map)

train_items, val_items = base.split_by_code_id(raw_data, val_ratio=0.15, seed=42)
base_samples = base.build_samples(train_items, level)
n0 = len(base_samples)
print(f"[{level}] base_n={n0}, target={TARGET_N}, seed={seed}")

out_dir = f"{base.BASE_DIR}/models/specialist_{level}_750/seed_{seed}"
if os.path.exists(f"{out_dir}/adapter_model.safetensors"):
    print(f"SKIP {level}_750/seed_{seed} (already trained)")
    sys.exit(0)

random.seed(seed)
reps = TARGET_N // n0
remainder = TARGET_N % n0
samples = base_samples * reps + random.sample(base_samples, remainder)
random.shuffle(samples)
assert len(samples) == TARGET_N

base.train(f"specialist_{level}_750", samples, out_dir, seed)
print("DONE", level, seed)
