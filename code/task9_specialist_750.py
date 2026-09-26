"""
Task9: 数据量对照。specialist各层重复采样250条到750条，与multilevel(天然750条)公平对比数据量。
不改动 train_multilevel_v2.py，复用其函数，只改数据构造和输出目录。
"""
import sys, json, random, hashlib
sys.path.insert(0, "/root/autodl-tmp")
import train_multilevel_v2 as base

TARGET_N = 750
SEEDS = [42, 123, 456]

def main():
    with open(f"{base.DATA_DIR}/{base.TRAIN_FILE}", encoding="utf-8") as f:
        raw_data = json.load(f)

    import os
    if os.path.exists(base.REGENERATED_EXPERT):
        with open(base.REGENERATED_EXPERT, encoding="utf-8") as f:
            regen = json.load(f)
        regen_map = {r["raw_code"]: r["new_expert_output"] for r in regen if not r["issues"]}
        raw_data = base.apply_regenerated_expert(raw_data, regen_map)

    train_items, val_items = base.split_by_code_id(raw_data, val_ratio=0.15, seed=42)

    for level in ["beginner", "intermediate", "expert"]:
        base_samples = base.build_samples(train_items, level)
        n0 = len(base_samples)
        print(f"\n[{level}] base_n={n0}, target={TARGET_N}")

        for seed in SEEDS:
            out_dir = f"{base.BASE_DIR}/models/specialist_{level}_750/seed_{seed}"
            if os.path.exists(f"{out_dir}/adapter_model.safetensors"):
                print(f"  SKIP {level}_750/seed_{seed} (already trained)")
                continue

            random.seed(seed)
            reps = TARGET_N // n0
            remainder = TARGET_N % n0
            samples = base_samples * reps + random.sample(base_samples, remainder)
            random.shuffle(samples)
            assert len(samples) == TARGET_N, f"expected {TARGET_N}, got {len(samples)}"

            base.train(f"specialist_{level}_750", samples, out_dir, seed)

    print("\nALL DONE task9")

if __name__ == "__main__":
    main()
