# %%
from himga.data import load_dataset

locomo_samples = load_dataset("locomo")
lme_samples = load_dataset("longmemeval", limit=10)

print(f"LoCoMo: {len(locomo_samples)} samples")
print(f"LongMemEval: {len(lme_samples)} samples")
# %%
