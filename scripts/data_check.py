# %%
# Imports
import json
import pickle as pkl
from pathlib import Path

from himga.agent import BaseAgent
from himga.data import load_dataset
from himga.eval import compute_metrics, run_eval
from himga.eval.judge import batch_judge
from himga.llm import get_client
from himga.memory import NullMemory

# %%
# Load datasets
locomo_samples = load_dataset("locomo")
lme_samples = load_dataset("longmemeval", limit=10)

print(f"LoCoMo: {len(locomo_samples)} samples")
print(f"LongMemEval: {len(lme_samples)} samples")

# %%
# Inspect sessions of a single LoCoMo sample
sample_idx = 1

for ses_idx, ses in enumerate(locomo_samples[sample_idx].sessions):
    print(f"Session {ses_idx} date: {ses.date}")
    for msg in ses.messages:
        print(f"{msg.role}: {msg.content}")
    print("💩" * 40)

# %%
# Inspect QA pairs of the same sample
for qa in locomo_samples[sample_idx].qa_pairs:
    print(f"QAPair {qa.question_id}: type={qa.question_type}")
    print(f"Question: {qa.question}")
    print(f"Evidence: {qa.evidence}")
    print(f"Answer: {qa.answer}")
    print("💩" * 40)

# %%
# Smoke-test NullMemory ingest & retrieve
mem = NullMemory()
mem.ingest(
    session=locomo_samples[sample_idx].sessions[0],
    message=locomo_samples[sample_idx].sessions[0].messages[0],
)
ans = mem.retrieve(query="What did the user say?")
print(f"Retrieved answer: {ans}")

# %%
# Build agent
llm = get_client(provider="openai", base_url="https://www.dmxapi.cn/v1", batch_size=5)
agent = BaseAgent(memory=mem, llm=llm)
agent.answer(question=locomo_samples[sample_idx].qa_pairs[0].question)

# %%
# Run evaluation (cached)
_cache = Path("outputs/locomo_results.pkl")
if _cache.exists():
    with open(_cache, "rb") as f:
        results = pkl.load(f)
else:
    results = run_eval([locomo_samples[0]], agent, show_progress=True)
    with open(_cache, "wb") as f:
        pkl.dump(results, f)

# %%
# LLM-based judging (cached)
llm_scores = batch_judge(
    results,
    llm=llm,
    cache_path=Path("outputs/locomo_judge.json"),
)

# %%
# Compute & save metrics
metrics = compute_metrics(
    results,
    judge_scores=llm_scores,
    metrics=(
        "judge_score",
        "exact_match",
        "f1",
        "rouge1",
        "rouge2",
        "rougeL",
        "bleu1",
        "bleu2",
        "bleu4",
        "meteor",
        "bert_f1",
        "sbert_similarity",
    ),
)
with open("outputs/locomo_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)


# %%
# Report metrics as DataFrame
def report_metrics(metrics):
    import pandas as pd

    rows = {"overall": metrics["overall"]}
    rows.update(metrics["by_type"])
    df = pd.DataFrame(rows).T
    return df


report_metrics(metrics)

# %%
