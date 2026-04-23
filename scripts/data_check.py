# %%
from pathlib import Path

from himga.agent import BaseAgent
from himga.data import load_dataset
from himga.eval import compute_metrics, run_eval
from himga.eval.judge import batch_judge
from himga.llm import get_client
from himga.memory import NullMemory

locomo_samples = load_dataset("locomo")
lme_samples = load_dataset("longmemeval", limit=10)

print(f"LoCoMo: {len(locomo_samples)} samples")
print(f"LongMemEval: {len(lme_samples)} samples")
# %%
sample_idx = 1

for ses_idx, ses in enumerate(locomo_samples[sample_idx].sessions):
    date = ses.date
    print(f"Session {ses_idx} date: {date}")
    for i, msg in enumerate(ses.messages):
        print(f"{msg.role}: {msg.content}")
    print("💩" * 40)

# %%
for qa in locomo_samples[sample_idx].qa_pairs:
    print(f"QAPair {qa.question_id}: type={qa.question_type}")
    print(f"Question: {qa.question}")
    print(f"Evidence: {qa.evidence}")
    print(f"Answer: {qa.answer}")
    print("💩" * 40)
# %%
# 1. 初始化组件
llm = get_client()
agent = BaseAgent(memory=NullMemory(), llm=llm)
# %%
# 2. 加载数据
samples = load_dataset("locomo", limit=1)
# %%
# 3. 跑预测 + LLM 评分（支持缓存，重跑不重复计费）
results = run_eval(samples, agent, show_progress=True)
judge_scores = batch_judge(
    results,
    llm=llm,
    mode="auto",
    cache_path=Path("outputs/locomo_judge.json"),
)
# %%
# 4. 计算并打印指标
metrics = compute_metrics(
    results,
    judge_scores,
    metrics=("judge_score", "exact_match", "f1", "rouge1"),
)

# %%
