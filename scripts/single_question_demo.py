"""
单题示例：分别从 LoCoMo 和 LongMemEval 各取一道题，用 NullMemory 基线跑通完整流水线。

Token 消耗估算（NullMemory，实测数据）
--------------------------------------
LoCoMo  (continuous judge):  ~483 tokens / 题
LongMemEval (binary judge):  ~322 tokens / 题
两题合计:                     ~805 tokens ≈ $0.003（claude-sonnet-4-6 定价）

运行前提
--------
1. 安装依赖：uv sync --dev  （或 pip install himga）
2. 配置 .env：
       ANTHROPIC_API_KEY=sk-ant-...
       DATASETS_ROOT=/path/to/datasets   # 含 locomo/ 和 longmemeval/ 子目录

运行方式
--------
    uv run python scripts/single_question_demo.py
"""

from __future__ import annotations

from pathlib import Path

from himga.agent import BaseAgent
from himga.data import load_dataset
from himga.eval import compute_metrics
from himga.eval.judge import batch_judge
from himga.llm import get_client
from himga.memory import NullMemory

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def _print_section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def run_single_question(dataset_name: str, llm, *, question_index: int = 0) -> None:
    """加载数据集第一个样本，回答其中一道题，打印完整结果。"""

    _print_section(f"数据集：{dataset_name}")

    # ── 加载数据（只取 1 个样本）────────────────────────────────────
    samples = load_dataset(dataset_name, limit=1)
    sample = samples[0]

    n_sessions = len(sample.sessions)
    n_messages = sum(len(s.messages) for s in sample.sessions)
    n_chars = sum(len(m.content) for s in sample.sessions for m in s.messages)
    n_qa = len(sample.qa_pairs)

    print(f"\n样本 ID    : {sample.sample_id}")
    print(f"对话 sessions : {n_sessions}  |  消息总数 : {n_messages}")
    print(f"对话字符数 : {n_chars:,}  ≈  {n_chars // 4:,} tokens（估算）")
    print(f"QA 题目总数: {n_qa}")

    # ── 选一道题 ─────────────────────────────────────────────────────
    qa = sample.qa_pairs[question_index]
    print(f"\n选取题目（index={question_index}）")
    print(f"  类型    : {qa.question_type.value}")
    print(f"  问题    : {qa.question}")
    print(f"  标准答案: {qa.answer}")

    # ── 只对这一道题构造单条 EvalResult，避免跑全部 QA ──────────────
    # 注意：ingest_sample 仍然写入完整对话（NullMemory 下无实际开销）
    agent = BaseAgent(memory=NullMemory(), llm=llm)
    agent.ingest_sample(sample)
    prediction = agent.answer(qa.question)
    print(f"\n  模型预测: {prediction}")

    # ── 手动构造 EvalResult，调用 judge ──────────────────────────────
    from himga.eval.runner import EvalResult

    result = EvalResult(
        sample_id=sample.sample_id,
        question_id=qa.question_id,
        question_type=qa.question_type,
        question=qa.question,
        ground_truth=qa.answer,
        prediction=prediction,
    )

    judge_scores = batch_judge(
        [result],
        llm=llm,
        mode="auto",
        cache_path=OUTPUT_DIR / f"{dataset_name}_demo_judge.json",
    )

    metrics = compute_metrics(
        [result],
        judge_scores,
        metrics=("judge_score", "exact_match", "f1", "rouge1"),
    )

    print(f"\n  judge_score  : {metrics['overall']['judge_score']:.3f}")
    print(f"  exact_match  : {metrics['overall']['exact_match']:.3f}")
    print(f"  f1           : {metrics['overall']['f1']:.3f}")
    print(f"  rouge1       : {metrics['overall']['rouge1']:.3f}")


def main() -> None:
    print("=" * 60)
    print("  HiMGA 单题示例（NullMemory 基线）")
    print("=" * 60)

    llm = get_client()

    run_single_question("locomo", llm, question_index=0)
    run_single_question("longmemeval", llm, question_index=0)

    print(f"\n{'─' * 60}")
    print("  完成。judge 缓存已写入 outputs/")
    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    main()
