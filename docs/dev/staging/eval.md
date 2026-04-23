# eval 模块设计

> 状态：✅ 已完成
> 路径：`src/himga/eval/`

---

## 职责

驱动完整评测循环，输出与 MAGMA Table 1（LoCoMo）和 Table 2（LongMemEval）对齐的分类指标。

---

## 文件结构

```
eval/
├── __init__.py    # 导出 run_eval, EvalResult, compute_metrics
├── runner.py      # 评测主循环
├── judge.py       # LLM-as-a-Judge 打分（多模式）
└── metrics.py     # 12 项指标聚合
```

---

## runner.py

```python
@dataclass
class EvalResult:
    sample_id:     str
    question_id:   str
    question_type: QuestionType
    question:      str
    ground_truth:  str
    prediction:    str

def run_eval(
    dataset: list[Sample],
    agent:   BaseAgent,
    *,
    show_progress: bool = True,
) -> list[EvalResult]:
    results = []
    for sample in tqdm(dataset, disable=not show_progress):
        agent.memory.reset()
        agent.ingest_sample(sample)
        for qa in sample.qa_pairs:
            prediction = agent.answer(qa.question)
            results.append(EvalResult(
                sample_id     = sample.sample_id,
                question_id   = qa.question_id,
                question_type = qa.question_type,
                question      = qa.question,
                ground_truth  = qa.answer,
                prediction    = prediction,
            ))
    return results
```

---

## judge.py

支持多种 judge 模式，对齐 LoCoMo（continuous + adversarial）和 LongMemEval（4 种 binary 变体）。

```python
def is_unanswerable(text: str) -> bool:
    """规则匹配"无法回答"，用于 ADVERSARIAL 类型，无需 LLM。"""

def judge_answer(
    question:     str,
    ground_truth: str,
    prediction:   str,
    *,
    llm:  BaseLLMClient | None = None,
    mode: str = "continuous",
) -> float:
    """返回 [0.0, 1.0]；mode="adversarial" 时无需 llm。"""

def batch_judge(
    results: list[EvalResult],
    *,
    llm:        BaseLLMClient,
    mode:       str = "auto",       # "auto" 按 question_type 自动路由
    cache_path: Path | None = None, # 避免重复计费
) -> list[float]:
    """批量 judge，支持结果缓存到本地 JSON。"""
```

**Judge 模式路由**

| mode | 适用 QuestionType | 返回值 |
|------|-----------------|--------|
| `continuous` | LoCoMo SINGLE_HOP / MULTI_HOP / TEMPORAL / OPEN_DOMAIN | 0.0–1.0 |
| `adversarial` | LoCoMo ADVERSARIAL | {0.0, 1.0}，规则匹配 |
| `temporal_reasoning` | LongMemEval TEMPORAL_REASONING | {0.0, 1.0}，允许 ±1 日期误差 |
| `knowledge_update` | LongMemEval KNOWLEDGE_UPDATE | {0.0, 1.0}，最新答案优先 |
| `preference` | LongMemEval SINGLE_SESSION_PREFERENCE | {0.0, 1.0}，rubric 部分匹配 |
| `default_binary` | LongMemEval 其余类型 | {0.0, 1.0} |

---

## metrics.py

12 项指标，按 `QuestionType` 分组聚合。重量级指标（`bert_f1`、`sbert_similarity`）需要 `pip install "himga[eval]"`，未安装时调用抛出带安装指引的 `ImportError`。

```python
ALL_METRICS: tuple[str, ...] = (
    "judge_score", "exact_match", "f1",
    "rouge1", "rouge2", "rougeL",
    "bleu1", "bleu2", "bleu4",
    "meteor", "bert_f1", "sbert_similarity",
)

def compute_metrics(
    results:      list[EvalResult],
    judge_scores: list[float],
    *,
    metrics: tuple[str, ...] | list[str] | None = None,  # None = 全部
) -> dict:
    """
    返回结构：
    {
      "overall": {"judge_score": 0.xx, "f1": 0.xx, "rouge1": 0.xx, ...},
      "by_type": {
        "single_hop":    {"judge_score": ..., "f1": ..., "count": ...},
        "temporal":      {...},
        ...
      }
    }
    """
```

---

## 关键设计决策

| 决策 | 理由 |
|------|------|
| judge 多模式 | LoCoMo 连续评分 vs LongMemEval 类型专用 binary，统一模式会降低对齐精度 |
| `mode="auto"` | 按 question_type 自动路由，调用方无需关心数据集差异 |
| `bert_f1` / `sbert_similarity` 为可选 extra | 模型文件大（~1.4 GB），不适合所有环境；未安装时抛 ImportError 而非静默返回 0.0 |
| `metrics` 参数 | 开发测试可跳过重量级模型，`@slow` 才触发完整计算 |
| judge 与 metrics 分离 | judge 耗时且有成本，metrics 是纯计算；分离后可离线重算指标 |
| judge 缓存以 question_id 为键 | 新增样本可增量更新，不需要失效全部旧缓存 |

---

## 测试策略

```
tests/eval/test_eval.py
```

- **普通测试**：mock LLM，不加载真实模型，`pytest tests/` 默认运行
- **`@pytest.mark.slow`**：加载 BERTScore / SBERT 模型，需 `pytest tests/ --run-slow`
- **`@pytest.mark.integration`**：调用真实 API，需 `pytest tests/ --run-integration`

快速测试使用 `metrics=("judge_score", "exact_match", "f1", "rouge1", "rouge2", "rougeL", "bleu1", "bleu2", "bleu4", "meteor")` 跳过重量级指标。

---

## 验收标准（已达成）

- [x] `run_eval` 使用 NullMemory + MockLLM 跑通 3 条样本
- [x] `judge_answer` 全模式（continuous / adversarial / 4 种 binary）正确返回
- [x] `is_unanswerable` 覆盖精确值和 14 个短语模式
- [x] 12 项指标计算正确，边界情况（空字符串、空 results）不抛异常
- [x] `batch_judge` 缓存机制：第二次调用不触发 LLM
- [x] `compute_metrics` `metrics` 参数：只返回请求的指标键
- [x] 未安装 eval extras 时调用 bert_f1 / sbert_similarity 抛出 ImportError
