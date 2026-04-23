# `himga.eval` — API 设计与使用说明

> 版本：基于当前实现（2026-04-22）
> 模块路径：`src/himga/eval/`

---

## 目录

1. [概述](#1-概述)
2. [模块结构](#2-模块结构)
3. [安装](#3-安装)
4. [类型参考](#4-类型参考)
5. [函数参考](#5-函数参考)
   - [run\_eval（runner.py）](#51-run_evalrunnerpy)
   - [is\_unanswerable（judge.py）](#52-is_unanswerablejudgepy)
   - [judge\_answer（judge.py）](#53-judge_answerjudgepy)
   - [batch\_judge（judge.py）](#54-batch_judgejudgepy)
   - [compute\_metrics（metrics.py）](#55-compute_metricsmetricspy)
6. [指标一览](#6-指标一览)
7. [评测流水线](#7-评测流水线)
8. [使用示例](#8-使用示例)
9. [指标对齐说明](#9-指标对齐说明)
10. [设计决策说明](#10-设计决策说明)

---

## 1. 概述

`himga.eval` 驱动完整评测循环，输出与 MAGMA Table 1（LoCoMo）和 Table 2（LongMemEval）对齐的分类指标。

**三层结构**

| 层 | 文件 | 职责 |
|----|------|------|
| **Runner** | `runner.py` | 驱动预测循环，收集 `EvalResult`；纯本地计算，不调用 judge |
| **Judge** | `judge.py` | LLM-as-a-Judge 评分，支持多种评分模式与结果缓存 |
| **Metrics** | `metrics.py` | 12 项指标聚合，按 `QuestionType` 分组输出 |

三层故意分离：runner 跑完后，judge 和 metrics 可离线反复调用，重跑指标不需要重新跑模型。

---

## 2. 模块结构

```
src/himga/eval/
├── __init__.py    # 导出 EvalResult, run_eval, compute_metrics
├── runner.py      # 评测主循环
├── judge.py       # LLM-as-a-Judge 打分
└── metrics.py     # 12 项指标聚合
```

**公共导入路径**

```python
from himga.eval import EvalResult, run_eval, compute_metrics
from himga.eval.judge import judge_answer, batch_judge, is_unanswerable
from himga.eval.metrics import token_f1, ALL_METRICS
```

---

## 3. 安装

**核心安装**（包含 exact_match、F1、ROUGE、BLEU、METEOR）：

```bash
pip install himga
# 或
uv add himga
```

**完整指标**（额外包含 BERTScore、Sentence-BERT）：

```bash
pip install "himga[eval]"
# 或
uv add "himga[eval]"
```

**GPU 加速**（BERTScore / SBERT 自动检测 CUDA，无需代码改动）：

```bash
# 先安装 CUDA 版 PyTorch，再安装 eval extras
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install "himga[eval]"
```

开发环境（`uv sync --dev`）默认安装所有依赖，包含完整 eval extras。

---

## 4. 类型参考

### `EvalResult`

```python
@dataclass
class EvalResult:
    sample_id:     str
    question_id:   str
    question_type: QuestionType
    question:      str
    ground_truth:  str
    prediction:    str
```

单个 QA pair 的预测记录，是 runner → judge → metrics 流水线的数据载体。

| 字段 | 类型 | 说明 |
|-----|------|------|
| `sample_id` | `str` | 来自 `Sample.sample_id`，用于追溯来源 |
| `question_id` | `str` | 来自 `QAPair.question_id`，作为 judge 缓存的键 |
| `question_type` | `QuestionType` | 问题类型，决定 judge 模式并用于 `by_type` 分组 |
| `question` | `str` | 问题文本，传给 judge prompt |
| `ground_truth` | `str` | 标准答案 |
| `prediction` | `str` | Agent 生成的答案 |

---

## 5. 函数参考

### 5.1 `run_eval`（runner.py）

```python
def run_eval(
    dataset: list[Sample],
    agent: BaseAgent,
    *,
    show_progress: bool = True,
) -> list[EvalResult]
```

驱动预测循环，每个 Sample 前 reset memory，然后依次回答所有 QA pair。

**注意**：`run_eval` 不调用 judge，不计算指标，只收集预测结果。

---

### 5.2 `is_unanswerable`（judge.py）

```python
def is_unanswerable(text: str) -> bool
```

规则匹配：判断 `text` 是否表达了"无法回答"的意思。用于 LoCoMo `ADVERSARIAL` 类型问题的 judge，无需 LLM 调用。

匹配 `""`, `"n/a"`, `"unanswerable"` 等精确值，以及 `"not mentioned"`, `"cannot answer"`, `"no information"` 等 14 个短语模式。

---

### 5.3 `judge_answer`（judge.py）

```python
def judge_answer(
    question: str,
    ground_truth: str,
    prediction: str,
    *,
    llm: BaseLLMClient | None = None,
    mode: str = "continuous",
) -> float
```

判断 `prediction` 是否正确，返回 `[0.0, 1.0]` 的分数。

**`mode` 参数**

| 模式 | 返回值 | 适用类型 | 说明 |
|------|--------|---------|------|
| `"continuous"` | `[0.0, 1.0]` | LoCoMo SINGLE_HOP / MULTI_HOP / TEMPORAL / OPEN_DOMAIN | LLM 返回 JSON `{"score": float, "reasoning": str}`，使用 0.0/0.2/0.4/0.6/0.8/1.0 六档评分量表 |
| `"adversarial"` | `0.0` 或 `1.0` | LoCoMo ADVERSARIAL | 规则匹配，**不调用 LLM**；`is_unanswerable(prediction)` 为 True 则得 1.0 |
| `"temporal_reasoning"` | `0.0` 或 `1.0` | LongMemEval TEMPORAL_REASONING | 允许 off-by-one 日期误差 |
| `"knowledge_update"` | `0.0` 或 `1.0` | LongMemEval KNOWLEDGE_UPDATE | 包含最新答案即正确，即使同时包含旧信息 |
| `"preference"` | `0.0` 或 `1.0` | LongMemEval SINGLE_SESSION_PREFERENCE | 按 rubric 判断，不要求完全覆盖 |
| `"default_binary"` | `0.0` 或 `1.0` | LongMemEval SINGLE_SESSION_USER / ASSISTANT / MULTI_SESSION | 通用二元判断 |

**注意**：`mode="adversarial"` 时 `llm` 可为 `None`；其他模式要求传入 `llm`，否则抛出 `ValueError`。

---

### 5.4 `batch_judge`（judge.py）

```python
def batch_judge(
    results: list[EvalResult],
    *,
    llm: BaseLLMClient,
    mode: str = "auto",
    cache_path: Path | None = None,
) -> list[float]
```

批量 judge，支持结果缓存到本地 JSON 文件。

**`mode="auto"`（推荐）**：根据每条结果的 `question_type` 自动选择对应的 judge 模式，无需手动指定。映射关系：

| QuestionType | 自动选择的 mode |
|-------------|---------------|
| SINGLE_HOP / MULTI_HOP / TEMPORAL / OPEN_DOMAIN | `continuous` |
| ADVERSARIAL | `adversarial` |
| TEMPORAL_REASONING | `temporal_reasoning` |
| KNOWLEDGE_UPDATE | `knowledge_update` |
| SINGLE_SESSION_PREFERENCE | `preference` |
| SINGLE_SESSION_USER / ASSISTANT / MULTI_SESSION | `default_binary` |

**缓存行为**：若 `cache_path` 指向已存在的文件且包含所有 `question_id`，直接从缓存读取，不调用 LLM；否则补充计算缺失项并更新文件。

---

### 5.5 `compute_metrics`（metrics.py）

```python
def compute_metrics(
    results: list[EvalResult],
    judge_scores: list[float],
    *,
    metrics: tuple[str, ...] | list[str] | None = None,
) -> dict
```

将预测结果和 judge 分数聚合为多维指标。

**`metrics` 参数**：控制计算哪些指标。`None`（默认）计算全部 12 项。传入子集可跳过重量级模型：

```python
# 跳过 BERTScore / SBERT，快速运行
out = compute_metrics(results, scores, metrics=("judge_score", "f1", "rouge1"))
```

**返回结构**

```python
{
    "overall": {
        "judge_score": float,
        "exact_match": float,
        "f1": float,
        "rouge1": float,
        "rouge2": float,
        "rougeL": float,
        "bleu1": float,
        "bleu2": float,
        "bleu4": float,
        "meteor": float,
        "bert_f1": float,          # 需要 himga[eval]
        "sbert_similarity": float, # 需要 himga[eval]
    },
    "by_type": {
        "single_hop": {
            "judge_score": float, ..., "count": int
        },
        # 仅包含 results 中实际出现的 QuestionType
    }
}
```

**空输入**：`results=[]` 时返回全零 `overall` 和空 `by_type`，不抛异常。

---

## 6. 指标一览

| 指标 | 依赖 | 范围 | 说明 |
|------|------|------|------|
| `judge_score` | LLM（外部） | [0.0, 1.0] | LLM-as-a-Judge 评分 |
| `exact_match` | 无 | {0.0, 1.0} | 大小写不敏感完全匹配 |
| `f1` | 无 | [0.0, 1.0] | Token-level F1（SQuAD 风格，标点替换为空格） |
| `rouge1` | `rouge-score`（核心依赖） | [0.0, 1.0] | ROUGE-1 F1 |
| `rouge2` | `rouge-score` | [0.0, 1.0] | ROUGE-2 F1 |
| `rougeL` | `rouge-score` | [0.0, 1.0] | ROUGE-L F1 |
| `bleu1` | `nltk`（核心依赖） | [0.0, 1.0] | BLEU-1，SmoothingFunction.method1 |
| `bleu2` | `nltk` | [0.0, 1.0] | BLEU-2 |
| `bleu4` | `nltk` | [0.0, 1.0] | BLEU-4 |
| `meteor` | `nltk` + wordnet 语料 | [0.0, 1.0] | METEOR |
| `bert_f1` | `bert-score`（**eval extra**） | [0.0, 1.0] | BERTScore F1，模型 `roberta-large`（~1.3 GB） |
| `sbert_similarity` | `sentence-transformers`（**eval extra**） | [-1.0, 1.0] | Sentence-BERT 余弦相似度，模型 `all-MiniLM-L6-v2`（~80 MB） |

`bert_f1` 和 `sbert_similarity` 未安装 eval extras 时调用会抛出 `ImportError`，并提示安装命令。

---

## 7. 评测流水线

```
load_dataset("locomo")
        │
        ▼
  list[Sample]
        │
        ▼  run_eval(dataset, agent)
        │    ├── agent.memory.reset()         ← 每个 sample 前
        │    ├── agent.ingest_sample(sample)
        │    └── agent.answer(qa.question)    ← 每个 QA pair
        │
        ▼
  list[EvalResult]
        │
        ├─────────────────────────────────────────────┐
        ▼                                             ▼
 batch_judge(results, llm, cache_path)       compute_metrics(results, [0.0]*n)
 (mode="auto"，按 question_type 路由)         (离线立即计算 F1/ROUGE/BLEU/METEOR)
        │
        ▼
  list[float]  (judge_scores)
        │
        ▼
 compute_metrics(results, judge_scores)
        │
        ▼
  {"overall": {...}, "by_type": {...}}
```

---

## 8. 使用示例

### 完整评测流水线（LoCoMo）

```python
from pathlib import Path
from himga.data import load_dataset
from himga.agent import BaseAgent
from himga.memory import NullMemory
from himga.llm import AnthropicClient
from himga.eval import run_eval, compute_metrics
from himga.eval.judge import batch_judge

agent_llm = AnthropicClient(model="claude-sonnet-4-6")
judge_llm = AnthropicClient(model="claude-haiku-4-5-20251001")
cache     = Path("outputs/locomo_judge_cache.json")

samples  = load_dataset("locomo")
agent    = BaseAgent(memory=NullMemory(), llm=agent_llm)
results  = run_eval(samples, agent=agent, show_progress=True)

judge_scores = batch_judge(results, llm=judge_llm, cache_path=cache)  # mode="auto"

metrics = compute_metrics(results, judge_scores)
print(f"Overall  judge={metrics['overall']['judge_score']:.3f}  "
      f"f1={metrics['overall']['f1']:.3f}  "
      f"rouge1={metrics['overall']['rouge1']:.3f}")
```

### 快速指标（跳过重量级模型）

```python
# 不需要 himga[eval]，无 LLM 调用
fast_metrics = ("exact_match", "f1", "rouge1", "rouge2", "rougeL", "bleu1", "meteor")
metrics = compute_metrics(results, [0.0] * len(results), metrics=fast_metrics)
print(f"F1={metrics['overall']['f1']:.3f}  ROUGE-1={metrics['overall']['rouge1']:.3f}")
```

### judge 缓存节省 API 费用

```python
cache = Path("outputs/lme_judge_cache.json")

# 第一次：调用 LLM，结果写入缓存
scores1 = batch_judge(results, llm=judge_llm, cache_path=cache)

# 第二次（如修改了聚合逻辑）：从缓存读取，不调用 LLM
scores2 = batch_judge(results, llm=judge_llm, cache_path=cache)
```

### 离线重算指标

```python
import json
from dataclasses import asdict
from himga.data.schema import QuestionType
from himga.eval import EvalResult

# 保存
with open("outputs/results.json", "w") as f:
    json.dump([asdict(r) for r in results], f, indent=2)

# 重建
with open("outputs/results.json") as f:
    raw = json.load(f)
results = [
    EvalResult(**{**r, "question_type": QuestionType(r["question_type"])})
    for r in raw
]
scores  = batch_judge(results, llm=judge_llm, cache_path=Path("outputs/cache.json"))
metrics = compute_metrics(results, scores)
```

---

## 9. 指标对齐说明

### Judge Score

| 数据集 | 问题类型 | Judge 模式 | 说明 |
|-------|---------|-----------|------|
| LoCoMo | SINGLE_HOP / MULTI_HOP / TEMPORAL / OPEN_DOMAIN | `continuous` | 0.0–1.0 连续评分，对齐 MAGMA rubric |
| LoCoMo | ADVERSARIAL | `adversarial` | 规则匹配，无 LLM 调用 |
| LongMemEval | TEMPORAL_REASONING | `temporal_reasoning` | 允许 ±1 日期误差 |
| LongMemEval | KNOWLEDGE_UPDATE | `knowledge_update` | 最新答案优先 |
| LongMemEval | SINGLE_SESSION_PREFERENCE | `preference` | Rubric 部分匹配 |
| LongMemEval | 其余类型 | `default_binary` | 通用二元判断 |

### 与 MAGMA Table 1 / Table 2 的对应关系

| MAGMA 列名 | HiMGA 对应字段 |
|-----------|--------------|
| Score (Judge) | `metrics["overall"]["judge_score"]` |
| Score (F1) | `metrics["overall"]["f1"]` |
| 各行（问题类型） | `metrics["by_type"][type_value]` |

---

## 10. 设计决策说明

### 为什么 judge 支持多模式而非统一二元？

LoCoMo 和 LongMemEval 的评测协议不同：LoCoMo 用 0–1 连续分衡量语义相似度，LongMemEval 用二元 yes/no 并针对不同题型有专用 prompt（如允许 off-by-one 的 temporal_reasoning、优先最新答案的 knowledge_update）。统一模式会降低对齐精度，`mode="auto"` 让调用方无需关心细节。

### 为什么 `bert_f1` 和 `sbert_similarity` 是可选 extra？

两者分别需要 ~1.3 GB 和 ~80 MB 的预训练模型，下载时间长、不适合所有环境。将其设为 `[eval]` extras 使核心安装保持轻量。未安装时调用会立即抛出带安装指引的 `ImportError`，而非静默返回 0.0（避免误解结果）。

### 为什么 `compute_metrics` 有 `metrics` 参数？

开发和测试阶段通常不需要 BERTScore / SBERT，但默认加载会拖慢测试速度。`metrics` 参数允许按需计算，`@pytest.mark.slow` 测试才触发重量级模型。

### 为什么 runner 不内嵌 judge？

Judge 有网络延迟、API 费用、结果可缓存复用。分离后修改指标聚合逻辑不需要重跑 judge；重跑 runner 也不需要重新缓存 judge 结果。

### 为什么 judge 缓存以 `question_id` 为键？

`question_id` 在同一数据集内唯一，新增样本时可增量更新缓存，不需要失效全部旧缓存。
