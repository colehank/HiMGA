# HiMGA：RAG基座使用指南

> 路径：`docs/usage/staging.md`
---

## 目录

1. [构建 RAG 需要哪些环节](#1-构建-rag-需要哪些环节)
2. [HiMGA 如何实现 RAG 基座](#2-himga-如何实现-rag-基座)
3. [用 HiMGA 跑一个完整评测](#3-用-himga-跑一个完整评测)
4. [评测指标详解](#4-评测指标详解)
5. [实现你自己的 RAG 记忆系统](#5-实现你自己的-rag-记忆系统)

---

## 1. 构建 RAG 需要哪些环节

一个完整的对话记忆 RAG 系统由四个环节组成：

| 环节 | 做什么 | 关键挑战 |
|------|-------|---------|
| **① 数据** | 原始对话记录，结构化为可处理的格式 | 两个数据集格式不一，日期表示各异 |
| **② 存入（Ingestion）** | 把对话逐条写入记忆库 | 如何组织、压缩、建索引 |
| **③ 检索（Retrieval）** | 根据问题找出最相关的记忆片段 | 语义相关性、时序关系、跨会话推理 |
| **④ 生成（Generation）** | 把问题和检索结果拼成 prompt，调用 LLM | prompt 设计、答案格式 |

HiMGA 的核心研究目标就在 **环节②③**：用层次化多图结构（Hierarchical Multi-Graph）实现更好的记忆存储和检索。

---

## 2. HiMGA 如何实现 RAG 基座

HiMGA 将四个环节分别封装为独立模块，各模块职责单一，可以单独替换。

```
himga.data    ──→  himga.memory  ──→  himga.llm
  （数据）          （存入 + 检索）      （生成）
     └──────────────────┴──────────────────┘
                        │
                   himga.agent
                  （流程编排器）
                        │
                   himga.eval
                  （结果评测）
```

### 2.1 `himga.data` — 对话数据

**对应环节①**

负责把两个基准数据集（LoCoMo、LongMemEval）解析为统一的数据结构，屏蔽原始格式差异。

```python
from himga.data import load_dataset, Sample, Session, Message, QAPair

samples = load_dataset("locomo", limit=5)  # 返回 list[Sample]

sample = samples[0]
print(sample.sample_id)           # 样本 ID
print(len(sample.sessions))       # 该样本共几轮对话 session
print(sample.sessions[0].messages[0].content)  # 第一条消息内容
print(sample.qa_pairs[0].question)             # 第一个评测问题
print(sample.qa_pairs[0].answer)               # 标准答案
```

**核心数据结构层级**：

```
Sample                         ← 一个完整评测单元（一组对话 + 所有问题）
 ├── sessions: list[Session]   ← 按时间分段的对话
 │    └── messages: list[Message]  ← 单条消息（role / content / date）
 └── qa_pairs: list[QAPair]    ← 需要回答的问题集
      ├── question              ← 问题文本
      ├── answer                ← 标准答案
      └── question_type         ← 问题类型（影响评测方式）
```

### 2.2 `himga.memory` — 记忆存储与检索

**对应环节②③，是 RAG 的核心**

`BaseMemory` 是所有记忆系统必须实现的抽象接口，只有三个方法：

```python
from himga.memory import BaseMemory, NullMemory
from himga.data.schema import Message, Session

class BaseMemory:
    def ingest(self, message: Message, session: Session) -> None:
        """把一条消息存入记忆库（环节②）"""

    def retrieve(self, query: str) -> str:
        """根据问题检索相关记忆，返回拼好的上下文字符串（环节③）"""

    def reset(self) -> None:
        """清空记忆，每个评测样本开始前调用"""
```

`NullMemory` 是内置的空实现——什么都不存，`retrieve` 永远返回 `""`。它的作用是：

- 作为**基线（baseline）**：纯 LLM 参数知识的上限/下限
- 在开发初期**打通评测流水线**，无需等记忆系统完成

### 2.3 `himga.llm` — LLM 客户端

**对应环节④**

统一的 LLM 调用接口，隔离具体 API 实现：

```python
from himga.llm import get_client, AnthropicClient, BaseLLMClient

llm = get_client()  # 从环境变量自动选择 provider

# 或者直接指定
llm = AnthropicClient(model="claude-sonnet-4-6")

# 统一的调用方式（OpenAI 消息格式）
response = llm.chat([
    {"role": "user", "content": "What is RAG?"}
])
```

扩展其他 LLM（如 OpenAI、本地模型）只需继承 `BaseLLMClient` 并实现 `chat()` 方法。

### 2.4 `himga.agent` — 流程编排

**串联四个环节的胶水层**

`BaseAgent` 组合了 `memory` 和 `llm`，实现了完整的 RAG 流程：

```python
from himga.agent import BaseAgent

agent = BaseAgent(memory=memory, llm=llm)

# 环节①②：把一个 Sample 的所有对话写入记忆
agent.ingest_sample(sample)

# 环节③④：检索 + 生成
answer = agent.answer("When did Alice mention her new job?")
```

`answer()` 内部的流程：

```
question
   │
   ▼
memory.retrieve(question)   ← 环节③：检索
   │
   ▼
[system prompt + context + question] → llm.chat()  ← 环节④：生成
   │
   ▼
answer string
```

### 2.5 `himga.eval` — 评测框架

评测框架驱动整个评测循环，不感知记忆系统的内部实现：

```python
from himga.eval import run_eval, compute_metrics
from himga.eval.judge import batch_judge

results      = run_eval(samples, agent)        # 跑预测
judge_scores = batch_judge(results, llm=llm)   # LLM 评分
metrics      = compute_metrics(results, judge_scores)  # 聚合指标
```

输出的 12 项指标按 `QuestionType` 分组，直接对应 MAGMA / LongMemEval 论文表格。

---

## 3. 用 HiMGA 跑一个完整评测

以下是用 `NullMemory`（基线）跑通完整流水线的最小示例。

### 3.1 环境准备

```bash
pip install himga            # 核心安装
# 若需要 BERTScore / SBERT：pip install "himga[eval]"
```

在项目根目录创建 `.env`（从 `.env_example` 复制）：

```
ANTHROPIC_API_KEY=sk-ant-...
DATASETS_ROOT=/path/to/your/datasets   # 可选，默认 .cache/datasets
```

### 3.2 完整代码

```python
from pathlib import Path

from himga.agent import BaseAgent
from himga.data import load_dataset
from himga.eval import compute_metrics, run_eval
from himga.eval.judge import batch_judge
from himga.llm import get_client
from himga.memory import NullMemory

# ── 1. 初始化组件 ───────────────────────────────────────────────
llm    = get_client()
memory = NullMemory()
agent  = BaseAgent(memory=memory, llm=llm)

# ── 2. 加载数据 ─────────────────────────────────────────────────
samples = load_dataset("locomo", limit=10)   # 先用 10 条快速验证

# ── 3. 跑预测 ───────────────────────────────────────────────────
results = run_eval(samples, agent, show_progress=True)
print(f"总计 {len(results)} 条 QA 预测完成")

# ── 4. LLM 评分（支持缓存，重跑不重复计费）─────────────────────
judge_scores = batch_judge(
    results,
    llm=llm,
    mode="auto",                                      # 按题型自动选择评分模式
    cache_path=Path("outputs/locomo_judge.json"),     # 结果缓存
)

# ── 5. 计算并打印指标 ────────────────────────────────────────────
metrics = compute_metrics(
    results,
    judge_scores,
    metrics=("judge_score", "exact_match", "f1", "rouge1"),  # 只算需要的
)

print(f"\n{'指标':<15} {'得分':>8}")
print("-" * 25)
for k, v in metrics["overall"].items():
    print(f"{k:<15} {v:>8.3f}")

print("\n── 按问题类型 ──")
for qtype, entry in sorted(metrics["by_type"].items()):
    print(f"{qtype:<30} judge={entry['judge_score']:.3f}  n={entry['count']}")
```

### 3.3 预期输出（NullMemory 基线，数值因模型而异）

```
总计 247 条 QA 预测完成

指标              得分
-------------------------
judge_score      0.087
exact_match      0.012
f1               0.065
rouge1           0.071

── 按问题类型 ──
adversarial                    judge=0.400  n=50
multi_hop                      judge=0.040  n=60
open_domain                    judge=0.120  n=37
single_hop                     judge=0.080  n=75
temporal                       judge=0.060  n=25
```

NullMemory 下分数很低是正常的——它不存任何记忆，LLM 只能靠参数知识猜。这正是基线的价值：**你的记忆系统要超过这个数才算有效果**。

---

## 4. 评测指标详解

HiMGA 内置 12 项指标，分为四类。每类侧重不同维度，综合使用才能全面衡量记忆系统的质量。

```python
from himga.eval import compute_metrics
from himga.eval.metrics import ALL_METRICS

print(ALL_METRICS)
# ('judge_score', 'exact_match', 'f1', 'rouge1', 'rouge2', 'rougeL',
#  'bleu1', 'bleu2', 'bleu4', 'meteor', 'bert_f1', 'sbert_similarity')
```

按需选择计算哪些指标（跳过重量级模型加速开发）：

```python
metrics = compute_metrics(results, judge_scores, metrics=("judge_score", "f1", "rouge1"))
```

---

### 4.1 LLM-as-a-Judge 类

#### `judge_score`

**本质**：让 LLM 扮演评卷老师，对模型答��打分。

**为什么需要它**：字面匹配指标（F1、ROUGE）只看词汇重叠，无法判断语义等价。"May 7th" 和 "7 May 2023" 字面不同但含义相同，只有 judge 能正确给满分。

**两种打分模式**（根据题目类型自动切换）：

| 数据集 | 题目类型 | 模式 | 返回值 |
|--------|---------|------|--------|
| LoCoMo | SINGLE_HOP / MULTI_HOP / TEMPORAL / OPEN_DOMAIN | 连续评分 | 0.0 – 1.0（六档量表） |
| LoCoMo | ADVERSARIAL | 规则匹配 | 0.0 或 1.0（无 LLM 调用） |
| LongMemEval | 各类型 | 二元判断 | 0.0（no）或 1.0（yes） |

**优点**：最接近人类判断，对格式变体、同义表达宽容。  
**缺点**：有 API 费用，但支持缓存（重跑不重复计费）。

---

### 4.2 基于词汇匹配的指标

这类指标不调用任何模型，速度极快，适合开发阶段快速迭代。

#### `exact_match`

**本质**：大小写不敏感的完全相等判断，返回 0.0 或 1.0。

```
预测 "7 May 2023"  vs 标准 "7 May 2023"  → 1.0
预测 "7 may 2023"  vs 标准 "7 May 2023"  → 1.0（大小写不敏感）
预测 "May 7, 2023" vs 标准 "7 May 2023"  → 0.0（格式不同）
```

**适用场景**：答案格式高度统一的封闭域问题（如数字、日期、专有名词）。  
**不适用**：开放域问题，稍有格式差异就得零分。

---

#### `f1`（Token-level F1）

**本质**：把预测和标准答案都拆成词（token），计算词袋级别的 F1 分数，对齐 SQuAD 评测标准。

```
预测 "Caroline went to the support group in May"
答案 "7 May 2023"

交集词：{"may"}  → precision = 1/8, recall = 1/3
F1 = 2 × (1/8) × (1/3) / (1/8 + 1/3) ≈ 0.22
```

**特点**：
- 标点替换为空格，全部小写后再分词
- 允许部分匹配，比 exact_match 更宽容
- 对词序不敏感（只看词汇重叠，不看顺序）

**适用场景**：答案中有关键实体词的问题，能奖励部分正确。

---

### 4.3 基于 N-gram 的指标

#### `rouge1` / `rouge2` / `rougeL`

**本质**：ROUGE（Recall-Oriented Understudy for Gisting Evaluation）系列，原本用于摘要评测，这里计算 F1（而非只计算召回率）。

| 指标 | 衡量什么 |
|------|---------|
| `rouge1` | 单词（unigram）重叠 |
| `rouge2` | 相邻两词（bigram）重叠，对连续短语更敏感 |
| `rougeL` | 最长公共子序列，同时考虑词序 |

```
预测 "The meeting was on Monday morning"
答案 "The meeting happened Monday"

rouge1: 词汇重叠多 → 较高
rouge2: "The meeting" 重叠 → 中等
rougeL: 最长公共子序列 "The meeting Monday" → 中等
```

**适用场景**：答案较长、有叙述性内容时，比 F1 更细腻。  
**注意**：`rougeL` 比 `rouge1` 更严格，更接近人类对流畅度的感知。

---

#### `bleu1` / `bleu2` / `bleu4`

**本质**：BLEU（Bilingual Evaluation Understudy）系列，原本用于机器翻译，衡量预测中有多大比例的 n-gram 出现在标准答案中（precision 导向），配合 SmoothingFunction 避免零分。

| 指标 | 衡量什么 | 特点 |
|------|---------|------|
| `bleu1` | 单词精准率 | 宽松 |
| `bleu2` | 1-gram + 2-gram 几何平均 | 中等 |
| `bleu4` | 1-gram 到 4-gram 几何平均 | 严格，短答案易得低分 |

**与 ROUGE 的区别**：ROUGE 关注召回（答案的词有没有出现在预测里），BLEU 关注精准（预测的词有没有出现在答案里）。两者结合更全面。

**适用场景**：对生成质量要求较高时参考，但对话记忆场景中 BLEU-4 通常偏低（答案短、词汇受限）。

---

#### `meteor`

**本质**：METEOR（Metric for Evaluation of Translation with Explicit ORdering），在 BLEU 基础上增加了词形还原（stemming）和同义词匹配（WordNet），同时兼顾精准率和召回率，并对词序有一定惩罚。

```
预测 "she attended the meeting"
答案 "she went to the meeting"

BLEU：无 bigram 重叠 → 低分
METEOR：WordNet 知道 "attended" ≈ "went" → 较高分
```

**优点**：比 BLEU 和 ROUGE 更接近人类判断，对同义词友好。  
**依赖**：首次运行自动下载 NLTK `wordnet` 语料（~10 MB）。

---

### 4.4 基于神经网络的语义指标

这两项需要额外安装 `pip install "himga[eval]"`，首次运行会下载预训练模型。

#### `bert_f1`（BERTScore F1）

**本质**：用 RoBERTa-large（1.3 GB）把预测和答案都编码成上下文向量，在词元级别计算余弦相似度，再聚合为 Precision / Recall / F1。

**优点**：能捕捉语义等价（"went" ≈ "traveled"），不依赖字面匹配。  
**已知问题**：两段不相关但都是流畅英文的文本，BERTScore 会虚高（通常在 0.75–0.90），需结合 judge_score 解读。典型案例：答案是短日期，预测是长段拒绝回答，bert_f1 仍可达 0.78。

**适用场景**：答案较长、有语义改写时比 ROUGE/BLEU 更可靠；但不适合单独作为判断标准。

---

#### `sbert_similarity`（Sentence-BERT 余弦相似度）

**本质**：用 `all-MiniLM-L6-v2`（80 MB）把预测和答案分别编码为**整句向量**，计算余弦相似度，范围 [-1.0, 1.0]（实际通常在 [0.0, 1.0]）。

**与 BERTScore 的区别**：

| | BERTScore | SBERT |
|--|-----------|-------|
| 粒度 | 词元级别（token-level） | 句子级别（sentence-level） |
| 适用 | 部分匹配、局部重叠 | 整体语义是否一致 |
| 虚高风险 | 较高 | 较低（对整句语义更严格） |

**适用场景**：判断整体语义方向是否正确，对"答非所问"的惩罚更明显。

---

### 4.5 指标选用建议

| 场景 | 推荐指标 |
|------|---------|
| 快速开发迭代（无 API 费用） | `f1` + `rouge1` + `meteor` |
| 对标论文结果 | `judge_score` + `f1` |
| 语义等价问题（同义词/格式变体） | `judge_score` + `meteor` + `sbert_similarity` |
| 需要完整评测报告 | 全部 12 项（`metrics=None`） |
| CI 自动化测试 | `exact_match` + `f1`（无外部依赖） |

**一条经验规则**：`judge_score` 是最接近论文结果的指标，其他指标用来辅助诊断——F1 低但 judge 高说明答案语义正确但措辞不同；F1 高但 judge 低说明词汇重叠是巧合。

---

## 5. 实现你自己的 RAG 记忆系统

只需要做一件事：**继承 `BaseMemory`，实现三个方法**。评测流水线完全不需要改动。

### 5.1 接口约定

```python
from himga.memory import BaseMemory
from himga.data.schema import Message, Session


class MyMemory(BaseMemory):

    def ingest(self, message: Message, session: Session) -> None:
        """
        每条消息都会调用一次。
        可用字段：
          message.role        — "user" 或 "assistant"
          message.content     — 消息文本
          message.date_str    — 原始日期字符串（可能为 None）
          session.session_id  — 所属 session ID
          session.date        — session 日期（datetime，可能为 None）
        """
        ...

    def retrieve(self, query: str) -> str:
        """
        根据 query 返回拼好的上下文字符串，直接注入 prompt。
        返回 "" 表示没有相关记忆。
        """
        ...

    def reset(self) -> None:
        """eval runner 在每个新 Sample 开始前调用，清空所有状态。"""
        ...
```

### 5.2 一个简单示例：全量拼接记忆

最简单的 RAG——把所有对话内容直接拼起来（不做检索，仅作示例）：

```python
from himga.memory import BaseMemory
from himga.data.schema import Message, Session


class ConcatMemory(BaseMemory):
    """把所有消息拼成一个长字符串，retrieve 时整体返回（无检索）。"""

    def __init__(self) -> None:
        self._buffer: list[str] = []

    def ingest(self, message: Message, session: Session) -> None:
        date = f"[{message.date_str}] " if message.date_str else ""
        self._buffer.append(f"{date}{message.role}: {message.content}")

    def retrieve(self, query: str) -> str:
        return "\n".join(self._buffer)

    def reset(self) -> None:
        self._buffer.clear()
```

### 5.3 接入评测，与基线对比

```python
from himga.agent import BaseAgent
from himga.data import load_dataset
from himga.eval import compute_metrics, run_eval
from himga.eval.judge import batch_judge
from himga.llm import get_client
from himga.memory import NullMemory

llm     = get_client()
samples = load_dataset("locomo", limit=10)

fast_metrics = ("judge_score", "f1", "rouge1")

# ── 基线：NullMemory ────────────────────────────────────────────
baseline_agent   = BaseAgent(memory=NullMemory(), llm=llm)
baseline_results = run_eval(samples, baseline_agent)
baseline_scores  = batch_judge(baseline_results, llm=llm)
baseline_metrics = compute_metrics(baseline_results, baseline_scores,
                                   metrics=fast_metrics)

# ── 你的方法：MyMemory ──────────────────────────────────────────
my_agent   = BaseAgent(memory=ConcatMemory(), llm=llm)
my_results = run_eval(samples, my_agent)
my_scores  = batch_judge(my_results, llm=llm)
my_metrics = compute_metrics(my_results, my_scores, metrics=fast_metrics)

# ── 对比 ────────────────────────────────────────────────────────
print(f"{'指标':<15} {'NullMemory':>12} {'MyMemory':>12} {'提升':>8}")
print("-" * 50)
for k in fast_metrics:
    b = baseline_metrics["overall"][k]
    m = my_metrics["overall"][k]
    print(f"{k:<15} {b:>12.3f} {m:>12.3f} {m - b:>+8.3f}")
```

### 5.4 下一步：实现真正的 RAG

实际的记忆系统在 `ingest` 和 `retrieve` 里实现真正的向量检索或图结构：

| 步骤 | `ingest` 里做什么 | `retrieve` 里做什么 |
|------|-----------------|-------------------|
| **向量检索** | 把消息 embed，存入向量数据库 | query embed 后做 top-k ANN 搜索，返回相关片段 |
| **图结构（MAGMA 风格）** | 建实体节点、会话节点、时序边 | 图遍历找相关节点，组装上下文 |
| **HiMGA（本研究）** | 建层次化多图（utterance → session → topic） | 分层检索：先找相关 topic，再下钻到具体 turn |

无论哪种方案，评测流水线（`run_eval` → `batch_judge` → `compute_metrics`）**一行代码不用改**。

---

## 小结

```
你需要做的：实现 BaseMemory 的三个方法
HiMGA 帮你做的：数据加载、Agent 编排、评测循环、12 项指标计算
```

| 你实现 | HiMGA 提供 |
|-------|-----------|
| `ingest()` — 存入策略 | `load_dataset()` — 统一数据格式 |
| `retrieve()` — 检索策略 | `run_eval()` — 评测循环 |
| `reset()` — 状态清理 | `batch_judge()` + `compute_metrics()` — 评分与指标 |
