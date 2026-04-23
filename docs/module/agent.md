# `himga.agent` — API 设计与使用说明

> 版本：基于当前实现（2026-04-22）
> 模块路径：`src/himga/agent/`

---

## 目录

1. [概述](#1-概述)
2. [模块结构](#2-模块结构)
3. [类型参考](#3-类型参考)
   - [BaseAgent](#31-baseagent)
4. [使用示例](#4-使用示例)
   - [基础用法](#41-基础用法)
   - [在 eval 循环中使用](#42-在-eval-循环中使用)
   - [自定义 prompt 结构（子类覆写）](#43-自定义-prompt-结构子类覆写)
   - [替换记忆系统](#44-替换记忆系统)
5. [依赖关系图](#5-依赖关系图)
6. [设计决策说明](#6-设计决策说明)

---

## 1. 概述

`himga.agent` 将 `BaseMemory` 和 `BaseLLMClient` 组合为可评测的 Agent：负责按时序将对话历史注入记忆系统，并在给定问题时检索记忆、构造 prompt、调用 LLM 生成答案。

**职责边界**

| 职责 | 由谁负责 |
|------|---------|
| 解析数据集 | `himga.data` |
| 存储与检索记忆 | `himga.memory`（具体实现） |
| 调用 LLM API | `himga.llm`（具体实现） |
| 组合以上三者，提供 ingest / answer 接口 | `himga.agent.BaseAgent` |
| 驱动评测循环、收集结果 | `himga.eval.runner.run_eval` |

---

## 2. 模块结构

```
src/himga/agent/
├── __init__.py    # 导出 BaseAgent
└── base.py        # BaseAgent 实现
```

**公共导入路径**

```python
from himga.agent import BaseAgent
```

---

## 3. 类型参考

### 3.1 `BaseAgent`

```python
class BaseAgent
```

评测 Agent 的基础实现，可直接使用，也可通过子类覆写 `_build_messages` 定制 prompt。

#### 构造函数

```python
def __init__(self, memory: BaseMemory, llm: BaseLLMClient) -> None
```

| 参数 | 类型 | 说明 |
|-----|------|------|
| `memory` | `BaseMemory` | 记忆系统实例，生命周期由外部（通常是 `run_eval`）管理 |
| `llm` | `BaseLLMClient` | LLM 客户端实例 |

**属性**

| 属性 | 类型 | 说明 |
|-----|------|------|
| `agent.memory` | `BaseMemory` | 记忆系统引用，`eval` runner 通过 `agent.memory.reset()` 进行样本间隔离 |
| `agent.llm` | `BaseLLMClient` | LLM 客户端引用 |

---

#### 方法


##### `ingest_sample(sample)`

```python
def ingest_sample(self, sample: Sample) -> None
```

将 `sample` 的全部对话历史按时序注入记忆系统。

**参数**

| 参数 | 类型 | 说明 |
|-----|------|------|
| `sample` | `Sample` | 评测单元，其 `sessions` 字段已在 loader 阶段按时间升序排列 |

**行为**

对 `sample.sessions` 中的每个 session，按 `session.messages` 顺序逐条调用 `self.memory.ingest(message, session)`。sessions 的顺序即为 `Sample.sessions` 的原有顺序（loader 保证时序正确）。

**调用时机**：`eval.runner.run_eval` 在每个 Sample 的 `memory.reset()` 之后调用此方法。

---

##### `answer(question)`

```python
def answer(self, question: str) -> str
```

检索记忆并生成对 `question` 的回答。

**参数**

| 参数 | 类型 | 说明 |
|-----|------|------|
| `question` | `str` | 自然语言问题，通常来自 `QAPair.question` |

**返回**

`str`：LLM 生成的答案文本。

**执行流程**

```
question
  → memory.retrieve(question)      → context: str
  → _build_messages(question, context)  → messages: list[dict]
  → llm.chat(messages)             → answer: str
```

---

##### `_build_messages(question, context)`

```python
def _build_messages(self, question: str, context: str) -> list[dict]
```

构造传给 LLM 的消息列表（OpenAI 格式）。

**参数**

| 参数 | 类型 | 说明 |
|-----|------|------|
| `question` | `str` | 要回答的问题 |
| `context` | `str` | `memory.retrieve` 的返回值；空字符串表示无可用记忆 |

**返回**

`list[dict]`：包含 system 消息和 user 消息的列表。

**两种 prompt 结构**

*有 context 时（context 非空字符串）：*

```
system: "You are a helpful assistant with access to past conversation history.
         Answer the question based on the provided context."

user:   "Context:
         {context}

         Question: {question}"
```

*无 context 时（context 为空字符串）：*

```
system: "You are a helpful assistant with access to past conversation history.
         Answer the question based on the provided context."

user:   "{question}"
```

**子类覆写**

子类可覆写此方法以实现不同的 prompt 结构（如多轮对话、Chain-of-Thought、结构化输出等）：

```python
class HiMGAAgent(BaseAgent):
    def _build_messages(self, question: str, context: str) -> list[dict]:
        # 自定义实现
        ...
```

---

## 4. 使用示例

### 4.1 基础用法

```python
from himga.agent import BaseAgent
from himga.memory import NullMemory
from himga.llm import get_client
from himga.data import load_dataset

llm    = get_client()
memory = NullMemory()
agent  = BaseAgent(memory=memory, llm=llm)

samples = load_dataset("locomo")
sample  = samples[0]

# 注入对话历史
agent.ingest_sample(sample)

# 对某个问题生成答案
qa = sample.qa_pairs[0]
prediction = agent.answer(qa.question)
print(f"Q: {qa.question}")
print(f"A (ground truth): {qa.answer}")
print(f"A (prediction):   {prediction}")
```

---

### 4.2 在 eval 循环中使用

`BaseAgent` 与 `run_eval` 配合使用是最常见的模式：

```python
from himga.agent import BaseAgent
from himga.memory import NullMemory
from himga.llm import get_client
from himga.eval import run_eval, compute_metrics
from himga.eval.judge import batch_judge
from himga.data import load_dataset

agent   = BaseAgent(memory=NullMemory(), llm=get_client())
samples = load_dataset("locomo")

# run_eval 内部自动：
#   1. agent.memory.reset()       — 每个 sample 前清空记忆
#   2. agent.ingest_sample(sample) — 注入对话历史
#   3. agent.answer(qa.question)   — 逐题生成预测
results      = run_eval(samples, agent=agent)
judge_scores = batch_judge(results, llm=get_client())
metrics      = compute_metrics(results, judge_scores)

print(f"Judge Score: {metrics['overall']['judge_score']:.3f}")
print(f"F1:          {metrics['overall']['f1']:.3f}")
```

---

### 4.3 自定义 prompt 结构（子类覆写）

覆写 `_build_messages` 以调整 prompt 格式，无需修改其他代码：

```python
from himga.agent import BaseAgent

class ChainOfThoughtAgent(BaseAgent):
    """在 prompt 中要求 LLM 先推理再给出最终答案。"""

    def _build_messages(self, question: str, context: str) -> list[dict]:
        system = (
            "You are a helpful assistant. Think step by step before giving your final answer."
        )
        if context:
            user = (
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Let's think step by step:"
            )
        else:
            user = f"Question: {question}\n\nLet's think step by step:"
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
```

---

### 4.4 替换记忆系统

`BaseAgent` 不耦合任何具体记忆实现，切换记忆系统只需替换构造参数：

```python
from himga.agent import BaseAgent
from himga.llm import get_client

# NullMemory baseline
from himga.memory import NullMemory
agent_null = BaseAgent(memory=NullMemory(), llm=get_client())

# 自定义关键词记忆（示例）
from my_project.memory import SimpleKeywordMemory
agent_kw = BaseAgent(memory=SimpleKeywordMemory(), llm=get_client())

# HiMGA 多图记忆（未来实现）
# from himga.memory.himga import HiMGAMemory
# agent_himga = BaseAgent(memory=HiMGAMemory(), llm=get_client())
```

---

## 5. 依赖关系图

```
BaseAgent
  │
  ├── memory: BaseMemory  ─── ingest(message, session)
  │                       ─── retrieve(query) → str
  │                       ─── reset()
  │
  └── llm: BaseLLMClient  ─── chat(messages) → str
```

`BaseAgent` 本身不持有 `Sample` 引用，不负责 `reset` 的调用时机。这两件事分别由调用方（`run_eval`）和 `memory` 自身管理。

---

## 6. 设计决策说明

### 为什么 `ingest_sample` 以 `Sample` 为粒度而非 `Session`？

`eval.runner.run_eval` 以 `Sample` 为驱动单元，接口粒度对齐可减少 runner 的职责（runner 不需要知道如何遍历 sessions）。同时，`Sample` 是逻辑完整的评测单元，agent 只关心"注入一个 sample"，不关心 sample 内部的 session 划分。

### 为什么 agent 不负责调用 `memory.reset()`？

职责分离：`reset` 的调用时机是评测协议的一部分，属于 `run_eval` 的职责，而不是 agent 的内部行为。这使 agent 可以在 eval 以外的场景（如单次交互测试）中使用，不会意外清空记忆。

### 为什么 `_build_messages` 以 `context=""` 而非 `context=None` 表示无记忆？

`BaseMemory.retrieve` 契约规定返回 `str`（空字符串表示无内容），与 `_build_messages` 的入参类型保持一致，避免 `None` 传播。同时，`if context:` 对空字符串和 `None` 都为假，即使调用方传入 `None` 也能正确退化为"无 context"分支（尽管 type hint 不允许 `None`）。

### 为什么 `BaseAgent` 是具体类而非抽象类？

大多数内置记忆实验（NullMemory、MAGMA replay 等）不需要修改 prompt 结构，直接实例化 `BaseAgent` 即可。将 `_build_messages` 设计为可覆写的普通方法，而不强制子类继承，降低了使用门槛。
