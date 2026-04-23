# `himga.memory` — API 设计与使用说明

> 版本：基于当前实现（2026-04-22）
> 模块路径：`src/himga/memory/`

---

## 目录

1. [概述](#1-概述)
2. [模块结构](#2-模块结构)
3. [类型参考](#3-类型参考)
   - [BaseMemory](#31-basememory)
   - [NullMemory](#32-nullmemory)
4. [接口契约](#4-接口契约)
5. [使用示例](#5-使用示例)
   - [使用 NullMemory 跑通流水线](#51-使用-nullmemory-跑通流水线)
   - [实现自定义 Memory](#52-实现自定义-memory)
   - [在 eval 循环中使用](#53-在-eval-循环中使用)
6. [设计决策说明](#6-设计决策说明)

---

## 1. 概述

`himga.memory` 定义记忆系统的统一抽象接口。所有记忆系统实现（NullMemory、MAGMA 复现、HiMGA 多图记忆等）均实现 `BaseMemory` 接口，使 `agent` 和 `eval` 层与具体实现完全解耦。

**目前提供的实现**

| 实现 | 说明 |
|------|------|
| `NullMemory` | 空实现（无存储、无检索），用于 pipeline 验证和 baseline 评测 |

---

## 2. 模块结构

```
src/himga/memory/
├── __init__.py    # 导出 BaseMemory, NullMemory
├── base.py        # BaseMemory 抽象接口
└── null.py        # NullMemory 实现
```

**公共导入路径**

```python
from himga.memory import BaseMemory, NullMemory
```

---

## 3. 类型参考

### 3.1 `BaseMemory`

```python
class BaseMemory(ABC)
```

所有记忆系统的抽象基类，定义三个生命周期方法：写入（ingest）、检索（retrieve）、清空（reset）。

#### 方法


##### `ingest(message, session)`

```python
@abstractmethod
def ingest(self, message: Message, session: Session) -> None
```

将一条消息写入记忆系统。

**参数**

| 参数 | 类型 | 说明 |
|-----|------|------|
| `message` | `Message` | 要写入的消息，提供 `role`、`content`、`turn_id` 等信息 |
| `session` | `Session` | 消息所属的会话，提供时间戳（`date_str`）、会话 ID 等上下文 |

`session` 参数的存在使记忆系统能够感知时序信息（如按 `session.date_str` 建立时间索引），即使不使用也须传入。

---

##### `retrieve(query)`

```python
@abstractmethod
def retrieve(self, query: str) -> str
```

检索与查询相关的记忆，返回可直接注入 prompt 的上下文字符串。

**参数**

| 参数 | 类型 | 说明 |
|-----|------|------|
| `query` | `str` | 自然语言查询，通常直接使用评测问题 |

**返回**

`str`：组装好的上下文字符串，无相关记忆时返回空字符串 `""`。

**契约**：返回值须是可直接拼入 prompt 的文本；内部结构化处理由具体实现自行完成。

---

##### `reset()`

```python
@abstractmethod
def reset(self) -> None
```

清空全部已存储的记忆，准备处理下一个 `Sample`。

由 `eval.runner.run_eval` 在每个 Sample 处理前调用，保证样本间的完全隔离。

---

### 3.2 `NullMemory`

```python
class NullMemory(BaseMemory)
```

无操作实现：`ingest` 丢弃所有输入，`retrieve` 始终返回 `""`，`reset` 无任何副作用。

**用途**

1. **流水线验证**：在真正的记忆系统就绪前，先用 NullMemory 确认 data → memory → agent → eval 全链路可跑通
2. **Baseline 评测**：NullMemory 下 LLM 仅凭自身参数知识作答，对应 MAGMA 中的"LLM only"基线，有独立的参考价值

```python
class NullMemory(BaseMemory):
    def ingest(self, message: Message, session: Session) -> None: pass
    def retrieve(self, query: str) -> str: return ""
    def reset(self) -> None: pass
```

---

## 4. 接口契约

实现 `BaseMemory` 时须满足以下语义：

| 契约 | 说明 |
|------|------|
| `retrieve("")` 不崩溃 | 空查询须返回字符串，不得抛出异常 |
| 多次 `ingest` 后 `reset` → `retrieve` 返回空 | `reset` 须彻底清除所有状态 |
| `reset` 后继续 `ingest` 正常工作 | reset 不破坏对象的可用性 |
| 不同实例之间相互独立 | 类级别不得有共享可变状态 |
| `retrieve` 返回 `str` | 不得返回 `None` 或其他类型 |

---

## 5. 使用示例

### 5.1 使用 NullMemory 跑通流水线

```python
from himga.memory import NullMemory
from himga.data import load_dataset

memory = NullMemory()

samples = load_dataset("locomo")
sample = samples[0]

# 注入历史
for sess in sample.sessions:
    for msg in sess.messages:
        memory.ingest(msg, sess)

# 检索（NullMemory 始终返回空字符串）
context = memory.retrieve("What did Alice say about her job?")
print(repr(context))  # ''

# 清空准备下一个 sample
memory.reset()
```

---

### 5.2 实现自定义 Memory

继承 `BaseMemory` 并实现三个抽象方法：

```python
from himga.memory import BaseMemory
from himga.data.schema import Message, Session


class SimpleKeywordMemory(BaseMemory):
    """按关键词索引的简单记忆实现（示例用途）。"""

    def __init__(self) -> None:
        self._store: list[str] = []

    def ingest(self, message: Message, session: Session) -> None:
        # 将消息内容（附带时间戳）写入线性存储
        timestamp = session.date_str or "unknown"
        self._store.append(f"[{timestamp}] {message.role}: {message.content}")

    def retrieve(self, query: str) -> str:
        # 朴素关键词匹配
        keywords = set(query.lower().split())
        relevant = [
            line for line in self._store
            if any(kw in line.lower() for kw in keywords)
        ]
        return "\n".join(relevant[-10:])  # 最多返回最近 10 条

    def reset(self) -> None:
        self._store.clear()
```

确认实现符合接口：

```python
from himga.data.schema import Message, Session

mem = SimpleKeywordMemory()
sess = Session(session_id="s1", messages=[], date_str="2024-01-01")
msg  = Message(role="user", content="I love hiking in the mountains.")

mem.ingest(msg, sess)
print(mem.retrieve("hiking"))   # "[2024-01-01] user: I love hiking in the mountains."
mem.reset()
print(mem.retrieve("hiking"))   # ""
```

---

### 5.3 在 eval 循环中使用

`BaseMemory` 的设计粒度与 `eval.runner.run_eval` 完全对齐：

```python
from himga.memory import NullMemory
from himga.agent import BaseAgent
from himga.llm import get_client
from himga.eval import run_eval, compute_metrics
from himga.eval.judge import batch_judge
from himga.data import load_dataset

# 替换为任意 BaseMemory 实现
memory = NullMemory()
llm    = get_client("anthropic")
agent  = BaseAgent(memory=memory, llm=llm)

samples = load_dataset("locomo")
results = run_eval(samples, agent=agent, show_progress=True)

judge_scores = batch_judge(results, llm=llm)
metrics = compute_metrics(results, judge_scores)
print(metrics["overall"])
```

---

## 6. 设计决策说明

### 为什么 `ingest` 接受 `Message` 而不是 `str`？

完整的 `Message` 对象携带 `role`（说话者）、`turn_id`（证据索引）等信息，这些对记忆系统的索引构建有实际价值（如仅索引 user 消息、按 dia_id 反查证据）。若只传 `str` 则这些元信息丢失。

### 为什么 `ingest` 同时传 `session`？

部分记忆系统需要按会话时间排序（如时间衰减机制），或需要 session 粒度的聚合（如摘要记忆）。统一在接口层传入，不强制使用，但不传则无法支持这类实现。

### 为什么 `retrieve` 返回 `str` 而非结构化对象？

`BaseAgent._build_messages` 直接将 `retrieve` 的返回值拼入 prompt，要求它是可注入的文本。不同记忆实现的内部结构差异很大（向量列表、图节点、摘要段落），统一归约到字符串是最大公约数。复杂实现可在内部保留结构，对外只暴露文本。

### 为什么 `reset` 语义是"清空当前 sample"而非"彻底销毁"？

`reset` 后对象必须可继续使用（`eval` 循环会反复 reset 同一实例），而非"关闭"。与 `__init__` 的区别在于：reset 是轻量的状态清除，适合高频调用。
