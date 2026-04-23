# `himga.data` — API 设计与使用说明

> 版本：基于当前实现（2026-04-21）
> 模块路径：`src/himga/data/`

---

## 目录

1. [概述](#1-概述)
2. [模块结构](#2-模块结构)
3. [Schema — 数据类型参考](#3-schema--数据类型参考)
   - [QuestionType](#31-questiontype)
   - [Message](#32-message)
   - [Session](#33-session)
   - [EvidenceRef](#34-evidenceref)
   - [QAPair](#35-qapair)
   - [Sample](#36-sample)
4. [Loader — 函数参考](#4-loader--函数参考)
   - [load_dataset](#41-load_dataset)
   - [load_locomo](#42-load_locomo)
   - [load_longmemeval](#43-load_longmemeval)
5. [数据集行为差异说明](#5-数据集行为差异说明)
6. [使用示例](#6-使用示例)
   - [基础加载](#61-基础加载)
   - [遍历对话历史](#62-遍历对话历史)
   - [按问题类型筛选](#63-按问题类型筛选)
   - [访问证据引用](#64-访问证据引用)
   - [访问原始数据](#65-访问原始数据)
   - [统计分析](#66-统计分析)
   - [接入 memory 系统（典型评测循环）](#67-接入-memory-系统典型评测循环)
7. [字段映射速查表](#7-字段映射速查表)
8. [设计决策说明](#8-设计决策说明)

---

## 1. 概述

`himga.data` 是 HiMGA 评测基座平台的数据层，职责是将两个评测数据集（LoCoMo 和 LongMemEval）的原始 JSON 文件解析为统一的 Python 数据结构，供上层的 `memory`、`agent`、`eval` 模块消费。

**核心设计目标**

- **统一接口**：两个数据集输出同一套数据类，上层代码无需感知来源差异。
- **无损保留**：所有原始字段通过 `raw: dict` 保留，不丢失数据集特有信息。
- **评测直接可用**：`QuestionType` 枚举覆盖两个数据集的问题分类，可直接按类别聚合指标，与 MAGMA Table 1/2 对齐。
- **懒解析**：时间戳以原始字符串保留（`date_str`），解析交由 `TemporalParser`，避免 loader 承载过多职责。

---

## 2. 模块结构

```
src/himga/data/
├── __init__.py            # 对外导出所有公共符号
├── schema.py              # 数据类定义（QuestionType / Message / Session / EvidenceRef / QAPair / Sample）
└── loaders/
    ├── __init__.py        # load_dataset() 统一入口
    ├── locomo.py          # LoCoMo 专用 loader
    └── longmemeval.py     # LongMemEval 专用 loader
```

**公共导入路径**

```python
from himga.data import (
    QuestionType,
    Message,
    Session,
    EvidenceRef,
    QAPair,
    Sample,
    load_dataset,
    load_locomo,
    load_longmemeval,
)
```

---

## 3. Schema — 数据类型参考

所有类型定义于 `himga.data.schema`，均为标准 `@dataclass`，支持直接构造、字段访问和 `==` 比较。

---

### 3.1 `QuestionType`

```python
class QuestionType(str, Enum)
```

继承自 `str`，可作字符串使用（如写入 JSON、字符串比较）。枚举值同时覆盖 LoCoMo 和 LongMemEval 的问题分类。

#### LoCoMo 分类（来自 `category` 整数字段）

| 枚举成员 | 字符串值 | LoCoMo category |
|---------|---------|----------------|
| `SINGLE_HOP` | `"single_hop"` | 1 |
| `TEMPORAL` | `"temporal"` | 2 |
| `MULTI_HOP` | `"multi_hop"` | 3 |
| `OPEN_DOMAIN` | `"open_domain"` | 4 |
| `ADVERSARIAL` | `"adversarial"` | 5 |

#### LongMemEval 分类（来自 `question_type` 字符串字段）

| 枚举成员 | 字符串值 | 原始字符串（支持连字符和下划线两种格式） |
|---------|---------|--------------------------------------|
| `SINGLE_SESSION_PREFERENCE` | `"single_session_preference"` | `"single-session-preference"` / `"single_session_preference"` |
| `SINGLE_SESSION_ASSISTANT` | `"single_session_assistant"` | `"single-session-assistant"` / `"single_session_assistant"` |
| `TEMPORAL_REASONING` | `"temporal_reasoning"` | `"temporal-reasoning"` / `"temporal_reasoning"` |
| `MULTI_SESSION` | `"multi_session"` | `"multi-session"` / `"multi_session"` |
| `KNOWLEDGE_UPDATE` | `"knowledge_update"` | `"knowledge-update"` / `"knowledge_update"` |
| `SINGLE_SESSION_USER` | `"single_session_user"` | `"single-session-user"` / `"single_session_user"` |

#### 示例

```python
from himga.data import QuestionType

# 作字符串使用
print(QuestionType.SINGLE_HOP)          # "single_hop"
assert QuestionType.SINGLE_HOP == "single_hop"

# 用于筛选
temporal_qa = [qa for qa in sample.qa_pairs
               if qa.question_type == QuestionType.TEMPORAL]

# 用于分组统计
from collections import Counter
type_counts = Counter(qa.question_type for s in samples for qa in s.qa_pairs)
```

---

### 3.2 `Message`

```python
@dataclass
class Message:
    role: str
    content: str
    turn_id: str | None = None
    date_str: str | None = None
```

对话中的一条发言。

| 字段 | 类型 | 必填 | 说明 |
|-----|------|------|------|
| `role` | `str` | ✓ | 说话者标识。LoCoMo 为真实姓名（如 `"Caroline"`）；LongMemEval 为 `"user"` 或 `"assistant"` |
| `content` | `str` | ✓ | 发言文本。LoCoMo 图片消息已在 loader 中将 `blip_caption` 内联为 `[Image: ...] text` 格式 |
| `turn_id` | `str \| None` | — | LoCoMo 的 `dia_id`，格式为 `"D{session}:{line}"`，如 `"D1:3"`。LongMemEval 无此字段，为 `None` |
| `date_str` | `str \| None` | — | 消息级时间戳原始字符串，通常为 `None`（时间信息在 Session 级） |

#### 关于 `turn_id` 的格式

LoCoMo 的 `dia_id` 格式为 `"D{N}:{M}"`，其中 `N` 为对话文档编号，`M` 为行号。与 `QAPair.evidence.turn_ids` 中的引用一一对应，可用于定位证据原文：

```python
# 找到 QA 证据对应的消息
def find_evidence_messages(sample: Sample, qa: QAPair) -> list[Message]:
    target_ids = set(qa.evidence.turn_ids)
    return [
        msg
        for sess in sample.sessions
        for msg in sess.messages
        if msg.turn_id in target_ids
    ]
```

---

### 3.3 `Session`

```python
@dataclass
class Session:
    session_id: str
    messages: list[Message]
    date_str: str | None = None
    title: str | None = None
```

一段时间窗口内的连续对话，是对话历史的基本组织单元。

| 字段 | 类型 | 必填 | 说明 |
|-----|------|------|------|
| `session_id` | `str` | ✓ | LoCoMo 为会话编号字符串（`"1"`, `"2"`, ...）；LongMemEval 为原始 `session_id`（如 `"sess_001"`） |
| `messages` | `list[Message]` | ✓ | 有序发言列表，按原始顺序排列 |
| `date_str` | `str \| None` | — | 会话时间戳原始字符串。LoCoMo 格式示例：`"1:56 pm on 8 May, 2023"`；LongMemEval 格式示例：`"2024-03-01"` |
| `title` | `str \| None` | — | 会话标题，仅 LongMemEval 提供，LoCoMo 为 `None` |

**注意**：LoCoMo 的 sessions 在 loader 中已按 `int(session_id)` 升序排序，保证时序正确。

---

### 3.4 `EvidenceRef`

```python
@dataclass
class EvidenceRef:
    turn_ids: list[str] = field(default_factory=list)
    session_ids: list[str] = field(default_factory=list)
```

QA pair 的证据指针，统一兼容两个数据集的粒度差异。

| 字段 | 类型 | 来源 | 说明 |
|-----|------|------|------|
| `turn_ids` | `list[str]` | LoCoMo `evidence` 字段 | Turn 级引用，如 `["D1:3", "D2:7"]`。LongMemEval 为空列表 |
| `session_ids` | `list[str]` | LongMemEval `answer_session_ids` 字段 | Session 级引用，如 `["session_2"]`。LoCoMo 为空列表 |

**注意**：两个字段不会同时非空——`turn_ids` 仅 LoCoMo 使用，`session_ids` 仅 LongMemEval 使用。

---

### 3.5 `QAPair`

```python
@dataclass
class QAPair:
    question_id: str
    question: str
    answer: str
    question_type: QuestionType
    evidence: EvidenceRef = field(default_factory=EvidenceRef)
    raw: dict = field(default_factory=dict)
```

一个问答评测对。

| 字段 | 类型 | 必填 | 说明 |
|-----|------|------|------|
| `question_id` | `str` | ✓ | 在 Sample 内的唯一标识。LoCoMo 为按序生成的整数字符串（`"0"`, `"1"`, ...）；LongMemEval 为原始 `question_id` |
| `question` | `str` | ✓ | 自然语言问题 |
| `answer` | `str` | ✓ | 标准答案，**始终为 `str` 类型**。LoCoMo 的 `int` 答案（如年份 `2022`）已在 loader 中转换为 `"2022"`；category=5 的对抗性问题已解析为 `adversarial_answer` 字段的内容 |
| `question_type` | `QuestionType` | ✓ | 归一化后的问题类型，用于按类别聚合评测指标 |
| `evidence` | `EvidenceRef` | — | 证据指针，默认为空 |
| `raw` | `dict` | — | 原始 QA 字典，保留 `category`、`adversarial_answer` 等全部原始字段 |

#### LoCoMo 对抗性答案（category=5）的解析规则

```
category == 5 时：
  answer = adversarial_answer  （若存在）
         OR raw["answer"]       （fallback）
         OR ""                  （最终 fallback）
```

对抗性问题的"正确答案"是模型应当识别出"无法从对话中确定"，`adversarial_answer` 字段通常包含类似 `"I cannot determine that from the conversation."` 的内容。

#### 访问原始字段

```python
qa = sample.qa_pairs[0]
print(qa.raw["category"])           # 原始 category 整数
print(qa.raw.get("adversarial_answer"))  # 对抗性答案原文（若有）
print(qa.raw["evidence"])           # 原始证据列表
```

---

### 3.6 `Sample`

```python
@dataclass
class Sample:
    sample_id: str
    dataset: str
    sessions: list[Session]
    qa_pairs: list[QAPair]
    speaker_a: str | None = None
    speaker_b: str | None = None
    raw: dict = field(default_factory=dict)
```

**评测单元**，是 loader 的最终输出粒度，也是 memory 系统 `ingest` + `eval` 的输入粒度。

| 字段 | 类型 | 必填 | 说明 |
|-----|------|------|------|
| `sample_id` | `str` | ✓ | 唯一标识。LoCoMo 优先使用原始 `sample_id` 字段，无则用全局序号；LongMemEval 等于对应 `question_id` |
| `dataset` | `str` | ✓ | 来源数据集，固定为 `"locomo"` 或 `"longmemeval"` |
| `sessions` | `list[Session]` | ✓ | 对话历史，按时间升序。LoCoMo 为多个 session；LongMemEval 为 haystack sessions |
| `qa_pairs` | `list[QAPair]` | ✓ | 评测问答对。**LoCoMo 一个 Sample 含多个**（通常 5～30 个）；**LongMemEval 一个 Sample 恰好含一个** |
| `speaker_a` | `str \| None` | — | LoCoMo 主说话者姓名。LongMemEval 为 `None` |
| `speaker_b` | `str \| None` | — | LoCoMo 次说话者姓名。LongMemEval 为 `None` |
| `raw` | `dict` | — | 数据集特有的辅助字段：<br>• LoCoMo：`event_summary`、`observation`、`session_summary`<br>• LongMemEval：`question_date` |

#### LoCoMo `raw` 中的辅助字段

```python
# event_summary: {session_id -> {speaker -> [event_str, ...]}}
sample.raw["event_summary"]["1"]["Alice"]  # → ["Greeted Bob", ...]

# session_summary: {session_id -> summary_str}
sample.raw["session_summary"]["1"]  # → "Alice and Bob greeted each other."

# observation: {session_id -> {speaker -> [[obs, evidence], ...]}}
sample.raw.get("observation")
```

#### LongMemEval `raw` 中的辅助字段

```python
# question_date: 问题提出的日期，ISO 格式字符串
sample.raw["question_date"]  # → "2024-03-15"
```

---

## 4. Loader — 函数参考

### 4.1 `load_dataset`

```python
def load_dataset(name: str) -> list[Sample]
```

统一入口，按名称加载数据集，若本地不存在则自动下载。

**参数**

| 参数 | 类型 | 说明 |
|-----|------|------|
| `name` | `str` | 数据集名称，支持 `"locomo"` 或 `"longmemeval"` |

**返回**

`list[Sample]`：解析完毕的样本列表。

**异常**

| 异常 | 触发条件 |
|-----|---------|
| `ValueError` | `name` 不在支持列表中 |
| 网络相关异常 | 下载失败（由 `get_dataset` 抛出） |

**示例**

```python
from himga.data import load_dataset

samples = load_dataset("locomo")       # 自动下载（若未缓存）
samples = load_dataset("longmemeval")
```

**内部流程**

```
load_dataset(name)
  → get_dataset(name)       # himga.utils：检查本地缓存，缺则下载 → 返回 Path
  → load_locomo(path)       # 或 load_longmemeval(path)
  → list[Sample]
```

---

### 4.2 `load_locomo`

```python
def load_locomo(path: Path) -> list[Sample]
```

从目录加载所有 LoCoMo 样本。

**参数**

| 参数 | 类型 | 说明 |
|-----|------|------|
| `path` | `Path` | `get_dataset("locomo")` 返回的目录路径 |

**返回**

`list[Sample]`：按文件名字母序、文件内顺序排列的样本列表。

**文件格式要求**

目录下的每个 `.json` 文件应为：
- 一个 raw sample dict 的列表（常见）
- 或一个单独的 raw sample dict（自动兼容）

非 `.json` 文件和子目录均被忽略。

**解析行为**

1. 按 `sorted()` 遍历目录中的 `.json` 文件
2. 每个 sample dict 中提取 `conversation` 键下的 `session_N` 和 `session_N_date_time` 字段
3. 空 session（无有效 message）被过滤
4. sessions 按 `int(session_id)` 升序排序
5. QA 的 `answer` 类型统一为 `str`；category=5 使用 `adversarial_answer`
6. `sample_id` 优先使用原始数据的 `sample_id` 字段，无则使用全局自增序号

**图片消息处理**

LoCoMo 包含图片对话轮次（`img_url` + `blip_caption`），loader 将图片标题内联为文本：

```
原始：{"speaker": "Alice", "dia_id": "D1:5", "text": "Look at this!", "blip_caption": "a sunset photo"}
解析：Message(role="Alice", content="[Image: a sunset photo] Look at this!", turn_id="D1:5")
```

---

### 4.3 `load_longmemeval`

```python
def load_longmemeval(path: Path) -> list[Sample]
```

从目录加载所有 LongMemEval 样本。

**参数**

| 参数 | 类型 | 说明 |
|-----|------|------|
| `path` | `Path` | `get_dataset("longmemeval")` 返回的目录路径 |

**返回**

`list[Sample]`：每个 `LongMemQuestion` 对应一个 `Sample`。

**文件格式要求**

目录下的每个 `.json` 文件应为一个问题 dict 的列表。每个 dict 须包含：

| 字段 | 说明 |
|-----|------|
| `question_id` | 问题唯一 ID |
| `question_type` | 问题类型字符串（支持连字符和下划线两种格式） |
| `question` | 问题文本 |
| `question_date` | 问题日期（ISO 格式） |
| `answer` | 标准答案（任意类型，loader 转为 str） |
| `answer_session_ids` | 证据 session ID 列表 |
| `haystack_session_ids` | 所有 haystack session ID 列表 |
| `haystack_dates` | 与 haystack session ID 并行的日期列表 |
| `haystack_sessions` | 与 haystack session ID 并行的消息列表的列表 |

**解析行为**

1. 每个 question dict → 一个 `Sample`
2. `haystack_sessions`、`haystack_session_ids`、`haystack_dates` 按索引并行组合为 `list[Session]`
3. `question_type` 同时支持连字符（`"single-session-user"`）和下划线（`"single_session_user"`）格式
4. 未知 `question_type` fallback 为 `QuestionType.SINGLE_SESSION_USER`
5. `answer` 强制转为 `str`

---

## 5. 数据集行为差异说明

| 维度 | LoCoMo | LongMemEval |
|------|--------|-------------|
| **Sample 粒度** | 一个完整对话 + 多个 QA pair | 一个 question + 对应的 haystack sessions |
| **每 Sample QA 数** | 多个（通常 5-30） | 恰好 1 个 |
| **Session 时间戳** | 详细（如 `"1:56 pm on 8 May, 2023"`） | ISO 日期（如 `"2024-03-01"`） |
| **Message role** | 真实姓名（`"Caroline"`, `"Melanie"`） | `"user"` / `"assistant"` |
| **Turn ID** | 有（`dia_id`，如 `"D1:3"`） | 无（`None`） |
| **图片消息** | 有（blip_caption 内联） | 无 |
| **证据粒度** | Turn 级（`evidence.turn_ids`） | Session 级（`evidence.session_ids`） |
| **speaker 字段** | `speaker_a`, `speaker_b` 非 None | 均为 `None` |
| **`sample.raw` 内容** | `event_summary`, `observation`, `session_summary` | `question_date` |

---

## 6. 使用示例

### 6.1 基础加载

```python
from himga.data import load_dataset

# 一行加载（自动下载缓存）
locomo_samples = load_dataset("locomo")
lme_samples    = load_dataset("longmemeval")

print(f"LoCoMo: {len(locomo_samples)} samples")
print(f"LongMemEval: {len(lme_samples)} samples")
```

也可以直接传入已知路径（跳过下载逻辑）：

```python
from pathlib import Path
from himga.data import load_locomo, load_longmemeval

samples = load_locomo(Path("/path/to/locomo/dir"))
```

---

### 6.2 遍历对话历史

```python
sample = locomo_samples[0]

print(f"[{sample.sample_id}] {sample.speaker_a} & {sample.speaker_b}")
print(f"  sessions: {len(sample.sessions)}, qa_pairs: {len(sample.qa_pairs)}")

for sess in sample.sessions:
    print(f"\n  [Session {sess.session_id}]  {sess.date_str or '(no date)'}")
    for msg in sess.messages:
        tid = f" ({msg.turn_id})" if msg.turn_id else ""
        print(f"    {msg.role}{tid}: {msg.content[:60]}")
```

输出示例：

```
[conv-26] Caroline & Melanie
  sessions: 19, qa_pairs: 12

  [Session 1]  1:56 pm on 8 May, 2023
    Caroline (D1:1): Hey Mel! Good to see you! How have you been?
    Melanie  (D1:2): Hey Caroline! I'm swamped with the kids & work.
```

---

### 6.3 按问题类型筛选

```python
from himga.data import QuestionType

def get_qa_by_type(samples, qtype: QuestionType):
    return [
        (s, qa)
        for s in samples
        for qa in s.qa_pairs
        if qa.question_type == qtype
    ]

# LoCoMo
temporal_pairs  = get_qa_by_type(locomo_samples, QuestionType.TEMPORAL)
adversarial_pairs = get_qa_by_type(locomo_samples, QuestionType.ADVERSARIAL)

# LongMemEval
multi_session_pairs = get_qa_by_type(lme_samples, QuestionType.MULTI_SESSION)

print(f"Temporal QAs: {len(temporal_pairs)}")
print(f"Adversarial QAs: {len(adversarial_pairs)}")
```

---

### 6.4 访问证据引用

```python
# LoCoMo：定位证据 turn
def find_evidence_messages(sample, qa):
    target_ids = set(qa.evidence.turn_ids)
    return [
        msg
        for sess in sample.sessions
        for msg in sess.messages
        if msg.turn_id in target_ids
    ]

qa = locomo_samples[0].qa_pairs[0]
evidence_msgs = find_evidence_messages(locomo_samples[0], qa)
print(f"Q: {qa.question}")
print(f"A: {qa.answer}")
for m in evidence_msgs:
    print(f"  Evidence [{m.turn_id}] {m.role}: {m.content}")
```

```python
# LongMemEval：定位证据 session
def find_evidence_sessions(sample, qa):
    target_ids = set(qa.evidence.session_ids)
    return [s for s in sample.sessions if s.session_id in target_ids]

qa = lme_samples[0].qa_pairs[0]
evidence_sessions = find_evidence_sessions(lme_samples[0], qa)
```

---

### 6.5 访问原始数据

```python
# LoCoMo：访问 event_summary
sample = locomo_samples[0]
for sess_id, speakers in sample.raw.get("event_summary", {}).items():
    for speaker, events in speakers.items():
        print(f"  Session {sess_id} / {speaker}: {events}")

# LoCoMo：访问 session_summary
for sess_id, summary in sample.raw.get("session_summary", {}).items():
    print(f"  Session {sess_id}: {summary}")

# QAPair 原始字段
qa = sample.qa_pairs[0]
print(qa.raw["category"])           # 原始 category int
print(qa.raw.get("adversarial_answer"))

# LongMemEval：访问 question_date
lme_sample = lme_samples[0]
print(lme_sample.raw["question_date"])  # "2024-03-15"
```

---

### 6.6 统计分析

```python
from collections import Counter, defaultdict

# 问题类型分布
type_counts = Counter(
    qa.question_type
    for s in locomo_samples
    for qa in s.qa_pairs
)
for qtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
    print(f"  {qtype.value:30s}: {count}")

# 每个 sample 的会话数分布
session_counts = [len(s.sessions) for s in locomo_samples]
print(f"sessions per sample — min: {min(session_counts)}, "
      f"max: {max(session_counts)}, "
      f"avg: {sum(session_counts)/len(session_counts):.1f}")

# 统计各 session 总 turn 数
total_turns = sum(
    len(sess.messages)
    for s in locomo_samples
    for sess in s.sessions
)
print(f"Total turns: {total_turns}")
```

---

### 6.7 接入 memory 系统（典型评测循环）

`Sample` 的设计即为此循环量身定制：

```python
from himga.data import load_dataset
from himga.memory import BaseMemory   # 下一阶段实现

def run_eval(samples, memory: BaseMemory):
    results = []
    for sample in samples:
        memory.reset()

        # 1. 按 session 顺序将对话历史注入记忆系统
        for sess in sample.sessions:
            for msg in sess.messages:
                memory.ingest(msg)

        # 2. 对每个 QA pair 检索 + 生成答案
        for qa in sample.qa_pairs:
            context = memory.retrieve(qa.question)
            response = generate_answer(qa.question, context)  # 调用 LLM
            results.append({
                "sample_id":     sample.sample_id,
                "question_id":   qa.question_id,
                "question_type": qa.question_type,
                "question":      qa.question,
                "answer":        qa.answer,
                "response":      response,
            })

    return results
```

---

## 7. 字段映射速查表

### LoCoMo 原始字段 → HiMGA 字段

| 原始字段（位置） | HiMGA 字段 |
|----------------|-----------|
| `sample["sample_id"]` | `Sample.sample_id` |
| `conv["speaker_a"]` | `Sample.speaker_a` |
| `conv["speaker_b"]` | `Sample.speaker_b` |
| `conv["session_N"]` | `Sample.sessions[i].messages` |
| `conv["session_N_date_time"]` | `Sample.sessions[i].date_str` |
| `turn["speaker"]` | `Message.role` |
| `turn["text"]` (+ blip_caption) | `Message.content` |
| `turn["dia_id"]` | `Message.turn_id` |
| `qa["question"]` | `QAPair.question` |
| `qa["answer"]` (category≠5) | `QAPair.answer` |
| `qa["adversarial_answer"]` (category=5) | `QAPair.answer` |
| `qa["category"]` | `QAPair.question_type`（via 映射表） |
| `qa["evidence"]` | `QAPair.evidence.turn_ids` |
| `sample["event_summary"]` | `Sample.raw["event_summary"]` |
| `sample["session_summary"]` | `Sample.raw["session_summary"]` |
| `sample["observation"]` | `Sample.raw["observation"]` |

### LongMemEval 原始字段 → HiMGA 字段

| 原始字段 | HiMGA 字段 |
|---------|-----------|
| `item["question_id"]` | `Sample.sample_id` = `QAPair.question_id` |
| `item["question_type"]` | `QAPair.question_type`（via 映射表） |
| `item["question"]` | `QAPair.question` |
| `str(item["answer"])` | `QAPair.answer` |
| `item["answer_session_ids"]` | `QAPair.evidence.session_ids` |
| `item["haystack_session_ids"][i]` | `Sample.sessions[i].session_id` |
| `item["haystack_dates"][i]` | `Sample.sessions[i].date_str` |
| `item["haystack_sessions"][i]` | `Sample.sessions[i].messages` |
| `message["role"]` | `Message.role` |
| `message["content"]` | `Message.content` |
| `item["question_date"]` | `Sample.raw["question_date"]` |

---

## 8. 设计决策说明

### 为什么 `Message.role` 不做枚举？

LoCoMo 的 role 是真实姓名，枚举无法覆盖。统一用 `str` 避免信息损失，上层代码可通过 `sample.speaker_a`/`speaker_b` 判断说话者身份。

### 为什么 `QAPair.answer` 始终为 `str`？

LoCoMo 的 answer 字段可为 `int`（如年份 `2022`），在 loader 阶段统一 `str(answer)` 规范化，使评测层的指标计算无需感知类型差异。

### 为什么 adversarial 解析在 loader 层而非 schema 层？

`QAPair` 的语义是"标准答案已就绪"，保持 schema 的纯洁性；adversarial 是 LoCoMo 特有的数据集约定，放在 loader 中处理不污染通用接口。

### 为什么时间戳保留为 `date_str` 而非 `datetime`？

LoCoMo 的时间格式（`"1:56 pm on 8 May, 2023"`）需要专门的 `TemporalParser` 解析，解析逻辑复杂且与 NLP 相关，不应耦合进数据加载层。`date_str` 保留原始字符串，由调用方按需解析。

### 为什么 LongMemEval 一个问题对应一个 Sample？

LongMemEval 的每个 question 携带独立的 `haystack_sessions`（对话历史），即每个问题的"上下文"是独立的，与 LoCoMo 中所有 QA 共享同一对话不同。将一个 question 映射为一个 Sample，使两个数据集在评测循环中的处理逻辑完全对称。

### 为什么 `EvidenceRef` 同时有 `turn_ids` 和 `session_ids`？

两个数据集的证据粒度不同：LoCoMo 精确到 turn，LongMemEval 精确到 session。用一个类统一表达，调用方按数据集检查对应字段即可，而不是用 `Union` 类型或两套不同的类。
