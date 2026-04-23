# LongMemEval 数据集说明

> 覆盖：数据集背景、原始格式、字段说明、统计信息、HiMGA 接口映射
> 最后更新：2026-04-21

---

## 目录

1. [数据集背景](#1-数据集背景)
2. [文件结构](#2-文件结构)
3. [顶层结构](#3-顶层结构)
4. [haystack_sessions 详解](#4-haystack_sessions-详解)
5. [问题类型详解](#5-问题类型详解)
6. [answer 字段说明](#6-answer-字段说明)
7. [数据统计](#7-数据统计)
8. [与 LoCoMo 的核心差异](#8-与-locomo-的核心差异)
9. [HiMGA 接口映射](#9-himga-接口映射)
10. [常见使用模式](#10-常见使用模式)

---

## 1. 数据集背景

**LongMemEval** 是一个专门评测大语言模型在真实助手对话场景下长期记忆能力的基准数据集，由吴晓吴（Xiaowu Wu）等人发布，HuggingFace 源：`xiaowu0162/longmemeval-cleaned`。

**核心特点**：

- **真实助手对话**：haystack sessions 来自真实的用户-助手对话（ShareGPT 等来源），而非两个用户的闲聊
- **大规模 haystack**：每道题的对话历史包含 38-62 个 session（平均 ~48 个），对应约 100K tokens 的上下文
- **针对性问题类型**：六种类型精确覆盖长期记忆的不同维度（偏好记忆、助手行为、时间推理、跨会话聚合、知识更新）
- **一问一 haystack**：每道题携带自己独立的 haystack sessions（背景对话），不同题目之间的 haystack 相互独立

**评测任务**：在长达数十个会话的历史对话中，回答涉及用户偏好、事实、时间关系等方面的问题。

**版本说明**：HiMGA 使用的是 `longmemeval-cleaned` 版本，相较于原版去除了会干扰答案正确性的噪声历史会话。

---

## 2. 文件结构

```
$DATASETS_ROOT/longmemeval/
├── longmemeval_s_cleaned.json   # Small 版，500 题，较快评测用
├── longmemeval_m_cleaned.json   # Medium 版，更多题目
└── longmemeval_oracle.json      # Oracle 版（含答案 session 原文，供分析用）
```

每个文件均为 JSON **列表**，每个元素是一道独立的评测题（含完整 haystack）：

```python
import json
data = json.load(open("longmemeval_s_cleaned.json"))
# data: list, len=500
# data[0]: dict，一道完整评测题
```

---

## 3. 顶层结构

每道题是一个 dict，包含以下字段：

```text
{
  "question_id":          "e47becba",
  "question_type":        "single-session-user",
  "question":             "What degree did I graduate with?",
  "question_date":        "2023/05/30 (Tue) 23:40",
  "answer":               "'Business Administration'",
  "answer_session_ids":   ["answer_280352e9"],
  "haystack_dates":       ["2023/05/20 (Sat) 02:21", "2023/05/20 (Sat) 02:57", ...],
  "haystack_session_ids": ["sharegpt_yywfIrx_0", "85a1be56_1", ...],
  "haystack_sessions":    [ [...], [...], ... ]
}
```

| 字段 | 类型 | 说明 |
|-----|------|------|
| `question_id` | `str` | 题目唯一 ID，短哈希格式，如 `"e47becba"` |
| `question_type` | `str` | 问题类型，连字符格式，如 `"single-session-user"`（详见第 5 节） |
| `question` | `str` | 自然语言问题，使用第一人称（"I"），模拟用户向助手提问 |
| `question_date` | `str` | 问题提出的时间，格式 `"YYYY/MM/DD (weekday) HH:MM"` |
| `answer` | `str` 或 `int` | 标准答案（详见第 6 节） |
| `answer_session_ids` | `list[str]` | 包含答案的 session 的 ID 列表（是 `haystack_session_ids` 的子集） |
| `haystack_dates` | `list[str]` | 与 `haystack_session_ids` 并行的日期列表 |
| `haystack_session_ids` | `list[str]` | 所有 haystack session 的 ID 列表 |
| `haystack_sessions` | `list[list[dict]]` | 所有 haystack session 的消息内容（与 IDs 并行对应） |

---

## 4. `haystack_sessions` 详解

### 4.1 数据结构

`haystack_sessions` 是一个**列表的列表**：

```
haystack_sessions = [
  session_0_messages,    # list[dict]
  session_1_messages,    # list[dict]
  ...
  session_N_messages,    # list[dict]
]
```

每个 `session_i_messages` 对应 `haystack_session_ids[i]` 和 `haystack_dates[i]`，三者**并行对应**：

```python
for messages, session_id, date in zip(
    item["haystack_sessions"],
    item["haystack_session_ids"],
    item["haystack_dates"]
):
    print(f"Session {session_id} @ {date}: {len(messages)} messages")
```

### 4.2 消息结构

每条消息是一个 dict，只有两个字段：

```json
{"role": "user",      "content": "The farmer needs to transport a fox..."}
{"role": "assistant", "content": "To solve this puzzle, the farmer can follow these steps..."}
```

| 字段 | 类型 | 取值 | 说明 |
|-----|------|------|------|
| `role` | `str` | `"user"` 或 `"assistant"` | 消息发送者角色 |
| `content` | `str` | — | 消息文本 |

**注意**：

- 与 LoCoMo 不同，这里没有 `dia_id`（没有 turn 级标识）
- `role` 是标准化的 `"user"/"assistant"`，不是真实姓名
- 每个 session 是一段完整的用户-助手对话，内容相互独立（来自不同日期、不同话题）

### 4.3 Session ID 的含义

Session ID（如 `"sharegpt_yywfIrx_0"`、`"answer_280352e9"`）是来源标识：

- `sharegpt_*`：来自 ShareGPT 的真实对话
- `answer_*`：包含答案信息的 session（`answer_session_ids` 中的 ID 属于此类）
- 其他格式：其他来源的对话

`answer_session_ids` 是 `haystack_session_ids` 的子集，指向那些真正包含答案的 session，是评测的证据 session。

### 4.4 时间格式

```
"2023/05/20 (Sat) 02:21"
"2023/05/30 (Tue) 23:40"
"2023/02/15 (Wed) 23:50"
```

格式为 `"YYYY/MM/DD (weekday) HH:MM"`，比 LoCoMo 的格式更规范，但仍保留为原始字符串（`Session.date_str`）

---

## 5. 问题类型详解

`question_type` 字段使用连字符格式字符串，共六种类型：

### 5.1 `single-session-user`

**中文**：单会话用户事实记忆

**含义**：答案来自单个 session，考察模型对用户所述个人事实的记忆。

```json
{
  "question_type": "single-session-user",
  "question": "What degree did I graduate with?",
  "answer": "'Business Administration'",
  "answer_session_ids": ["answer_280352e9"]
}
```

**特点**：答案来源单一，是最基础的记忆类型。考察模型能否从大量噪声 session 中找到那个包含答案的 session。

---

### 5.2 `single-session-assistant`

**中文**：单会话助手行为记忆

**含义**：答案来自单个 session，但考察的是助手在那次对话中**说了什么、做了什么**，而非用户说的。

```json
{
  "question_type": "single-session-assistant",
  "question": "I'm checking our previous chat about the shift rotation sheet for GM social media agents. Can you remind me what was the rotation for Admon on a Sunday?",
  "answer": "Admon was assigned to the 8 am - 4 pm (Day Shift) on Sundays.",
  "answer_session_ids": ["answer_sharegpt_5Lzox6N_0"]
}
```

**特点**：用户在问"你之前告诉我什么"，模型需要记住自己之前的输出内容。

---

### 5.3 `single-session-preference`

**中文**：单会话偏好记忆

**含义**：答案来自单个 session，考察模型对用户**偏好、喜好、习惯**的记忆。

```json
{
  "question_type": "single-session-preference",
  "question": "Can you recommend some resources where I can learn more about video editing?",
  "answer": "The user would prefer responses that suggest resources specifically tailored to Adobe Premiere Pro, especially those that delve into its advanced settings. They might not prefer general video editing resources or resources related to other video editing software.",
  "answer_session_ids": ["answer_edb03329"]
}
```

**特点**：答案不是简单事实，而是对用户偏好的描述性总结。注意 `answer` 较长，是对偏好的完整刻画。

---

### 5.4 `temporal-reasoning`

**中文**：时间推理

**含义**：需要跨多个 session 进行时间计算或推理。

```json
{
  "question_type": "temporal-reasoning",
  "question": "How many days passed between my visit to the Museum of Modern Art (MoMA) and the 'Ancient Civilizations' exhibit at the Metropolitan Museum of Art?",
  "answer": "7 days. 8 days (including the last day) is also acceptable.",
  "answer_session_ids": ["answer_d00ba6d0_1", "answer_d00ba6d0_2"]
}
```

**特点**：答案依赖多个 session 的时间戳或对话中提到的日期，需要计算时间差、顺序、间隔等。注意 answer 中可能包含多种可接受答案。

---

### 5.5 `multi-session`

**中文**：跨会话聚合

**含义**：答案需要跨多个 session 汇总信息（常见于计数、列举类问题）。

```json
{
  "question_type": "multi-session",
  "question": "How many items of clothing do I need to pick up or return from a store?",
  "answer": 3,
  "answer_session_ids": ["answer_afa9873b_2", "answer_afa9873b_3", "answer_afa9873b_1"]
}
```

**特点**：答案分散在多个 session 中，需要整合后才能得出。`answer` 常为 `int`（计数结果）。`answer_session_ids` 包含多个 session ID。

---

### 5.6 `knowledge-update`

**中文**：知识更新追踪

**含义**：用户在不同 session 中对同一事实给出了不同描述（更新），模型需要使用最新的信息。

```json
{
  "question_type": "knowledge-update",
  "question": "What was my personal best time in the charity 5K run?",
  "answer": "25 minutes and 50 seconds (or 25:50)",
  "answer_session_ids": ["answer_a25d4a91_1", "answer_a25d4a91_2"]
}
```

**特点**：考察模型能否识别知识更新，避免使用过时信息。这类问题在实际助手场景中非常常见（用户告知新情况后期望助手更新认知）。

---

### 5.7 类型分布（_s 版，500 题）

| question_type | HiMGA QuestionType | 数量 | 占比 |
|---------------|--------------------|------|------|
| `multi-session` | `MULTI_SESSION` | 133 | 26.6% |
| `temporal-reasoning` | `TEMPORAL_REASONING` | 133 | 26.6% |
| `knowledge-update` | `KNOWLEDGE_UPDATE` | 78 | 15.6% |
| `single-session-user` | `SINGLE_SESSION_USER` | 70 | 14.0% |
| `single-session-assistant` | `SINGLE_SESSION_ASSISTANT` | 56 | 11.2% |
| `single-session-preference` | `SINGLE_SESSION_PREFERENCE` | 30 | 6.0% |

---

## 6. `answer` 字段说明

`answer` 的类型不固定：

| 类型 | 出现场景 | 示例 |
|-----|---------|------|
| `str` | 大多数题目 | `"'Business Administration'"` |
| `int` | 计数类问题（multi-session） | `3` |

**注意**：部分 `str` 类型的 answer 带有单引号（如 `"'Business Administration'"`），这是数据集原始格式，不是嵌套引号错误。

HiMGA loader 统一用 `str(answer)` 转换，`int` 3 → `"3"`，保证 `QAPair.answer` 始终为 `str`。

---

## 7. 数据统计

### _s 版（`longmemeval_s_cleaned.json`）

| 指标 | 数值 |
|-----|------|
| 总题目数 | 500 |
| haystack session 数范围 | 38 – 62 |
| 平均 haystack session 数 | ~47.7 |
| answer 为 str | 468（93.6%） |
| answer 为 int | 32（6.4%） |

### 两个版本对比

| 版本 | 文件 | 题目数 | 用途 |
|-----|------|--------|------|
| Small | `longmemeval_s_cleaned.json` | 500 | 快速评测、开发调试 |
| Medium | `longmemeval_m_cleaned.json` | 更多 | 完整评测 |
| Oracle | `longmemeval_oracle.json` | — | 包含答案 session 原文，供误差分析 |

---

## 8. 与 LoCoMo 的核心差异

理解两个数据集的差异对于正确使用 HiMGA 接口至关重要：

| 维度 | LoCoMo | LongMemEval |
|------|--------|-------------|
| **对话类型** | 两个真实用户的闲聊 | 用户与 AI 助手的功能性对话 |
| **样本粒度** | 一个完整对话 + 多个 QA | 一道题 + 对应的 haystack |
| **QA 数/样本** | ~199 个（多 QA 共享一个对话） | 恰好 1 个（每题有独立 haystack） |
| **Session 数** | 19-32（同一对对用户）| 38-62（来自不同话题的混杂 session）|
| **说话者** | 真实姓名（"Caroline"）| `"user"` / `"assistant"` |
| **Turn ID** | 有 `dia_id`（"D1:3"）| 无 |
| **时间格式** | `"1:56 pm on 8 May, 2023"` | `"2023/05/20 (Sat) 02:21"` |
| **证据粒度** | Turn 级 | Session 级 |
| **问题语气** | 第三人称（"When did Caroline..."）| 第一人称（"What degree did I graduate with?"） |
| **图片内容** | 有（blip_caption 处理）| 无 |
| **辅助字段** | event_summary / observation / session_summary | question_date |

**关键设计差异**：LongMemEval 中，"haystack" 是刻意构造的干扰环境——大多数 session 与答案无关，只有少数 `answer_session_ids` 中的 session 包含答案。这模拟了真实助手记忆场景中的"大海捞针"挑战。

---

## 9. HiMGA 接口映射

### 9.1 原始字段 → HiMGA 字段完整对照

| 原始字段 | HiMGA 字段 | 类型变化 |
|---------|-----------|---------|
| `item["question_id"]` | `Sample.sample_id` = `QAPair.question_id` | — |
| `item["question_type"]`（连字符）| `QAPair.question_type` | str → QuestionType |
| `item["question"]` | `QAPair.question` | — |
| `str(item["answer"])` | `QAPair.answer` | any → str |
| `item["answer_session_ids"]` | `QAPair.evidence.session_ids` | — |
| `item["haystack_session_ids"][i]` | `Sample.sessions[i].session_id` | — |
| `item["haystack_dates"][i]` | `Sample.sessions[i].date_str` | — |
| `item["haystack_sessions"][i]` | `Sample.sessions[i].messages` | list[dict] → list[Message] |
| `message["role"]` | `Message.role` | — |
| `message["content"]` | `Message.content` | — |
| `item["question_date"]` | `Sample.raw["question_date"]` | — |

**不映射的字段**：`haystack_session_ids` 和 `haystack_dates` 已通过并行 zip 赋给各 Session，原始值通过 `QAPair.raw` 可访问。

### 9.2 HiMGA 的特殊处理

**question_type 格式兼容**：数据集原始使用连字符（`"single-session-user"`），HiMGA loader 同时支持下划线格式（`"single_session_user"`），两者均映射到相同的 `QuestionType`：

```text
# 两种格式均可正确解析
"single-session-user"  → QuestionType.SINGLE_SESSION_USER
"single_session_user"  → QuestionType.SINGLE_SESSION_USER
```

**未知类型 fallback**：遇到 `_QTYPE_MAP` 中未收录的 `question_type` 字符串时，fallback 为 `QuestionType.SINGLE_SESSION_USER`，并不报错。

**一问一 Sample**：每道题映射为一个独立 `Sample`，`sample_id = question_id`，`qa_pairs` 列表中只有一个元素。这与 LoCoMo 的"一个 Sample 含多个 QA"形成对比，但评测循环代码完全对称。

### 9.3 加载代码

```python
from himga.data import load_dataset

# 方式一：统一入口（自动下载缓存）
samples = load_dataset("longmemeval")

# 方式二：直接传路径
from pathlib import Path
from himga.data import load_longmemeval
samples = load_longmemeval(Path("/Volumes/itgz/datasets/longmemeval"))

print(f"{len(samples)} samples loaded")
# → 500 samples loaded（_s 版）
```

### 9.4 加载后的数据结构

```python
sample = samples[0]

print(sample.sample_id)        # "e47becba"
print(sample.dataset)          # "longmemeval"
print(sample.speaker_a)        # None（LongMemEval 无说话者姓名）
print(len(sample.sessions))    # 53
print(len(sample.qa_pairs))    # 1（每 sample 恰好 1 个 QA）

sess = sample.sessions[0]
print(sess.session_id)   # "sharegpt_yywfIrx_0"
print(sess.date_str)     # "2023/05/20 (Sat) 02:21"
print(sess.title)        # None
print(len(sess.messages))  # 依 session 长度而定

msg = sess.messages[0]
print(msg.role)      # "user"
print(msg.content)   # "The farmer needs to transport a fox..."
print(msg.turn_id)   # None（LongMemEval 无 turn ID）

qa = sample.qa_pairs[0]
print(qa.question_id)     # "e47becba"
print(qa.question)        # "What degree did I graduate with?"
print(qa.answer)          # "'Business Administration'"
print(qa.question_type)   # QuestionType.SINGLE_SESSION_USER
print(qa.evidence.session_ids)  # ["answer_280352e9"]

# 访问 question_date
print(sample.raw["question_date"])  # "2023/05/30 (Tue) 23:40"
```

---

## 10. 常见使用模式

### 找到包含答案的 session

```python
def find_answer_sessions(sample):
    qa = sample.qa_pairs[0]
    answer_ids = set(qa.evidence.session_ids)
    return [s for s in sample.sessions if s.session_id in answer_ids]

answer_sessions = find_answer_sessions(samples[0])
for sess in answer_sessions:
    print(f"[{sess.session_id}] @ {sess.date_str}")
    for msg in sess.messages:
        print(f"  {msg.role}: {msg.content[:80]}")
```

### 区分答案 session 和噪声 session

```python
def split_sessions(sample):
    qa = sample.qa_pairs[0]
    answer_ids = set(qa.evidence.session_ids)
    answer_sess = [s for s in sample.sessions if s.session_id in answer_ids]
    noise_sess  = [s for s in sample.sessions if s.session_id not in answer_ids]
    return answer_sess, noise_sess

answer_sess, noise_sess = split_sessions(samples[0])
print(f"Answer sessions: {len(answer_sess)}, Noise sessions: {len(noise_sess)}")
# → Answer sessions: 1, Noise sessions: 52
```

### 按问题类型筛选

```python
from himga.data import QuestionType

def get_by_type(samples, qtype):
    return [s for s in samples if s.qa_pairs[0].question_type == qtype]

multi = get_by_type(samples, QuestionType.MULTI_SESSION)
temporal = get_by_type(samples, QuestionType.TEMPORAL_REASONING)
print(f"Multi-session: {len(multi)}, Temporal: {len(temporal)}")
```

### 统计 haystack 规模

```python
import statistics

sizes = [len(s.sessions) for s in samples]
total_msgs = [sum(len(sess.messages) for sess in s.sessions) for s in samples]

print(f"Haystack sessions: min={min(sizes)}, max={max(sizes)}, avg={statistics.mean(sizes):.1f}")
print(f"Total messages:    min={min(total_msgs)}, max={max(total_msgs)}, avg={statistics.mean(total_msgs):.1f}")
```

### 评测循环（典型用法）

```python
from himga.data import load_dataset, QuestionType

samples = load_dataset("longmemeval")

for sample in samples:
    qa = sample.qa_pairs[0]   # LongMemEval 每 sample 恰好 1 个 QA

    # 1. 将 haystack sessions 按时间顺序注入记忆系统
    for sess in sample.sessions:
        for msg in sess.messages:
            memory.ingest(msg)

    # 2. 查询
    context = memory.retrieve(qa.question)
    response = llm.generate(qa.question, context)

    # 3. 评测
    evaluate(
        question_type=qa.question_type,
        ground_truth=qa.answer,
        prediction=response,
    )

    memory.reset()
```
