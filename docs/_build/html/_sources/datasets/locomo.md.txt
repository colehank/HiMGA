# LoCoMo 数据集说明

> 覆盖：数据集背景、原始格式、字段说明、统计信息、HiMGA 接口映射
> 最后更新：2026-04-21

---

## 目录

1. [数据集背景](#1-数据集背景)
2. [文件结构](#2-文件结构)
3. [顶层结构](#3-顶层结构)
4. [conversation 字段详解](#4-conversation-字段详解)
5. [qa 字段详解](#5-qa-字段详解)
6. [辅助字段详解](#6-辅助字段详解)
7. [图片消息](#7-图片消息)
8. [数据统计](#8-数据统计)
9. [HiMGA 接口映射](#9-himga-接口映射)
10. [常见使用模式](#10-常见使用模式)

---

## 1. 数据集背景

**LoCoMo**（Long Conversation Memory）是一个评测 AI 系统长期对话记忆能力的基准数据集，由 Snap Research 发布。

**核心特点**：

- **真实长对话**：每个样本是两个真实用户之间跨越数月的多会话对话，平均包含 25+ 个 session，500+ 条发言
- **多模态**：对话中包含大量图片分享，图片通过 BLIP 模型自动生成的标题（`blip_caption`）转为文本
- **丰富标注**：每个样本配有人工标注的 QA pair、事件摘要（event_summary）、观察记录（observation）和会话摘要（session_summary）
- **多维度问题**：QA 覆盖单跳事实、时间推理、多跳推理、开放域和对抗性问题五种类型

**评测任务**：给定完整对话历史，回答关于对话内容的自然语言问题。

---

## 2. 文件结构

```
$DATASETS_ROOT/locomo/
├── locomo10.json          # 主评测文件，10 个完整样本
├── msc_personas_all.json  # MSC 人物设定数据（train/valid/test）
└── multimodal_dialog/     # 多模态对话原始数据目录
```

**`locomo10.json`**：评测的核心文件，是一个 JSON **列表**，包含 10 个样本。

```python
import json
data = json.load(open("locomo10.json"))
# data: list, len=10
# data[0]: dict，一个完整样本
```

---

## 3. 顶层结构

每个样本是一个 dict，包含以下顶层字段：

```text
{
  "sample_id":       str,
  "conversation":    { ... },
  "qa":              [ ... ],
  "event_summary":   { ... },
  "observation":     { ... },
  "session_summary": { ... }
}
```

| 字段 | 类型 | 说明 |
|-----|------|------|
| `sample_id` | `str` | 样本唯一 ID，如 `"conv-26"` |
| `conversation` | `dict` | 完整对话历史，含多个 session（详见第 4 节） |
| `qa` | `list` | QA pair 列表（详见第 5 节） |
| `event_summary` | `dict` | 按 session 和说话者整理的事件列表（详见第 6 节） |
| `observation` | `dict` | 按 session 整理的结构化观察（含证据引用） |
| `session_summary` | `dict` | 每个 session 的自然语言摘要 |

---

## 4. `conversation` 字段详解

`conversation` 是整个样本的对话历史主体，结构如下：

```text
{
  "speaker_a": "Caroline",
  "speaker_b": "Melanie",
  "session_1_date_time": "1:56 pm on 8 May, 2023",
  "session_1": [ ... ],
  "session_2_date_time": "1:14 pm on 25 May, 2023",
  "session_2": [ ... ],
  ...
  "session_19_date_time": "...",
  "session_19": [ ... ]
}
```

### 4.1 说话者字段

| 字段 | 说明 |
|-----|------|
| `speaker_a` | 主说话者姓名（如 `"Caroline"`） |
| `speaker_b` | 次说话者姓名（如 `"Melanie"`） |

两人是现实中的朋友，对话跨越数月，内容涵盖日常生活、情感、工作、爱好等。

### 4.2 Session 字段命名规则

Session 的字段以键名编号标识，**没有嵌套容器**，直接平铺在 `conversation` 字典中：

- `session_N` — 第 N 个会话的 turn 列表（N 从 1 开始）
- `session_N_date_time` — 第 N 个会话的时间字符串

遍历所有 session 需要按规律过滤键名：

```python
session_keys = sorted(
    [k for k in conv if k.startswith("session_") and not k.endswith("_date_time")
     and isinstance(conv[k], list)],
    key=lambda k: int(k.split("_")[1])
)
```

### 4.3 时间格式

```
"1:56 pm on 8 May, 2023"
"1:14 pm on 25 May, 2023"
"7:55 pm on 9 June, 2023"
```

格式为 `"H:MM am/pm on D Month, YYYY"`。

### 4.4 Turn 结构

每个 session 是一个 turn 列表，每条 turn 是一个 dict：

**普通文本 turn**：
```json
{
  "speaker": "Caroline",
  "dia_id":  "D1:1",
  "text":    "Hey Mel! Good to see you! How have you been?"
}
```

**含图片的 turn**（详见第 7 节）：
```json
{
  "speaker":      "Caroline",
  "dia_id":       "D1:5",
  "text":         "The transgender stories were so inspiring!",
  "img_url":      ["https://i.redd.it/l7hozpetnhlb1.jpg"],
  "blip_caption": "a photo of a dog walking past a wall with a painting of a woman",
  "query":        "transgender pride flag mural"
}
```

| 字段 | 类型 | 必有 | 说明 |
|-----|------|------|------|
| `speaker` | `str` | ✓ | 说话者姓名，与 `speaker_a`/`speaker_b` 对应 |
| `dia_id` | `str` | ✓ | Turn 的唯一标识，格式 `"D{N}:{M}"`，N=session编号，M=行号。如 `"D1:3"` 表示第1个session第3行 |
| `text` | `str` | ✓ | 发言文本（图片 turn 中可能为空字符串） |
| `img_url` | `list[str]` | — | 图片 URL 列表（图片 turn 才有） |
| `blip_caption` | `str` | — | BLIP 模型生成的图片描述（图片 turn 才有） |
| `query` | `str` | — | 图片搜索时使用的查询词（图片 turn 才有） |

#### `dia_id` 详解

`dia_id` 是跨 session 全局唯一的 turn 标识，与 QA 的 `evidence` 字段一一对应：

```
D1:3  →  session_1 的第 3 条 turn
D2:7  →  session_2 的第 7 条 turn
```

在同一个样本内，`dia_id` 中的 session 编号与 `session_N` 的 N 一致。

---

## 5. `qa` 字段详解

`qa` 是一个列表，每个元素是一个问答对：

### 5.1 普通 QA（category 1-4）

```json
{
  "question": "What did Caroline research?",
  "answer":   "Adoption agencies",
  "evidence": ["D2:8"],
  "category": 1
}
```

```json
{
  "question": "When did Melanie paint a sunrise?",
  "answer":   2022,
  "evidence": ["D1:12"],
  "category": 2
}
```

| 字段 | 类型 | 说明 |
|-----|------|------|
| `question` | `str` | 自然语言问题 |
| `answer` | `str` 或 `int` | 标准答案。注意：年份等数字答案为 `int` 类型 |
| `evidence` | `list[str]` | 支持该答案的 turn 的 `dia_id` 列表，可多条 |
| `category` | `int` | 问题类型编号（1-5，见下表） |

### 5.2 对抗性 QA（category=5）

```json
{
  "question":          "What did Caroline realize after her charity race?",
  "evidence":          ["D2:3"],
  "category":          5,
  "adversarial_answer": "self-care is important"
}
```

**category=5 的特殊之处**：

- **没有 `answer` 字段**（或 `answer` 字段不应使用）
- 改为 `adversarial_answer` 字段，内容是对话中**真正可推断的答案**
- 这类问题的设计目的是测试模型是否能正确识别出对话中存在答案，而不被问题的误导性表述欺骗
- HiMGA loader 在解析时自动将 `adversarial_answer` 赋给 `QAPair.answer`

### 5.3 category 编号含义

| category | HiMGA QuestionType | 含义 | 数量（10个样本） |
|----------|-------------------|------|----------------|
| 1 | `SINGLE_HOP` | 单跳事实问题，答案直接在一条 turn 中 | 282 |
| 2 | `TEMPORAL` | 时间推理，涉及"什么时候"、时间计算 | 321 |
| 3 | `MULTI_HOP` | 多跳推理，需要关联多条 turn 才能得出答案 | 96 |
| 4 | `OPEN_DOMAIN` | 开放域问题，需要综合多处信息 | 841 |
| 5 | `ADVERSARIAL` | 对抗性问题，测试模型抗干扰能力 | 446 |

**总计**：10 个样本共 **1986 个 QA pair**，平均每样本 ~199 个。

---

## 6. 辅助字段详解

这三个辅助字段在 HiMGA 中存储于 `Sample.raw`，供高层模块（如 HiMGA 的层次化记忆构建）使用。

### 6.1 `event_summary`

按 session 和说话者整理的事件列表，每个事件是一句话的自然语言描述：

```json
{
  "events_session_1": {
    "Caroline": ["Caroline attends an LGBTQ support group for the first time."],
    "Melanie":  [],
    "date":     "8 May, 2023"
  },
  "events_session_2": {
    "Caroline": ["Caroline participates in a 5K charity race."],
    "Melanie":  ["Melanie starts a new yoga class."],
    "date":     "25 May, 2023"
  }
}
```

键名格式：`events_session_{N}`（注意与 `session_N` 不同）。

每个条目包含：
- `Caroline` / `Melanie`（或对应说话者名）：该人在此 session 中发生的事件列表
- `date`：session 日期字符串

**用途**：为 HiMGA Level 1（片段层）的 Episode 节点提供高质量摘要种子。

### 6.2 `observation`

比 `event_summary` 更细粒度的结构化观察，每条观察都附有对应的 `dia_id` 证据：

```json
{
  "session_1_observation": {
    "Caroline": [
      ["Caroline attended an LGBTQ support group recently and found the transgender stories inspiring.", "D1:3"],
      ["The support group has made Caroline feel accepted and given her courage to embrace herself.", "D1:7"]
    ],
    "Melanie": [
      ["Melanie is currently managing kids and work and finds it overwhelming.", "D1:2"],
      ["Melanie painted a lake sunrise last year which holds special meaning to her.", "D1:14"]
    ]
  }
}
```

键名格式：`session_{N}_observation`。

每条观察是一个 `[观察文本, dia_id]` 二元组，`dia_id` 指向支持该观察的原始 turn。

### 6.3 `session_summary`

每个 session 的一段自然语言摘要：

```json
{
  "session_1_summary": "Caroline and Melanie had a conversation on 8 May 2023 at 1:56 pm. Caroline mentioned that she attended an LGBTQ support group and was inspired by the transgender stories she heard...",
  "session_2_summary": "..."
}
```

键名格式：`session_{N}_summary`。

---

## 7. 图片消息

LoCoMo 的对话中约有 **116 条图片 turn**（仅 locomo10.json），图片通过 BLIP 模型转为文字描述。

### 原始图片 turn 字段

```json
{
  "speaker":      "Caroline",
  "dia_id":       "D1:5",
  "text":         "The transgender stories were so inspiring! I was so happy.",
  "img_url":      ["https://i.redd.it/l7hozpetnhlb1.jpg"],
  "blip_caption": "a photo of a dog walking past a wall with a painting of a woman",
  "query":        "transgender pride flag mural"
}
```

| 字段 | 说明 |
|-----|------|
| `img_url` | 图片原始 URL（可能失效） |
| `blip_caption` | BLIP 自动生成的图片描述，是主要的文字化信息 |
| `query` | 图片的搜索关键词，描述图片的语义主题 |

### HiMGA 的处理方式

HiMGA loader 将 `blip_caption` 内联到 `Message.content` 中：

```
原始 text: "The transgender stories were so inspiring!"
blip_caption: "a photo of a dog walking past a wall with a painting of a woman"

→ Message.content: "[Image: a photo of a dog walking past a wall with a painting of a woman] The transgender stories were so inspiring!"
```

若 `text` 为空则只保留 `[Image: ...]` 部分。`img_url` 和 `query` 不保留到 `Message`（存在 `QAPair.raw` 中如需访问）。

---

## 8. 数据统计

基于 `locomo10.json`（10 个样本）的实测数据：

| 指标 | 数值 |
|-----|------|
| 样本数 | 10 |
| 总 QA pair 数 | 1986 |
| 平均每样本 QA 数 | ~199 |
| 样本 session 数范围 | 19 – 32 |
| 样本 turn 数范围 | 369 – 689 |
| 图片 turn 数 | 116 |

**QA 类型分布**（全部 10 样本）：

| category | 类型 | 数量 | 占比 |
|----------|------|------|------|
| 4 | open_domain | 841 | 42.3% |
| 5 | adversarial | 446 | 22.5% |
| 2 | temporal | 321 | 16.2% |
| 1 | single_hop | 282 | 14.2% |
| 3 | multi_hop | 96 | 4.8% |

**各样本详情**：

| sample_id | sessions | turns | qa |
|-----------|---------|-------|----|
| conv-26 | 19 | 419 | 199 |
| conv-30 | 19 | 369 | 105 |
| conv-41 | 32 | 663 | 193 |
| conv-42 | 29 | 629 | 260 |
| conv-43 | 29 | 680 | 242 |
| conv-44 | 28 | 675 | 158 |
| conv-47 | 31 | 689 | 190 |
| conv-48 | 30 | 681 | 239 |
| conv-49 | 25 | 509 | 196 |
| conv-50 | 30 | 568 | 204 |

---

## 9. HiMGA 接口映射

### 9.1 原始字段 → HiMGA 字段完整对照

| 原始位置 | 原始字段 | HiMGA 字段 | 类型变化 |
|---------|---------|-----------|---------|
| `sample` | `sample_id` | `Sample.sample_id` | — |
| `sample.conversation` | `speaker_a` | `Sample.speaker_a` | — |
| `sample.conversation` | `speaker_b` | `Sample.speaker_b` | — |
| `sample.conversation` | `session_N` | `Sample.sessions[i].messages` | list[dict] → list[Message] |
| `sample.conversation` | `session_N_date_time` | `Sample.sessions[i].date_str` | — |
| `turn` | `speaker` | `Message.role` | — |
| `turn` | `text` (+ blip_caption) | `Message.content` | 内联处理 |
| `turn` | `dia_id` | `Message.turn_id` | — |
| `qa_item` | `question` | `QAPair.question` | — |
| `qa_item` | `answer`（category≠5）| `QAPair.answer` | any → str |
| `qa_item` | `adversarial_answer`（category=5）| `QAPair.answer` | str（直接赋值） |
| `qa_item` | `category` | `QAPair.question_type` | int → QuestionType |
| `qa_item` | `evidence` | `QAPair.evidence.turn_ids` | — |
| `sample` | `event_summary` | `Sample.raw["event_summary"]` | — |
| `sample` | `observation` | `Sample.raw["observation"]` | — |
| `sample` | `session_summary` | `Sample.raw["session_summary"]` | — |

### 9.2 加载代码

```python
from himga.data import load_dataset

# 方式一：统一入口（自动下载缓存）
samples = load_dataset("locomo")

# 方式二：直接传路径
from pathlib import Path
from himga.data import load_locomo
samples = load_locomo(Path("/Volumes/itgz/datasets/locomo"))

print(f"{len(samples)} samples loaded")
# → 10 samples loaded
```

### 9.3 加载后的数据结构

```python
sample = samples[0]

print(sample.sample_id)     # "conv-26"
print(sample.dataset)       # "locomo"
print(sample.speaker_a)     # "Caroline"
print(sample.speaker_b)     # "Melanie"
print(len(sample.sessions)) # 19
print(len(sample.qa_pairs)) # 199

sess = sample.sessions[0]
print(sess.session_id)  # "1"
print(sess.date_str)    # "1:56 pm on 8 May, 2023"
print(len(sess.messages))  # 18

msg = sess.messages[0]
print(msg.role)     # "Caroline"
print(msg.content)  # "Hey Mel! Good to see you! How have you been?"
print(msg.turn_id)  # "D1:1"

qa = sample.qa_pairs[0]
print(qa.question)       # "When did Caroline go to the LGBTQ support group?"
print(qa.answer)         # "7 May 2023"
print(qa.question_type)  # QuestionType.TEMPORAL
print(qa.evidence.turn_ids)  # ["D1:3"]
```

---

## 10. 常见使用模式

### 找到 QA 对应的原始 turn

```python
def find_evidence_turns(sample, qa):
    target = set(qa.evidence.turn_ids)
    return [
        (sess.session_id, msg)
        for sess in sample.sessions
        for msg in sess.messages
        if msg.turn_id in target
    ]

qa = sample.qa_pairs[5]
for sess_id, msg in find_evidence_turns(sample, qa):
    print(f"[Session {sess_id} / {msg.turn_id}] {msg.role}: {msg.content}")
```

### 按 category 筛选 QA

```python
from himga.data import QuestionType

def get_qa_by_type(samples, qtype):
    return [(s, qa) for s in samples for qa in s.qa_pairs
            if qa.question_type == qtype]

temporal = get_qa_by_type(samples, QuestionType.TEMPORAL)
adversarial = get_qa_by_type(samples, QuestionType.ADVERSARIAL)
print(f"Temporal: {len(temporal)}, Adversarial: {len(adversarial)}")
```

### 访问辅助字段（event_summary）

```python
for sess_key, content in sample.raw.get("event_summary", {}).items():
    date = content.get("date", "")
    for speaker in [sample.speaker_a, sample.speaker_b]:
        events = content.get(speaker, [])
        for event in events:
            print(f"[{date}] {speaker}: {event}")
```

### 访问 observation（含证据引用）

```python
for sess_key, content in sample.raw.get("observation", {}).items():
    for speaker, obs_list in content.items():
        for obs_text, dia_id in obs_list:
            print(f"{speaker} @ {dia_id}: {obs_text}")
```

### 统计各类型 QA 数量

```python
from collections import Counter
type_dist = Counter(
    qa.question_type.value
    for s in samples
    for qa in s.qa_pairs
)
for qtype, count in sorted(type_dist.items(), key=lambda x: -x[1]):
    print(f"  {qtype:20s}: {count}")
```
