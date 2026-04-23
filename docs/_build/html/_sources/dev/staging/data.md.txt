# data 模块设计

> 状态：✅ 完成
> 路径：`src/himga/data/`

---

## 职责

将 LoCoMo 和 LongMemEval 两个原始数据集解析为统一的内存对象，供上层模块（agent、eval）无差别使用。

---

## 文件结构

```
data/
├── schema.py          # 统一数据结构定义
├── temporal.py        # 日期字符串 → datetime 解析
└── loaders/
    ├── __init__.py    # load_dataset(name, limit, sample_ids)
    ├── locomo.py      # LoCoMo JSON → Sample
    └── longmemeval.py # LongMemEval JSON → Sample（ijson 流式）
```

---

## 核心类型（schema.py）

```
Message       role / content / turn_id / date_str
Session       session_id / messages / date_str / date(datetime)
EvidenceRef   turn_ids / session_ids
QAPair        question_id / question / answer(str) / question_type / evidence / raw
Sample        sample_id / dataset / sessions / qa_pairs /
              speaker_a / speaker_b / question_date(datetime) / raw
QuestionType  LoCoMo×5 + LongMemEval×6 共 11 种
```

---

## 关键设计决策

| 决策 | 理由 |
|------|------|
| `answer` 统一为 `str` | LoCoMo 年份答案为 int，LME multi-session 答案为 int，统一后 eval 层无需分支 |
| `Session.date` 为 `datetime` | 两种格式（am/pm vs 周几）由 `temporal.py` 统一解析，上层只见 datetime |
| LME 用 `ijson` 流式解析 | `_s` 版 277 MB `json.load` 需 8s；`limit=10` 时 ijson 只读开头降至 1.5s |
| `msc_personas_all.json` 过滤 | LoCoMo 目录含非样本 JSON，loader 按 `"conversation"` 键判断跳过 |
| `Sample.question_date` 仅 LME 有值 | LoCoMo 无提问时间，字段保留为 None |

---

## 公开 API

```python
from himga.data import load_dataset, load_locomo, load_longmemeval
from himga.data import Sample, Session, Message, QAPair, EvidenceRef, QuestionType

# 全量加载
samples = load_dataset("locomo")
samples = load_dataset("longmemeval")

# 快速开发
samples = load_dataset("longmemeval", limit=10)

# 指定样本
samples = load_dataset("longmemeval", sample_ids=["e47becba", "4a3c2b1d"])
```

---

## 测试覆盖

`tests/data/` 共 139 个测试，覆盖：
- schema 默认值、独立实例隔离
- LoCoMo：目录扫描、图片 turn 内联、孤立 date_time、adversarial 字段、aux raw 结构
- LME：流式加载、全 6 种问题类型、int 答案转换、unknown 类型 fallback、多文件加载
- temporal：两种格式、12am/12pm 边界、边界输入
- limit / sample_ids 参数行为
