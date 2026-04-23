# agent 模块设计

> 状态：⏳ 待开发
> 路径：`src/himga/agent/`

---

## 职责

将 `BaseMemory` 和 `BaseLLMClient` 组合为可评测的 Agent：负责把会话历史注入记忆系统，并在给定问题时检索记忆、构造 prompt、调用 LLM 生成答案。

---

## 文件结构

```
agent/
├── __init__.py    # 导出 BaseAgent
└── base.py        # BaseAgent 实现
```

---

## 接口定义（base.py）

```python
from himga.data.schema import Sample, Session, Message
from himga.memory.base import BaseMemory
from himga.llm.client import BaseLLMClient

class BaseAgent:
    def __init__(self, memory: BaseMemory, llm: BaseLLMClient):
        self.memory = memory
        self.llm = llm

    def ingest_sample(self, sample: Sample) -> None:
        """将 sample 的所有 session 按时间顺序注入记忆系统。"""
        for session in sample.sessions:
            for message in session.messages:
                self.memory.ingest(message, session)

    def answer(self, question: str) -> str:
        """检索记忆 → 构造 prompt → LLM 生成答案。"""
        context = self.memory.retrieve(question)
        messages = self._build_messages(question, context)
        return self.llm.chat(messages)

    def _build_messages(self, question: str, context: str) -> list[dict]:
        system = (
            "You are a helpful assistant with access to past conversation history. "
            "Answer the question based on the provided context."
        )
        user = f"Context:\n{context}\n\nQuestion: {question}" if context else question
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
```

---

## 关键设计决策

| 决策 | 理由 |
|------|------|
| `ingest_sample` 而非 `ingest_session` | eval runner 以 Sample 为单位驱动，接口粒度对齐 |
| sessions 按 `sample.sessions` 原有顺序 ingest | 两个数据集的 sessions 在 loader 阶段已按时间排序 |
| `_build_messages` 可覆写 | 子类（如 HiMGA Agent）可定制 prompt 结构，不必修改基类 |
| agent 不持有 `Sample`，不负责 `reset` | 职责分离：reset 由 eval runner 在每个 sample 前调用 |

---

## 测试要点

- 注入 `NullMemory + MockLLMClient`，`answer` 返回字符串且不抛异常
- `ingest_sample` 调用次数 = 所有 sessions 的 messages 总数（通过 spy 验证）
- context 为空时 prompt 不含 "Context:" 前缀
- context 非空时 prompt 包含 context 内容
