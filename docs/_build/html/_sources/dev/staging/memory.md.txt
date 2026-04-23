# memory 模块设计

> 状态：⏳ 待开发
> 路径：`src/himga/memory/`

---

## 职责

定义记忆系统的统一抽象接口。所有记忆系统变体（NullMemory、MAGMA 复现、HiMGA 各层级）均实现此接口，eval 层与具体实现完全解耦。

---

## 文件结构

```
memory/
├── __init__.py    # 导出 BaseMemory, NullMemory
├── base.py        # 抽象接口
└── null.py        # NullMemory（pipeline 贯通用）
```

---

## 接口定义（base.py）

```python
from abc import ABC, abstractmethod
from himga.data.schema import Message, Session

class BaseMemory(ABC):

    @abstractmethod
    def ingest(self, message: Message, session: Session) -> None:
        """将一条消息写入记忆系统。
        session 提供时间戳等上下文，message 提供内容。
        """

    @abstractmethod
    def retrieve(self, query: str) -> str:
        """检索与 query 相关的记忆，返回组装好的上下文字符串。"""

    @abstractmethod
    def reset(self) -> None:
        """清空当前记忆，准备处理下一个 Sample。"""
```

---

## NullMemory（null.py）

```python
class NullMemory(BaseMemory):
    def ingest(self, message, session): pass
    def retrieve(self, query): return ""
    def reset(self): pass
```

作用：在真实记忆系统就绪前跑通整条流水线。用 NullMemory 跑出的评测数字是 baseline（无记忆时 LLM 凭自身知识能答对多少），有参考价值。

---

## 关键设计决策

| 决策 | 理由 |
|------|------|
| `ingest` 以 `Message` 为粒度 | 对齐 LongMemEval 的消息级处理；LoCoMo 可按 turn 逐条 ingest |
| `session` 作为 `ingest` 的附加参数 | 记忆系统可能需要时间戳排序，不强制使用但要传入 |
| `retrieve` 返回 `str` | 最大兼容性；复杂系统内部结构化但对外只暴露可直接注入 prompt 的文本 |
| `reset` 语义为"清空当前 sample" | eval runner 在每个 sample 前调用，保证样本间隔离 |

---

## 测试要点

- `NullMemory.retrieve` 任意输入都返回空字符串
- `NullMemory.reset` 调用后状态不变（本就无状态）
- `ingest` → `retrieve` 的基本语义（通过带状态的 stub 实现验证接口契约）
- 多次 `ingest` 后 `reset` → `retrieve` 返回空（隔离验证）
