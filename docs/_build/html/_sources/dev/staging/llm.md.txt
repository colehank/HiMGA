# llm 模块设计

> 状态：⏳ 待开发
> 路径：`src/himga/llm/`

---

## 职责

封装 LLM API 调用，提供统一的 `chat` 接口。上层（agent、eval/judge）不感知具体 provider。

---

## 文件结构

```
llm/
├── __init__.py    # 导出 BaseLLMClient, AnthropicClient, get_client()
└── client.py      # 抽象接口 + 实现
```

---

## 接口定义

```python
from abc import ABC, abstractmethod

class BaseLLMClient(ABC):

    @abstractmethod
    def chat(
        self,
        messages: list[dict],   # OpenAI 格式：[{"role": ..., "content": ...}]
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """调用 LLM，返回 assistant 回复文本。"""
```

---

## AnthropicClient 实现

```python
class AnthropicClient(BaseLLMClient):
    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic
        self._client = anthropic.Anthropic()
        self._default_model = model

    def chat(self, messages, *, model=None, max_tokens=1024, temperature=0.0) -> str:
        resp = self._client.messages.create(
            model=model or self._default_model,
            max_tokens=max_tokens,
            messages=messages,
        )
        return resp.content[0].text
```

---

## 工厂函数

```python
def get_client(provider: str | None = None) -> BaseLLMClient:
    """根据环境变量 LLM_PROVIDER 或参数返回对应 client。"""
    p = provider or os.getenv("LLM_PROVIDER", "anthropic")
    if p == "anthropic":
        return AnthropicClient()
    raise ValueError(f"Unknown provider: {p!r}")
```

---

## 关键设计决策

| 决策 | 理由 |
|------|------|
| messages 采用 OpenAI 格式 | 业界通用，Anthropic SDK 同样接受此格式（system 消息单独处理） |
| `temperature=0.0` 为默认 | 评测场景要求确定性；judge 调用时可按需覆盖 |
| 不在接口层做重试 | 重试逻辑属于基础设施，由调用方或中间件处理 |

---

## 测试要点

- 注入 `MockLLMClient`（返回固定字符串），agent 和 eval 测试不产生真实 API 调用
- `AnthropicClient.chat` 集成测试标记为 `@pytest.mark.integration`，CI 默认跳过
