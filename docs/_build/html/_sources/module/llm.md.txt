# `himga.llm` — API 设计与使用说明

> 版本：基于当前实现（2026-04-22）
> 模块路径：`src/himga/llm/`

---

## 目录

1. [概述](#1-概述)
2. [模块结构](#2-模块结构)
3. [类型参考](#3-类型参考)
   - [BaseLLMClient](#31-basellmclient)
   - [AnthropicClient](#32-anthropicclient)
4. [工厂函数](#4-工厂函数)
   - [get_client](#41-get_client)
5. [消息格式说明](#5-消息格式说明)
6. [使用示例](#6-使用示例)
   - [基础调用](#61-基础调用)
   - [自定义模型与参数](#62-自定义模型与参数)
   - [实现 MockLLMClient（测试用）](#63-实现-mockllmclient测试用)
   - [实现自定义 LLM 接入](#64-实现自定义-llm-接入)
7. [设计决策说明](#7-设计决策说明)

---

## 1. 概述

`himga.llm` 封装 LLM API 调用，提供统一的 `chat` 接口。`agent`、`eval/judge` 等上层模块只依赖 `BaseLLMClient`，不感知具体 provider，使替换模型或接入新 provider 无需修改上层代码。

**目前提供的实现**

| 实现 | Provider | 说明 |
|------|---------|------|
| `AnthropicClient` | Anthropic | 调用 Anthropic Messages API，默认模型 `claude-sonnet-4-6` |

---

## 2. 模块结构

```
src/himga/llm/
├── __init__.py    # 导出 BaseLLMClient, AnthropicClient, get_client
└── client.py      # 抽象接口 + 实现 + 工厂函数
```

**公共导入路径**

```python
from himga.llm import BaseLLMClient, AnthropicClient, get_client
```

---

## 3. 类型参考

### 3.1 `BaseLLMClient`

```python
class BaseLLMClient(ABC)
```

所有 LLM provider 的抽象基类，仅定义一个方法：`chat`。

#### 方法


##### `chat(messages, *, model, max_tokens, temperature)`

```python
@abstractmethod
def chat(
    self,
    messages: list[dict],
    *,
    model: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> str
```

发送对话请求，返回 assistant 回复文本。

**参数**

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `messages` | `list[dict]` | （必填） | OpenAI 格式消息列表，详见 [消息格式说明](#5-消息格式说明) |
| `model` | `str \| None` | `None` | 覆盖 client 默认模型。`None` 时使用 client 初始化时设定的模型 |
| `max_tokens` | `int` | `1024` | 最大生成 token 数 |
| `temperature` | `float` | `0.0` | 采样温度。评测场景默认 `0.0` 保证确定性 |

**返回**

`str`：assistant 回复文本，去除首尾空白。

---

### 3.2 `AnthropicClient`

```python
class AnthropicClient(BaseLLMClient)
```

基于 Anthropic Python SDK 的 LLM 客户端。

**依赖**：`anthropic` 包（已在 `pyproject.toml` 中声明），以及环境变量 `ANTHROPIC_API_KEY`。

#### 构造函数

```python
def __init__(self, model: str = "claude-sonnet-4-6") -> None
```

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `model` | `str` | `"claude-sonnet-4-6"` | 默认模型 ID，可在调用 `chat` 时按次覆盖 |

#### 实现细节

`AnthropicClient` 自动处理 OpenAI 格式与 Anthropic API 格式之间的差异：消息列表中 `role="system"` 的条目被提取并作为 Anthropic Messages API 的顶级 `system` 参数传入，其余消息保持原顺序。

```python
# 内部处理示例：
# 输入：[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
# → Anthropic API 调用：
#     system = "..."
#     messages = [{"role": "user", "content": "..."}]
```

#### 可用模型速查

| 模型 ID | 说明 |
|--------|------|
| `claude-sonnet-4-6` | 默认，均衡性能与速度 |
| `claude-haiku-4-5-20251001` | 快速且廉价，适合 judge 批量调用 |
| `claude-opus-4-7` | 最强能力，适合复杂推理任务 |

---

## 4. 工厂函数

### 4.1 `get_client`

```python
def get_client(provider: str | None = None) -> BaseLLMClient
```

根据 provider 名称返回对应的 LLM client 实例。

**参数**

| 参数 | 类型 | 说明 |
|-----|------|------|
| `provider` | `str \| None` | Provider 名称。`None` 时从环境变量 `LLM_PROVIDER` 读取，仍为空则默认 `"anthropic"` |

**支持的 provider**

| provider 值 | 返回类型 |
|------------|---------|
| `"anthropic"` | `AnthropicClient` |

**异常**

| 异常 | 触发条件 |
|-----|---------|
| `ValueError: Unknown provider: '...'` | provider 不在支持列表中 |

**优先级**：显式参数 > `LLM_PROVIDER` 环境变量 > 默认值 `"anthropic"`

```python
# 三种等价的获取方式：
client = get_client("anthropic")                  # 显式指定
os.environ["LLM_PROVIDER"] = "anthropic"
client = get_client()                              # 读环境变量
client = get_client()                              # 无环境变量时默认 anthropic
```

---

## 5. 消息格式说明

`chat` 接受 OpenAI 格式的消息列表：

```python
messages = [
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "user",      "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."},
    {"role": "user",      "content": "And Germany?"},
]
```

**字段说明**

| 字段 | 类型 | 说明 |
|-----|------|------|
| `role` | `str` | `"system"` / `"user"` / `"assistant"` |
| `content` | `str` | 消息文本内容 |

**注意事项**

- `system` 消息可放在列表任意位置，`AnthropicClient` 会自动提取并处理
- 若消息列表为空，`AnthropicClient` 将发送不含任何 user 消息的请求（API 可能报错）
- `BaseAgent._build_messages` 生成的消息列表始终符合此格式（一条 system + 一条 user）

---

## 6. 使用示例

### 6.1 基础调用

```python
from himga.llm import get_client

llm = get_client()  # 默认 AnthropicClient

response = llm.chat([
    {"role": "system",  "content": "You are a helpful assistant."},
    {"role": "user",    "content": "What is 2 + 2?"},
])
print(response)  # "4"
```

---

### 6.2 自定义模型与参数

```python
from himga.llm import AnthropicClient

# 初始化时设置默认模型
llm = AnthropicClient(model="claude-haiku-4-5-20251001")

# 全局默认参数调用
response = llm.chat([{"role": "user", "content": "Hi"}])

# 单次覆盖参数
response = llm.chat(
    [{"role": "user", "content": "Write a poem about memory."}],
    model="claude-opus-4-7",    # 本次用更强的模型
    max_tokens=512,
    temperature=0.7,             # 适当提高创造性
)
```

---

### 6.3 实现 MockLLMClient（测试用）

测试中不产生真实 API 调用：

```python
from himga.llm import BaseLLMClient

class MockLLMClient(BaseLLMClient):
    """返回固定响应的测试 stub。"""

    def __init__(self, response: str = "mock answer"):
        self._response = response
        self.call_count = 0
        self.last_messages: list[dict] | None = None

    def chat(self, messages, *, model=None, max_tokens=1024, temperature=0.0) -> str:
        self.call_count += 1
        self.last_messages = messages
        return self._response
```

用法：

```python
llm = MockLLMClient(response="Paris")
assert llm.chat([{"role": "user", "content": "Capital of France?"}]) == "Paris"
assert llm.call_count == 1
```

---

### 6.4 实现自定义 LLM 接入

接入 OpenAI 或本地模型时，继承 `BaseLLMClient` 并实现 `chat`：

```python
from himga.llm import BaseLLMClient

class OpenAIClient(BaseLLMClient):
    def __init__(self, model: str = "gpt-4o"):
        from openai import OpenAI
        self._client = OpenAI()
        self._default_model = model

    def chat(self, messages, *, model=None, max_tokens=1024, temperature=0.0) -> str:
        resp = self._client.chat.completions.create(
            model=model or self._default_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content
```

注册到 `get_client`（可在项目内 monkey-patch 或扩展 `get_client` 函数）：

```python
# 无需修改 himga.llm，直接实例化使用
from himga.agent import BaseAgent
from himga.memory import NullMemory

agent = BaseAgent(memory=NullMemory(), llm=OpenAIClient())
```

---

## 7. 设计决策说明

### 为什么采用 OpenAI 格式的 messages？

OpenAI 格式已成为业界通用标准，Anthropic、Cohere、本地模型（Ollama）等均提供兼容适配层。`BaseAgent` 和 `judge.py` 生成的 prompt 无需感知 provider 差异，降低了上层代码的耦合度。

### 为什么 `temperature=0.0` 为默认值？

评测场景要求结果可复现（方便与论文数值对比）。`judge_answer` 在批量调用时也使用 `0.0` 保证判断一致性。对需要随机性的场景（如采样多个候选答案），调用方可按次覆盖。

### 为什么不在接口层做重试？

重试属于基础设施关注点（网络可靠性、退避策略），不应耦合进 LLM 抽象。调用方可用 `tenacity` 等库在外部包装，或在具体 client 内部按需实现。

### 为什么 `AnthropicClient.__init__` 延迟导入 `anthropic`？

若用户使用其他 provider（如未来的 OpenAIClient），不应因未安装 `anthropic` 包而在 import 阶段报错。延迟导入（在 `__init__` 内部 `import anthropic`）使模块在不安装该包时仍可被其他代码引用。
