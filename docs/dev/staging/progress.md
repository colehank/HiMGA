# HiMGA 基座平台开发进度

> 路径：`docs/dev/staging/`
> 更新：2026-04-22

---

## 模块状态总览

| 模块 | 路径 | 状态 | 说明 |
|------|------|------|------|
| `data` | `src/himga/data/` | ✅ 完成 | schema、loader×2、temporal、139 个测试全通过 |
| `memory` | `src/himga/memory/` | ✅ 完成 | BaseMemory 抽象接口 + NullMemory；19 个测试全通过 |
| `llm` | `src/himga/llm/` | ✅ 完成 | BaseLLMClient + AnthropicClient + get_client()；20 个测试全通过 |
| `agent` | `src/himga/agent/` | ✅ 完成 | BaseAgent（ingest_sample / answer / _build_messages）；24 个测试全通过 |
| `eval` | `src/himga/eval/` | ✅ 完成 | runner / judge（含缓存） / metrics（F1 + judge score）；51 个测试全通过 |

**基座完成**：254 个测试全部通过（`uv run pytest -m "not integration" -q`）。

---

## 下一阶段

基座平台就绪，下一步进入 HiMGA 核心算法开发：

```
eval ✅ → HiMGA Memory（多图/超图结构）→ 对比 MAGMA baseline
```

每步继续遵循 TDD：设计接口 → 编写测试 → 实现代码 → 测试通过后提交。
