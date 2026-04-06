# Examples

Runnable reference implementations for the patterns described in this skill.

## Install

```bash
pip install langgraph langchain-anthropic pydantic
export ANTHROPIC_API_KEY=your_key_here
```

## Examples

### `simple_react_agent/`

A minimal but production-hardened single agent. Demonstrates:

- Tool definition with Pydantic schema validation
- `GuardedToolNode` with ALLOW/DENY permission rules and audit logging
- `AgentTelemetry` callback for full execution tracing
- Max iteration guard to prevent infinite loops
- Session persistence with `InMemorySaver` (swap to `SqliteSaver` / `PostgresSaver` for production)

```bash
python examples/simple_react_agent/agent.py
```

### `multi_agent_router/`

A multi-agent system with intent routing, specialist subagents, and clarification flow. Demonstrates:

- Router node with structured LLM output and confidence threshold
- Prompt injection / adversarial input sanitization
- Two specialist agents (billing, technical) each with scoped tool sets
- Clarification node with round limit (no infinite clarification loops)
- Telemetry propagation from parent to subagents

```bash
python examples/multi_agent_router/agent.py
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | required | Your Anthropic API key |
| `AGENT_MODEL_ID` | `claude-sonnet-4-6` | Pinned model version |
| `MAX_OUTPUT_TOKENS` | `4096` | Max tokens per LLM response |
| `MAX_ITERATIONS` | `10` | Max agent loop iterations before force-stop |
| `MODEL_PRICING_JSON` | `{}` | JSON map of model → pricing rates (see observability-layer.md) |

## Extending These Examples

- **Add persistence**: swap `InMemorySaver` for `SqliteSaver` or `PostgresSaver` (see `persistence-layer.md`)
- **Add more tools**: follow the Pydantic schema pattern and add `PermissionRule` entries
- **Add cost tracking**: integrate `SessionCostTracker` from `observability-layer.md`
- **Add resilience**: wrap tools with `CircuitBreaker` from `resilience-layer.md`
