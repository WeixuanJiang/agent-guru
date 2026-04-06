# Testing Layer

## Rationale

Production agents fail in ways that are fundamentally hard to reproduce: the failure may be 10 tool calls deep, involve a specific combination of tool errors, or only surface under load. Testing agents requires a different approach than testing deterministic functions. **Test the harness, not the LLM.** Mock the LLM to control reasoning; exercise the code paths that make agents safe.

## What to Test

| Target | Why | How |
|--------|-----|-----|
| Tool schemas and execution | Agents call what they're given | Unit test each tool independently |
| Permission rules | Safety must be code-verified | Unit test `GuardedToolNode` rules in isolation |
| Router classification | Misrouting is costly | Test against representative inputs + edge cases |
| Error handling paths | LLM sees `ToolMessage`, not crash | Inject failures; assert correct `ToolMessage` returned |
| Compaction logic | Context overflow crashes sessions | Test at 80% threshold with synthetic message history |
| Session resume | Persistence must survive restart | Kill process mid-session, verify state recovers |
| Cost / iteration guards | Runaway agents cost money | Assert agent stops at `max_iterations` |

## Unit Testing Tools

Test tools in isolation — no LLM needed. Verify schema validation, execution logic, and error paths.

```python
import pytest
from pydantic import ValidationError
from your_agent.tools import write_file, WriteFileInput

def test_write_file_creates_file(tmp_path):
    path = str(tmp_path / "output.txt")
    result = write_file.invoke({"path": path, "content": "hello", "create_dirs": False})
    assert "Successfully wrote" in result
    assert open(path).read() == "hello"

def test_write_file_schema_rejects_missing_path():
    with pytest.raises(ValidationError):
        WriteFileInput(content="hello", create_dirs=False)  # path is required

def test_write_file_creates_parent_dirs(tmp_path):
    path = str(tmp_path / "nested" / "dir" / "file.txt")
    write_file.invoke({"path": path, "content": "data", "create_dirs": True})
    assert open(path).read() == "data"
```

## Testing Permission Rules

Test `GuardedToolNode` rules without running an actual agent. Verify ALLOW, DENY, and ASK classifications for known inputs.

```python
from your_agent.safety import DEVOPS_RULES, PermissionMode
from your_agent.safety import GuardedToolNode  # reuse _check_permission

# Instantiate just to use _check_permission — no tools needed for rule tests
guard = GuardedToolNode(tools=[], rules=DEVOPS_RULES)

@pytest.mark.parametrize("tool_name,args,expected_mode", [
    # Read-only kubectl → always allow
    ("kubectl_get",     {"cmd": "get pods"},               PermissionMode.ALWAYS_ALLOW),
    # Destructive kubectl → must ask
    ("kubectl_delete",  {"cmd": "delete pod foo"},         PermissionMode.ASK),
    # Production namespace → always deny
    ("kubectl_apply",   {"cmd": "apply namespace production"}, PermissionMode.ALWAYS_DENY),
    # Dangerous bash → always deny
    ("bash",            {"command": "rm -rf /"},           PermissionMode.ALWAYS_DENY),
])
def test_permission_rules(tool_name, args, expected_mode):
    mode, reason = guard._check_permission(tool_name, args)
    assert mode == expected_mode, f"Expected {expected_mode}, got {mode}: {reason}"
```

## Mocking the LLM for Agent Tests

Use a mock LLM to control the reasoning path. This lets you test the agent harness — routing, tool calls, guards, state transitions — without paying for LLM calls or dealing with non-determinism.

```python
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, ToolCall

def make_mock_llm(responses: list):
    """
    Returns a mock LLM that yields responses in order.
    Each response is either an AIMessage (final answer) or
    an AIMessage with tool_calls (requests a tool).
    """
    mock = MagicMock()
    mock.invoke.side_effect = responses
    mock.bind_tools = lambda tools, **kw: mock  # support bind_tools pattern
    return mock

def test_agent_stops_after_max_iterations():
    # LLM always wants to call a tool — agent should stop at max_iterations
    tool_call_response = AIMessage(
        content="",
        tool_calls=[ToolCall(name="web_search", args={"query": "test"}, id="tc-1")]
    )
    mock_llm = make_mock_llm([tool_call_response] * 20)

    from your_agent.graph import build_graph
    graph = build_graph(llm=mock_llm, max_iterations=3)
    result = graph.invoke({"messages": [{"role": "user", "content": "search forever"}],
                           "iteration_count": 0, "max_iterations": 3})

    # Agent must have stopped — not called LLM more than max_iterations times
    assert mock_llm.invoke.call_count <= 3 + 1  # +1 for force_stop node

def test_agent_returns_error_as_observation():
    """Tool failure should be returned as ToolMessage, not raise an exception."""
    from your_agent.tools import ResilientToolNode
    from langchain_core.messages import AIMessage, ToolMessage, ToolCall

    failing_tool = MagicMock(side_effect=RuntimeError("network timeout"))
    failing_tool.name = "web_search"

    state = {"messages": [AIMessage(
        content="",
        tool_calls=[ToolCall(name="web_search", args={"query": "test"}, id="tc-1")]
    )]}

    node = ResilientToolNode(tools=[failing_tool])
    result = node(state)

    msgs = result["messages"]
    assert len(msgs) == 1
    assert isinstance(msgs[0], ToolMessage)
    assert "network timeout" in msgs[0].content
```

## Fault Injection

Simulate real failure modes to verify resilience patterns. Test that the agent observes errors, retries correctly, and degrades gracefully.

```python
import asyncio
import pytest
from unittest.mock import AsyncMock, patch

class FlakyTool:
    """Fails the first N calls, then succeeds."""
    def __init__(self, fail_count: int):
        self.call_count = 0
        self.fail_count = fail_count

    async def __call__(self, query: str) -> str:
        self.call_count += 1
        if self.call_count <= self.fail_count:
            raise ConnectionError(f"Simulated failure #{self.call_count}")
        return "success result"

@pytest.mark.asyncio
async def test_retry_succeeds_after_transient_failures():
    from your_agent.resilience import with_retry

    tool = FlakyTool(fail_count=2)
    result = await with_retry(
        lambda: tool("test"),
        max_attempts=3,
        base_delay=0.01   # fast in tests
    )
    assert result == "success result"
    assert tool.call_count == 3

@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_threshold():
    from your_agent.resilience import CircuitBreaker, CircuitState

    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
    always_fail = MagicMock(side_effect=RuntimeError("service down"))

    for _ in range(3):
        with pytest.raises(RuntimeError):
            breaker.call(always_fail)

    assert breaker.state == CircuitState.OPEN

    # Next call should be rejected fast, without hitting the service
    with pytest.raises(RuntimeError, match="Circuit OPEN"):
        breaker.call(always_fail)
    assert always_fail.call_count == 3  # not called again
```

## Replay Testing

Record real agent runs as trace fixtures. Replay them to catch regressions when you change tools, prompts, or routing logic.

```python
import json
from pathlib import Path

def record_run(agent, input_message: str, fixture_path: str):
    """Record a real agent run to a JSON fixture file."""
    from your_agent.telemetry import AgentTelemetry

    events = []
    telemetry = AgentTelemetry(sink=events.append)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": input_message}]},
        config={"callbacks": [telemetry]}
    )
    fixture = {
        "input": input_message,
        "events": events,
        "final_output": result["messages"][-1].content
    }
    Path(fixture_path).write_text(json.dumps(fixture, indent=2))
    return fixture

def test_pr_review_tools_called(agent):
    """Verify that a PR review run calls the expected tools in order."""
    fixture = json.loads(Path("tests/fixtures/pr_review_run.json").read_text())

    tool_calls = [e["tool_name"] for e in fixture["events"] if e.get("type") == "tool_call"]
    assert "read_file" in tool_calls
    assert "grep_codebase" in tool_calls
    # Destructive tools must NOT appear
    assert "delete_file" not in tool_calls
    assert "deploy_service" not in tool_calls
```

## Testing Compaction Logic

Verify that compaction triggers at the right threshold and preserves the most recent messages.

```python
from langchain_core.messages import HumanMessage, AIMessage
from your_agent.memory import should_compact, compact_messages

def make_messages(n: int) -> list:
    """Generate n pairs of human/AI messages, each ~1000 chars."""
    msgs = []
    for i in range(n):
        msgs.append(HumanMessage(content=f"User message {i}: " + "x" * 900))
        msgs.append(AIMessage(content=f"AI response {i}: " + "x" * 900))
    return msgs

def test_compaction_triggers_at_80_percent():
    # ~200 messages × ~1800 chars / 4 chars per token ≈ 90k tokens > 80% of 128k
    msgs = make_messages(200)
    assert should_compact(msgs, model_context_window=128_000)

def test_compaction_preserves_recent_messages(mock_llm):
    msgs = make_messages(50)
    compacted = compact_messages(msgs, llm=mock_llm, keep_recent=10)

    # Most recent 10 non-system messages must be preserved verbatim
    original_recent = [m for m in msgs if not hasattr(m, "type") or m.type != "system"][-10:]
    compacted_non_summary = [m for m in compacted if "[CONVERSATION SUMMARY" not in str(m.content)]
    assert len(compacted_non_summary) == len(original_recent)
```

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Testing with real LLM calls | Mock the LLM — test the harness, not the model |
| No fault injection tests | Explicitly test tool failure, network timeout, rate limit scenarios |
| Permission rules tested only in integration | Unit test `_check_permission` directly — fast and exhaustive |
| No replay fixtures | Record representative runs as fixtures; replay on every CI build |
| Testing final output only | Assert tool call sequence and intermediate states, not just the last message |
| Compaction never tested | Generate synthetic long histories; assert compaction fires at threshold |
| Async tools tested synchronously | Use `pytest-asyncio` and `AsyncMock` for async tool and retry tests |
