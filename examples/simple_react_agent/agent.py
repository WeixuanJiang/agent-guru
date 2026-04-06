"""
Simple ReAct Agent — Minimal Production-Hardened Example
=========================================================
Demonstrates:
  - Tool definition with Pydantic schema
  - GuardedToolNode with ALLOW/DENY permission rules
  - AgentTelemetry callback
  - Max iteration guard
  - Session persistence (InMemorySaver for dev; swap to SqliteSaver for prod)

Install:
  pip install langgraph langchain-anthropic pydantic

Run:
  python agent.py
"""

import os
import json
import time
import uuid
import re
from enum import Enum
from typing import TypedDict
from datetime import datetime

from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


# ---------------------------------------------------------------------------
# 1. Model (pin version explicitly)
# ---------------------------------------------------------------------------

MODEL_ID = os.getenv("AGENT_MODEL_ID", "claude-sonnet-4-6")

llm = ChatAnthropic(
    model=MODEL_ID,
    max_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "4096")),
)


# ---------------------------------------------------------------------------
# 2. Tools with Pydantic schemas
# ---------------------------------------------------------------------------

class SearchInput(BaseModel):
    query: str = Field(description="Search query string")

class CalculateInput(BaseModel):
    expression: str = Field(description="Math expression to evaluate, e.g. '2 + 2 * 3'")

@tool(args_schema=SearchInput)
def web_search(query: str) -> str:
    """Search the web for information. Returns a short summary."""
    # In production: call a real search API
    return f"[Simulated result for '{query}']: Found 3 relevant results. Top result: example.com"

@tool(args_schema=CalculateInput)
def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    # Only allow safe characters — prevent code injection
    if not re.match(r"^[\d\s\+\-\*\/\.\(\)]+$", expression):
        return f"Error: Invalid expression '{expression}'. Only basic math is allowed."
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


TOOLS = [web_search, calculate]


# ---------------------------------------------------------------------------
# 3. Permission Guard
# ---------------------------------------------------------------------------

class PermissionMode(Enum):
    ALWAYS_ALLOW = "allow"
    ALWAYS_DENY  = "deny"
    ASK          = "ask"

class PermissionRule:
    def __init__(self, tool_pattern: str, arg_pattern: str = None,
                 mode: PermissionMode = PermissionMode.ASK, reason: str = ""):
        self.tool_re = re.compile(tool_pattern)
        self.arg_re  = re.compile(arg_pattern) if arg_pattern else None
        self.mode    = mode
        self.reason  = reason

    def matches(self, tool_name: str, args: dict) -> bool:
        if not self.tool_re.match(tool_name):
            return False
        if self.arg_re:
            return bool(self.arg_re.search(str(args)))
        return True

RULES = [
    PermissionRule(r"web_search",   mode=PermissionMode.ALWAYS_ALLOW, reason="read-only search"),
    PermissionRule(r"calculate",    mode=PermissionMode.ALWAYS_ALLOW, reason="safe math only"),
]

class GuardedToolNode(ToolNode):
    def __init__(self, tools, rules, audit_log=None):
        super().__init__(tools)
        self.rules = rules
        self.audit_log = audit_log if audit_log is not None else []

    def _check_permission(self, tool_name, args):
        for rule in self.rules:
            if rule.matches(tool_name, args):
                return rule.mode, rule.reason
        return PermissionMode.ASK, "no matching rule — defaulting to ask"

    def __call__(self, state):
        tool_calls = state["messages"][-1].tool_calls
        for tc in tool_calls:
            mode, reason = self._check_permission(tc["name"], tc["args"])
            self.audit_log.append({
                "tool": tc["name"], "args": tc["args"],
                "decision": mode.value, "reason": reason,
                "timestamp": datetime.utcnow().isoformat()
            })
            if mode == PermissionMode.ALWAYS_DENY:
                return {"messages": [ToolMessage(
                    content=f"DENIED: {reason}", tool_call_id=tc["id"]
                )]}
            if mode == PermissionMode.ASK:
                human_decision = interrupt({
                    "type": "tool_approval_request",
                    "tool": tc["name"], "args": tc["args"], "reason": reason,
                    "message": f"Agent wants to call `{tc['name']}`. Approve? (yes/no)"
                })
                if str(human_decision).lower() != "yes":
                    return {"messages": [ToolMessage(
                        content="User denied this tool call.", tool_call_id=tc["id"]
                    )]}
        return super().__call__(state)


# ---------------------------------------------------------------------------
# 4. Telemetry
# ---------------------------------------------------------------------------

class AgentTelemetry(BaseCallbackHandler):
    def __init__(self, session_id: str = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.span_stack = []
        self.events = []

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.span_stack.append({"type": "llm_call", "start": time.monotonic(),
                                 "session_id": self.session_id})

    def on_llm_end(self, response: LLMResult, **kwargs):
        span = self.span_stack.pop() if self.span_stack else {}
        usage = response.llm_output.get("usage", {}) if response.llm_output else {}
        event = {**span, "latency_ms": int((time.monotonic() - span.get("start", 0)) * 1000),
                 "input_tokens": usage.get("input_tokens", 0),
                 "output_tokens": usage.get("output_tokens", 0)}
        self.events.append(event)

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.span_stack.append({"type": "tool_call",
                                 "tool_name": serialized.get("name", "unknown"),
                                 "start": time.monotonic(),
                                 "session_id": self.session_id})

    def on_tool_end(self, output, **kwargs):
        span = self.span_stack.pop() if self.span_stack else {}
        self.events.append({**span,
                             "latency_ms": int((time.monotonic() - span.get("start", 0)) * 1000)})

    def summary(self) -> dict:
        llm_calls = [e for e in self.events if e.get("type") == "llm_call"]
        tool_calls = [e for e in self.events if e.get("type") == "tool_call"]
        return {
            "session_id": self.session_id,
            "llm_turns": len(llm_calls),
            "tool_calls": len(tool_calls),
            "total_input_tokens": sum(e.get("input_tokens", 0) for e in llm_calls),
            "total_output_tokens": sum(e.get("output_tokens", 0) for e in llm_calls),
        }


# ---------------------------------------------------------------------------
# 5. Agent State + Graph
# ---------------------------------------------------------------------------

MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))

class AgentState(TypedDict):
    messages: list
    iteration_count: int

audit_log = []
guarded_tools = GuardedToolNode(tools=TOOLS, rules=RULES, audit_log=audit_log)
bound_llm = llm.bind_tools(TOOLS)

def agent_node(state: AgentState) -> AgentState:
    response = bound_llm.invoke(state["messages"])
    return {**state, "messages": state["messages"] + [response],
            "iteration_count": state.get("iteration_count", 0) + 1}

def should_continue(state: AgentState) -> str:
    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        return "force_stop"
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "end"

def force_stop_node(state: AgentState) -> AgentState:
    return {**state, "messages": state["messages"] + [AIMessage(
        content=f"Maximum iterations ({MAX_ITERATIONS}) reached. "
                "Here is what I accomplished so far. Please ask again to continue."
    )]}

def build_graph():
    checkpointer = InMemorySaver()  # swap to SqliteSaver / RedisSaver / PostgresSaver in prod

    graph = StateGraph(AgentState)
    graph.add_node("agent",      agent_node)
    graph.add_node("tools",      guarded_tools)
    graph.add_node("force_stop", force_stop_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {
        "tools":      "tools",
        "force_stop": "force_stop",
        "end":        END,
    })
    graph.add_edge("tools",      "agent")
    graph.add_edge("force_stop", END)

    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = build_graph()
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id},
              "callbacks": [AgentTelemetry(session_id=session_id)]}

    print(f"Session: {session_id}")
    print("Ask anything (Ctrl+C to quit)\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except KeyboardInterrupt:
            break
        if not user_input:
            continue

        result = app.invoke(
            {"messages": [HumanMessage(user_input)], "iteration_count": 0},
            config=config
        )
        last_msg = result["messages"][-1]
        print(f"Agent: {last_msg.content}\n")

        if audit_log:
            print(f"[Audit] Last tool call: {audit_log[-1]['tool']} → {audit_log[-1]['decision']}")
