"""
Multi-Agent Router — Router + Specialist Agents Example
========================================================
Demonstrates:
  - Router node with structured LLM output + confidence threshold
  - Adversarial input sanitization before classification
  - Two specialist agents (billing, technical) with scoped tools
  - Clarification node with round limit
  - Telemetry propagation to subagents
  - Supervisor orchestrator pattern

Install:
  pip install langgraph langgraph-supervisor langchain-anthropic pydantic

Run:
  python agent.py
"""

import os
import re
import uuid
import time
from typing import Literal, TypedDict

from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent


# ---------------------------------------------------------------------------
# 1. Model
# ---------------------------------------------------------------------------

MODEL_ID = os.getenv("AGENT_MODEL_ID", "claude-sonnet-4-6")

llm = ChatAnthropic(model=MODEL_ID, max_tokens=2048)


# ---------------------------------------------------------------------------
# 2. Specialist Tools (each agent gets only what it needs)
# ---------------------------------------------------------------------------

@tool
def lookup_invoice(invoice_id: str) -> str:
    """Look up an invoice by ID."""
    return f"[Simulated] Invoice {invoice_id}: $149.99, status: paid, date: 2025-03-01"

@tool
def check_refund_policy(product: str) -> str:
    """Check the refund policy for a product."""
    return f"[Simulated] {product} has a 30-day refund policy. Contact billing@example.com."

@tool
def get_error_logs(user_id: str) -> str:
    """Retrieve recent error logs for a user."""
    return f"[Simulated] User {user_id}: 3 errors in last 24h — NullPointerException in auth module."

@tool
def check_service_status() -> str:
    """Check current service health status."""
    return "[Simulated] All systems operational. Last incident: 7 days ago."

BILLING_TOOLS   = [lookup_invoice, check_refund_policy]
TECHNICAL_TOOLS = [get_error_logs, check_service_status]


# ---------------------------------------------------------------------------
# 3. Telemetry (propagated to subagents)
# ---------------------------------------------------------------------------

class AgentTelemetry(BaseCallbackHandler):
    def __init__(self, session_id: str, agent_name: str = "root"):
        self.session_id = session_id
        self.agent_name = agent_name
        self.events = []

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.events.append({"type": "llm_start", "agent": self.agent_name,
                             "session_id": self.session_id, "ts": time.monotonic()})

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.events.append({"type": "tool_call", "agent": self.agent_name,
                             "tool": serialized.get("name"), "session_id": self.session_id})


def make_subagent_config(parent_config: dict, child_agent_name: str) -> dict:
    """Propagate parent telemetry callbacks into child agent config."""
    return {
        "callbacks": parent_config.get("callbacks", []),
        "tags": [*parent_config.get("tags", []), f"subagent:{child_agent_name}"],
        "metadata": {
            **parent_config.get("metadata", {}),
            "parent_session_id": parent_config.get("metadata", {}).get("session_id"),
            "subagent_name": child_agent_name,
        }
    }


# ---------------------------------------------------------------------------
# 4. Specialist Agents (scoped tools)
# ---------------------------------------------------------------------------

billing_agent = create_react_agent(
    model=llm,
    tools=BILLING_TOOLS,
    state_modifier=(
        "You are a billing specialist. Help with invoices, charges, refunds, and pricing. "
        "Be concise and reference specific invoice IDs when available."
    )
)

technical_agent = create_react_agent(
    model=llm,
    tools=TECHNICAL_TOOLS,
    state_modifier=(
        "You are a technical support specialist. Help with bugs, crashes, errors, and app issues. "
        "Always check service status and error logs before responding."
    )
)


# ---------------------------------------------------------------------------
# 5. Router
# ---------------------------------------------------------------------------

INJECTION_PATTERNS = [
    r"ignore (your |previous |all )?(instructions|prompt|rules)",
    r"you are now",
    r"disregard",
    r"pretend you are",
    r"override",
]

def sanitize_input(message: str) -> tuple[str, bool]:
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, message, re.IGNORECASE):
            return message, True
    sanitized = re.sub(r"[\x00-\x1F\x7F]", " ", message)[:2000]
    return sanitized, False


class RouteDecision(BaseModel):
    route: Literal["billing", "technical", "clarification"]
    confidence: float
    reason: str

router_llm = llm.with_structured_output(RouteDecision)

ROUTER_PROMPT = """You are a customer support triage router. Classify the user's intent.

Routes:
- billing: invoice, payment, charge, refund, pricing, subscription cost
- technical: bug, crash, error, not working, broken, slow, login issue
- clarification: unclear, could be multiple things, or needs more context

Respond conservatively — if uncertain, use 'clarification'.
Confidence must be 0.0 to 1.0."""

MAX_CLARIFICATION_ROUNDS = 2


class RouterState(TypedDict):
    user_message: str
    route: str
    confidence: float
    messages: list
    clarification_rounds: int


def router_node(state: RouterState) -> RouterState:
    msg, injection = sanitize_input(state["user_message"])
    if injection:
        return {**state, "route": "clarification", "confidence": 0.0}

    decision = router_llm.invoke([
        {"role": "system", "content": ROUTER_PROMPT},
        {"role": "user",   "content": msg}
    ])
    route = decision.route if decision.confidence >= 0.7 else "clarification"
    return {**state, "route": route, "confidence": decision.confidence}

def route_decision(state: RouterState) -> str:
    return state["route"]


# ---------------------------------------------------------------------------
# 6. Specialist Nodes
# ---------------------------------------------------------------------------

def billing_node(state: RouterState) -> RouterState:
    result = billing_agent.invoke(
        {"messages": [HumanMessage(state["user_message"])]}
    )
    response = result["messages"][-1].content
    return {**state, "messages": state["messages"] + [
        {"role": "assistant", "content": f"[Billing] {response}"}
    ]}

def technical_node(state: RouterState) -> RouterState:
    result = technical_agent.invoke(
        {"messages": [HumanMessage(state["user_message"])]}
    )
    response = result["messages"][-1].content
    return {**state, "messages": state["messages"] + [
        {"role": "assistant", "content": f"[Technical] {response}"}
    ]}

def clarification_node(state: RouterState) -> RouterState:
    rounds = state.get("clarification_rounds", 0)
    if rounds >= MAX_CLARIFICATION_ROUNDS:
        return {**state, "messages": state["messages"] + [{
            "role": "assistant",
            "content": (
                "I'm having difficulty understanding your request after a few attempts. "
                "Please contact support@example.com for direct assistance."
            )
        }]}
    return {**state,
            "messages": state["messages"] + [{"role": "assistant", "content": (
                "I want to make sure I connect you with the right team. "
                "Is your issue related to (a) billing / invoices, or (b) a technical problem / app bug?"
            )}],
            "clarification_rounds": rounds + 1}


# ---------------------------------------------------------------------------
# 7. Build Graph
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(RouterState)
    graph.add_node("router",        router_node)
    graph.add_node("billing",       billing_node)
    graph.add_node("technical",     technical_node)
    graph.add_node("clarification", clarification_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_decision, {
        "billing":       "billing",
        "technical":     "technical",
        "clarification": "clarification",
    })
    for node in ["billing", "technical", "clarification"]:
        graph.add_edge(node, END)

    return graph.compile()


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = build_graph()
    session_id = str(uuid.uuid4())
    telemetry = AgentTelemetry(session_id=session_id, agent_name="router")

    print(f"Multi-Agent Support Router | Session: {session_id}")
    print("Type your support question (Ctrl+C to quit)\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except KeyboardInterrupt:
            break
        if not user_input:
            continue

        result = app.invoke(
            {
                "user_message": user_input,
                "route": "",
                "confidence": 0.0,
                "messages": [],
                "clarification_rounds": 0,
            },
            config={"callbacks": [telemetry]}
        )

        msgs = result.get("messages", [])
        if msgs:
            last = msgs[-1]
            content = last["content"] if isinstance(last, dict) else last.content
            print(f"Support: {content}")
        print(f"  [routed to: {result['route']} | confidence: {result['confidence']:.2f}]\n")
