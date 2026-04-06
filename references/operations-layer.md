# Operations Layer

## Rationale

An agent that works in staging but fails silently in production is not production-grade. Operational concerns — model versioning, cost alerts, incident runbooks, multi-tenancy — are as important as the agent's core logic. **Design for observability, controllability, and graceful degradation before you go live.**

## Model Version Pinning

Silent model upgrades are a hidden source of regression. A model update can change tool-calling behavior, output format, or reasoning patterns without warning.

```python
import os

# Pin the model version explicitly — never use an alias like "claude-3" alone
MODEL_ID = os.getenv("AGENT_MODEL_ID", "claude-sonnet-4-6")

# At startup: log the pinned model version for audit trail
import logging
logging.info(f"Agent starting with model: {MODEL_ID}")

# In your LLM instantiation:
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model=MODEL_ID,
    max_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "8192")),
    timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
)
```

**Upgrade process:**
1. Deploy new model version behind a feature flag (`AGENT_MODEL_ID=new-model`)
2. Run regression suite against fixture traces (see `testing-layer.md`)
3. Enable for 5% of traffic via tenant-level flag
4. Promote to 100% only after metrics confirm no regressions
5. Pin old model as rollback target for 30 days

## Per-Tenant Rate Limiting and Cost Caps

Each tenant must have independently enforceable limits. A runaway tenant must not degrade service for others.

```python
from dataclasses import dataclass, field
from collections import defaultdict
import time

@dataclass
class TenantPolicy:
    tenant_id: str
    max_requests_per_minute: int = 60
    max_cost_usd_per_session: float = 5.0
    max_cost_usd_per_day: float = 50.0
    agent_enabled: bool = True
    allowed_tools: list = field(default_factory=list)   # empty = all tools allowed

class TenantRateLimiter:
    def __init__(self):
        self._request_counts: dict = defaultdict(list)   # tenant_id → [timestamps]
        self._daily_costs: dict = defaultdict(float)

    def check_rate_limit(self, tenant_id: str, policy: TenantPolicy) -> tuple[bool, str]:
        now = time.time()
        window_start = now - 60
        requests = self._request_counts[tenant_id]
        # Clean up old entries
        self._request_counts[tenant_id] = [t for t in requests if t > window_start]
        count = len(self._request_counts[tenant_id])

        if count >= policy.max_requests_per_minute:
            return False, f"Rate limit exceeded: {count}/{policy.max_requests_per_minute} req/min"
        self._request_counts[tenant_id].append(now)
        return True, ""

    def record_cost(self, tenant_id: str, cost_usd: float):
        self._daily_costs[tenant_id] += cost_usd

    def check_daily_budget(self, tenant_id: str, policy: TenantPolicy) -> tuple[bool, str]:
        spent = self._daily_costs[tenant_id]
        if spent >= policy.max_cost_usd_per_day:
            return False, f"Daily budget exceeded: ${spent:.2f} / ${policy.max_cost_usd_per_day}"
        return True, ""


# Usage in request handler:
def handle_request(tenant_id: str, user_message: str,
                   limiter: TenantRateLimiter, policies: dict[str, TenantPolicy]):
    policy = policies.get(tenant_id)
    if not policy:
        raise ValueError(f"Unknown tenant: {tenant_id}")

    if not policy.agent_enabled:
        return {"error": "Agent is currently disabled for your account."}

    ok, msg = limiter.check_rate_limit(tenant_id, policy)
    if not ok:
        return {"error": msg}

    ok, msg = limiter.check_daily_budget(tenant_id, policy)
    if not ok:
        return {"error": msg}

    # Proceed with agent invocation ...
```

## Per-Tenant Disable Without Redeployment

Disable a single tenant's access instantly via a remote config store — no code deploy needed.

```python
import httpx
import time
from functools import lru_cache

@lru_cache(maxsize=256)
def fetch_tenant_policy(tenant_id: str, ttl_hash: int) -> dict:
    """Fetch policy from remote config with 60-second TTL cache."""
    try:
        r = httpx.get(
            f"https://config.internal/tenant-policy/{tenant_id}",
            timeout=2.0
        )
        return r.json()
    except Exception:
        # Fail safe: if config fetch fails, allow access (don't lock out by default)
        return {"agent_enabled": True}

def get_ttl_hash(seconds: int = 60) -> int:
    return int(time.time() / seconds)

def is_tenant_enabled(tenant_id: str) -> bool:
    policy = fetch_tenant_policy(tenant_id, ttl_hash=get_ttl_hash(60))
    return policy.get("agent_enabled", True)
```

## Alert Escalation Matrix

Define alert thresholds and escalation paths before incidents happen.

| Alert | Threshold | Severity | Action |
|-------|-----------|----------|--------|
| Session cost spike | > $5 in one session | Warning | Log + notify team channel |
| Daily cost exceeded | > 90% of daily budget | High | Suspend new sessions, page on-call |
| Agent stuck (no progress) | > 5 min, 0 tool calls | High | Kill session, log trace, notify user |
| Tool failure rate | > 20% over 5 min window | High | Open circuit breaker, page on-call |
| Context exhaustion | Compaction failure × 3 | Medium | Terminate session gracefully, log |
| Unknown tenant request | Any | Critical | Reject, alert security team |

```python
from dataclasses import dataclass
from enum import Enum

class AlertSeverity(Enum):
    WARNING  = "warning"
    HIGH     = "high"
    CRITICAL = "critical"

@dataclass
class Alert:
    session_id: str
    tenant_id: str
    severity: AlertSeverity
    message: str
    metadata: dict = None

def fire_alert(alert: Alert, sink=None):
    """Route alert to appropriate channel based on severity."""
    import json, logging
    payload = {
        "session_id": alert.session_id,
        "tenant_id":  alert.tenant_id,
        "severity":   alert.severity.value,
        "message":    alert.message,
        "metadata":   alert.metadata or {},
    }
    logging.warning(f"[ALERT:{alert.severity.value.upper()}] {json.dumps(payload)}")

    if sink:
        sink(payload)  # plug in: PagerDuty, Slack webhook, CloudWatch alarm, etc.

# Usage in cost tracker:
def check_and_alert(session_cost: float, session_id: str, tenant_id: str, sink=None):
    if session_cost > 5.0:
        fire_alert(Alert(
            session_id=session_id,
            tenant_id=tenant_id,
            severity=AlertSeverity.WARNING,
            message=f"Session cost exceeded $5: ${session_cost:.2f}",
        ), sink=sink)
```

## Runbooks

Keep these short and actionable. The person on call should be able to resolve the incident in < 15 minutes.

### Runbook: Agent Stuck (No Progress)

**Symptoms:** Session has been running > 5 minutes with no tool calls or output tokens.

1. Pull the session trace from your observability store (filter by `session_id`)
2. Identify the last successful node / tool call
3. Check if the agent is waiting on a `interrupt()` that was never resumed
   - If yes: either resume (`Command(resume="yes/no")`) or terminate
4. Check for infinite loop: is the agent repeatedly calling the same tool?
   - If yes: kill the session, patch `max_iterations`, redeploy
5. Check for rate limit: is the LLM returning 429/529?
   - If yes: session will self-recover via `retry_with_heartbeat` if configured
6. If none of the above: terminate session, file a bug with the full trace

### Runbook: Cost Explosion

**Symptoms:** Single session or tenant exceeds 10× expected cost.

1. Identify the session(s) via cost tracker (`top_tools`, `tokens_by_turn`)
2. Kill the offending session(s) immediately
3. Check `tokens_by_turn`: are input tokens growing unboundedly? → compaction not firing
4. Check tool call counts: is one tool called 100× in a session? → missing circuit breaker
5. Check for missing `max_iterations` guard → add it
6. Apply emergency per-tenant cost cap if not already set

### Runbook: Tool Consistently Failing

**Symptoms:** Circuit breaker opens; tool error rate > 20% over 5 minutes.

1. Check if the external service is down (status page / health endpoint)
2. If service is down: circuit breaker should already be open — verify it is
3. Verify fallback chain is activating (`search_with_fallback` pattern)
4. If no fallback exists: temporarily disable the tool via `PermissionMode.ALWAYS_DENY`
5. Once service recovers: circuit breaker transitions to HALF_OPEN automatically
6. Monitor for 10 minutes before confirming recovery

### Runbook: Context Exhaustion Loop

**Symptoms:** Compaction failing 3× in a row; agent returns error or produces garbage output.

1. Pull the session message history from the checkpointer
2. Check `compact_failures` counter in state — if ≥ 3, circuit breaker should have stopped retrying
3. If compact_failures < 3: compaction LLM call may be failing — check LLM logs
4. Force-terminate the session: save last known output, send graceful message to user
5. Investigate: is there a tool producing huge outputs that bypass truncation?
6. Fix: add / lower `max_chars` in `truncate_tool_result`; redeploy

## Agent Versioning Strategy

```
v1.0  ─── main branch, production
v1.1  ─── staging, regression tested
v1.2  ─── feature branch, development
```

- **Pin model ID** in environment variable per deployment
- **Tag releases** in git: `git tag agent/v1.1 -m "LangGraph 0.3 upgrade"`
- **Blue-green deploy**: run v1.0 and v1.1 in parallel; shift traffic via load balancer
- **Rollback**: point load balancer back to v1.0 container — no DB migration needed if checkpointer schema is backward compatible
- **Session migration**: old sessions resume on v1.0; new sessions start on v1.1

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Using model alias (`claude-3`) instead of pinned ID | Pin full model ID in `AGENT_MODEL_ID` env var |
| No per-tenant cost cap | Each tenant needs `max_cost_usd_per_day` enforced in code |
| Runbooks written after an incident | Write runbooks before deploying — incidents happen fast |
| One rate limiter for all tenants | Per-tenant state; a runaway tenant must not affect others |
| Alert threshold too high | Tune thresholds in staging with realistic traffic; default is always wrong |
| Per-tenant disable requires redeploy | Use remote config fetch with TTL cache — disable in < 60 seconds |
| No model version in logs | Always log model ID at startup and in every trace event |
