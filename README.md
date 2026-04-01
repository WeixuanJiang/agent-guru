# Agent Guru

`agent-guru` is a reusable skill for designing, reviewing, and hardening production-grade agent systems.

The skill itself is defined in [SKILL.md](./SKILL.md). It focuses on the system design around agents rather than prompt writing alone: routing, orchestration, tool safety, memory, observability, resilience, and persistence.

## Core Idea

> The LLM is the reasoning engine. Your code is the execution engine. The loop is the contract between them.

In production, the hard problems are usually not "how do I write a smarter prompt?" They are:

- how to route requests correctly
- when to use one agent vs multiple agents
- how to scope tools and enforce permissions
- how to manage memory and context windows
- how to observe cost, latency, and failures
- how to recover from errors and resume sessions safely

This skill is meant to address those problems directly.

## What This Skill Covers

The skill is organized into layers:

```text
INGRESS
  -> ROUTER
  -> ORCHESTRATOR
  -> TOOL / SAFETY
  -> MEMORY
  -> OBSERVABILITY
  -> RESILIENCE
  -> PERSISTENCE
```

Each layer has a dedicated reference document under [`references/`](./references/):

- `router-layer.md` for intent routing and dispatch
- `orchestrator-layer.md` for task decomposition and subagent delegation
- `tool-safety-layer.md` for schemas, permission guards, HITL, truncation, and concurrency checks
- `memory-layer.md` for short-term, working, and long-term memory patterns
- `observability-layer.md` for tracing, cost tracking, and config snapshotting
- `resilience-layer.md` for retries, circuit breakers, loop guards, and graceful degradation
- `persistence-layer.md` for session resume, checkpointing, and branching
- `production-checklist.md` for deployment readiness

## When To Use It

Use this skill when you are:

- designing a new agent system
- deciding whether a workflow needs multiple agents
- adding safety controls around tools
- debugging cost explosions or context exhaustion
- implementing human approval for risky actions
- adding tracing, retries, or durable state
- preparing an agent for production deployment

## Key Principles

- Safety must live in code, not prompts.
- Start with a single agent and add multi-agent complexity only when needed.
- Subagents should receive only the context and tools they actually need.
- Separate planning from execution for risky or irreversible workflows.
- Truncate tool outputs before they consume the full context window.
- Use durable persistence in production instead of in-memory state.

## Framework Fit

This skill is designed for agentic frameworks such as:

- LangGraph
- Strands
- similar graph- or loop-based agent runtimes

The examples are primarily LangGraph-oriented, but the design guidance is framework-agnostic.

## Repository Structure

```text
agent-guru/
├── README.md
├── SKILL.md
└── references/
    ├── memory-layer.md
    ├── observability-layer.md
    ├── orchestrator-layer.md
    ├── persistence-layer.md
    ├── production-checklist.md
    ├── resilience-layer.md
    ├── router-layer.md
    └── tool-safety-layer.md
```

## How To Read This Skill

Start with `SKILL.md` for the overall mental model. Then pull in the reference file for the specific problem you are solving instead of loading everything at once.

Suggested reading order:

1. `SKILL.md`
2. `router-layer.md`
3. `orchestrator-layer.md`
4. `tool-safety-layer.md`
5. `memory-layer.md`
6. `observability-layer.md`
7. `resilience-layer.md`
8. `persistence-layer.md`
9. `production-checklist.md`

## Status

This repository is a design playbook and reference skill. It is not a packaged framework or a drop-in library. The value is in the architecture patterns, guardrails, and implementation guidance.
