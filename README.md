# 02 · Context Window Engine

Explicit context assembly and compaction for long conversations, with cache-aware prompt layout and quality-aware latency evaluation.

**Status: planning — implementation has not started.** This README describes intended scope and acceptance criteria. It does not establish working behavior, completed tests, security guarantees, or measured performance.

Week 2 of an eight-repository agent-systems portfolio.

## Planned scope

- Token budgeting with a stable prompt prefix and provider-specific cache metadata when available.
- Tiered retention of recent turns, compacted history, and durable task facts.
- Long-conversation replay with retrieval-of-facts checks and time-to-first-token measurement.

## Planned setup and observability

The implementation is intended to provide a `docker-compose.yml` and `Makefile` so a reviewer can use `make demo` for a deterministic offline demonstration and `make test` for offline regression checks. **These commands and their supporting files are not implemented yet.** Live-model runs will be optional and separately documented with configuration and cost assumptions.

Every model call and tool execution is intended to emit an OpenTelemetry span. Planned aggregate reporting includes token usage, estimated USD per 1,000 requests, p50/p95 latency, and p50/p95 time to first token (TTFT) where streaming applies. Offline fixtures can verify instrumentation and behavior; live-provider performance requires separate measurements.

The core mechanics will use raw model SDKs with explicit control flow; no LangChain, LlamaIndex, or other orchestration framework will implement them.

See [PRODUCTION_CHECKLIST.md](PRODUCTION_CHECKLIST.md) for shared acceptance criteria.

## Proposed architecture and tradeoffs

A context assembler separates stable instructions, durable task facts, compacted history, and recent turns. A budget policy chooses a compaction tier before a provider adapter sends the request. Compaction events and retained provenance are explicit state.

Compression reduces prompt size but may lose constraints or provenance. Stable prefixes can benefit provider caching, but cache behavior and billing vary by provider. Measuring TTFT requires separating live provider behavior from offline correctness checks.

## Roadmap and acceptance criteria

Complete the shared reproducibility and observability checklist, then implement and verify these project-specific criteria:

- [ ] Replay at least 100 turns with deterministic budget accounting and recorded compaction decisions.
- [ ] Preserve a defined set of task facts and constraints across compaction; report losses explicitly.
- [ ] Measure cache usage where supported and compare equivalent workloads with and without tiered compaction.
- [ ] Evaluate the target of approximately flat p95 TTFT across turn-count buckets; publish the observed trend and quality tradeoff even if the target is missed.

## Candidate failure modes

These are design risks to investigate, **not tested or observed findings**:

- Compaction drops a user constraint, tool reference, or unresolved obligation.
- Summary drift introduces a claim absent from the original conversation.
- Prefix changes invalidate caches, or compaction itself dominates latency and cost.

The implementation will add reproduction cases, actual outcomes, and mitigations as evidence becomes available.

## Baseline versus optimized metrics

No benchmark has been run. The following comparison is planned:

- **Baseline:** Uncompacted history until the provider limit, with over-limit requests recorded as failures.
- **Optimized:** Stable-prefix context assembly with tiered compaction under the same task workload.

| Metric | Baseline | Optimized |
| --- | --- | --- |
| Input/output/total tokens per request | Not yet measured | Not yet measured |
| Estimated USD per 1,000 requests | Not yet measured | Not yet measured |
| End-to-end latency p50 | Not yet measured | Not yet measured |
| End-to-end latency p95 | Not yet measured | Not yet measured |
| Streaming TTFT p50, where applicable | Not yet measured | Not yet measured |
| Streaming TTFT p95, where applicable | Not yet measured | Not yet measured |
| Project-specific quality and failure rate | Not yet measured | Not yet measured |

Future results must identify the workload revision, model/provider configuration, environment, concurrency, sample counts, and quality criteria. Cost estimates must disclose pricing date and assumptions. Report the complete workload, including errors and retries, and retain raw results so baseline and optimized runs can be reproduced. Do not substitute target values for measurements.

## Portfolio

- [01 · Raw Agent Tool Loop](https://github.com/sn41lz/agent-systems-01-tool-loop)
- [02 · Context Window Engine](https://github.com/sn41lz/agent-systems-02-context-window)
- [03 · Hierarchical Agent Memory](https://github.com/sn41lz/agent-systems-03-hierarchical-memory)
- [04 · Multi-Agent Orchestrator](https://github.com/sn41lz/agent-systems-04-multi-agent-orchestrator)
- [05 · Agent Evaluation Harness](https://github.com/sn41lz/agent-systems-05-eval-harness)
- [06 · Secure Code Sandbox](https://github.com/sn41lz/agent-systems-06-secure-sandbox)
- [07 · Durable Agent Runtime](https://github.com/sn41lz/agent-systems-07-durable-runtime)
- [08 · Agent Platform API and SDKs](https://github.com/sn41lz/agent-systems-08-api-sdk)
