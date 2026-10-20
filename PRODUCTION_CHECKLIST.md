# Production checklist and evidence

## Implemented and checked

- [x] Public source, raw SDK integration, explicit state reducer and compaction policy.
- [x] Docker Compose, Makefile, dependency hashes, digest-pinned base image and no-key demo.
- [x] Single-command demo/test verification; runtime network disabled for offline containers.
- [x] Non-root container defaults, read-only root, resource limits and dropped capabilities.
- [x] Offline tokenizer packaged with source/license notices and integrity checks.
- [x] Atomic task-state revisions, protected budget failure and complete historical tool groups.
- [x] Matched 120-turn benchmark with all samples, source hashes, environment and limitations.
- [x] Provider streaming, first-text latency and completed-response token usage.
- [x] Cache read/write accounting, labeled prices, null estimates for unknown write costs.
- [x] Current cache options and reasoning controls tested through actual SDK serialization.
- [x] Provider preflight rejects overflow before generation.
- [x] Native compact output forwarded without selecting, pruning or reordering items.
- [x] Exact-value/source grading, stale revisions, malformed streams, cancellation and timeouts.
- [x] Instance-scoped OTel spans without raw bodies or exception payloads in trace JSONL.
- [x] Python 3.12–3.14 CI, no-network container execution and installed-wheel smoke verification.
- [x] Failure modes, architecture, public usage examples and Spanish overview.

## Deliberately unclaimed

- [ ] Live cache hit rates and p50/p95 TTFT across 100+ turns.
- [ ] Live baseline-versus-optimized answer quality and price/performance measurements.
- [ ] Adversarial model evaluation, automatic fact extraction or semantic equivalence grading.
- [ ] Durable or distributed state; additional model-provider adapters; multimodal input.

Offline and mocked-transport tests cannot establish live provider behavior. Tools are
historical records here, so no tool-execution span is claimed. The separate tool-loop
repository implements execution. See the README for exact scope and reproduction.
