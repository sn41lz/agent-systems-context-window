# Reference benchmark

```sh
make setup
.venv/bin/context-window benchmark --turns 120 --runs 3 --output artifacts/benchmark
```

Or use `make benchmark RUNS=3` with Docker.
The committed [reference](reference/summary.json) was generated on 2026-10-20.
Exact environment, tokenizer checksum, source fingerprints and run parameters are embedded
in summary.json. samples.jsonl contains all 1,080 observations. report.html is self-contained;
download it and open locally. tokens.svg is the README chart.

Three policies receive the same deterministic launch-review conversation. Six explicit
constraints are introduced, budget/deadline are revised, and historical tools appear every
ninth turn. Each run rotates strategy order. Tokenization is warmed before measurement.
Timing covers compilation, including OTel export overhead; ingestion and initialization are
excluded. Percentiles interpolate at (n-1)q. Concurrency is one.

The capacity is 4,096 minus 512 output minus 128 framing tokens = 3,456. Full history is a
deliberately unconstrained counterfactual; tail and tiered enforce that capacity.
The tiered design adds explicit state absent from the baselines. All policies include the
same source turn labels. It is not a comparison of three automatic summary models.

Constraint availability is a substring check for each current ground-truth value in rendered
content. It does not penalize obsolete values that remain alongside current values and
does not establish answer quality. No model is called; actual API calls and cost are zero.
Usage, cache hits and TTFT fields remain null.

All over-budget baseline samples are retained. The tiered gate requires every sample to
fit and retain every current constraint value. Local timings vary by machine; use the source
hashes and raw rows to assess reproducibility rather than treating milliseconds as a guarantee.

To measure actual answers and latency, run the explicit live command twice with tail and
tiered strategies, the same model/options/turn count, fresh directories and --grade-answers.
Cache state, shared provider load, rates, reasoning and ordering can confound a comparison.
The native command is a separate compact-and-continue endpoint probe.
No live comparison is committed and no flat TTFT target is asserted.
