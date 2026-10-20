# Current model API integration

API review: **2026-10-20**, OpenAI Python SDK **3.31.0**. Use a model available to your
account and verify that it supports the options you select. The default request leaves
model-specific caching and reasoning options unset.

## Controls and compatibility

The raw SDK request is assembled in [modern.py](../src/context_window/modern.py).
No model names, API keys or price tables are embedded.

| CLI control | Wire behavior |
| --- | --- |
| --cache-mode default | Omit cache configuration |
| --cache-mode implicit | prompt_cache_options mode implicit, TTL 30m |
| --cache-mode explicit | Mode explicit; one breakpoint at the end of the stable developer message |
| --cache-mode off | Mode explicit with no breakpoints |
| --reasoning-effort | Forward the caller's selected effort |
| --preflight | Count the prepared request with the provider before generation |
| --grade-answers | Strict structured output plus deterministic fixture scoring |
| --output-tokens | Reserve and cap generated output, including model reasoning usage |

The current [cache guide](https://developers.openai.com/api/docs/guides/prompt-caching)
documents these cache options for supported current models. Explicit mode without
markers disables caching. The documented minimum eligible prefix is 1,024 tokens;
our default is intentionally shorter. Use an actual reusable policy through
--instructions-file, rather than padding prompts to chase hits. The ledger and
archive are excluded from our explicit breakpoint. The tool catalog is stable too.
Cache writes and reads are separate usage partitions, with potentially different prices.

prompt_cache_key carries a stable application fingerprint for traceability; it does not
prove a cache hit. We record observed cached_tokens and cache_write_tokens. Missing
write counts are unknown, and estimates requiring them stay null. A zero must be
reported, not guessed. Caller-supplied USD/million rates need a pricing label.

## Counting and output

The [token-count endpoint](https://developers.openai.com/api/docs/guides/token-counting)
receives the final input, tools, choice, reasoning and structured output format.
Overflow stops before generation. This is an additional network request; local assembly
still uses its pinned application encoding. Count overhead is separate in the sample,
included in overall duration and excluded from TTFT, which starts at generation.

Model calls use streaming, store=false, truncation=disabled and max_retries=0.
A completed response with valid usage and nonempty text is required. Incomplete streams,
invalid usage, timeouts and API failures produce sanitized errors. Cancellation propagates
and the stream closes. No partial text is recorded as a successful answer.

## Native compact-and-continue

The [compaction guide](https://developers.openai.com/api/docs/guides/compaction)
defines a complete returned context window, including opaque compaction state and
retained items. The native command forwards all of it unchanged in order, then appends
the next user query. It does not select only the encrypted item or tokenize its ciphertext.
Both stages require provider counting, and the source must fit before compacting.

The SDK output union may lag retained input-item shapes. Serialization keeps all
provider-supplied fields with exclude_unset rather than filtering to known item types.
Mock-transport tests verify retained messages, assistant phase, reasoning, encrypted
content and future fields. This verifies transport handling, not live endpoint compatibility.

The [conversation-state guide](https://developers.openai.com/api/docs/guides/conversation-state)
also explains why reasoning output items and assistant phases matter to continuation.
The manual text/tool format is for local compilation and fixed replay. A production
reasoning agent must retain its full provider output protocol separately.

Native compaction is a two-stage endpoint experiment, not a fourth measured offline
policy or an automatic fallback. Its usage/cost is reported separately from generation.
The repository contains no live native compression-ratio or quality measurements.

## Interpreting quality and cost

The structured grader checks exact fixture key/value/source triples. Duplicate claims,
stale values, wrong sources and malformed JSON fail. Correct paraphrases can fail this
strict metric; general usefulness and injection resistance are unmeasured.

Costs cover observed completed model calls only. Failed billed attempts, token-count
requests and non-token fees are not estimated. The CLI has a bounded turn count and
timeouts, but no monetary spending cap. Output content and native windows are not written
unless --include-content is selected; that opt-in can expose sensitive or opaque state.

These choices are exercised with actual SDK serialization against an in-process mock
HTTP transport. No API credentials were configured for the committed reference run.
