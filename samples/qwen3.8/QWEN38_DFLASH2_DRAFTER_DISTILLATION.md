# Qwen 3.8 27B DFlash2 drafter: qualification distillation

## Bottom line

DFlash2 is effective on RTX Spark, but its realized speed is governed primarily
by draft acceptance rather than raw drafter latency. With a seven-token draft
width, the qualified INT4 models deliver roughly **20-35 decode tok/s** on
representative source-analysis and patch-generation prompts, versus
**10-15 tok/s** without speculation. The same stack reaches **60-80+ tok/s**
when the continuation is highly predictable. These are different workload
regimes and should be reported separately.

## What the measurements show

| Workload | Acceptance | Typical result | Interpretation |
| --- | ---: | ---: | --- |
| Source analysis / patch generation, 4K-262K | **61-72%** | **20-35 tok/s** | Representative qualification range |
| Technical prose, 4K | **76-78%** | **41.8 tok/s** | Evidence that 40+ tok/s requires acceptance in the upper 70s |
| Patterned TypeScript, 4K | **91.4%** | **68.1 tok/s** | Strongly predictable coding ceiling |
| Prompt-grounded code copy, 4K | **97.8%** | **78.7-81.9 tok/s** | Upper bound, not a general coding claim |
| Prompt-grounded code copy, 262K | **99.1%** | **64.5 tok/s** | High acceptance remains valuable at long context |

The latest width-seven source-analysis sweep averaged **66.89% acceptance** for
the shifted INT4 graph and **67.16%** for the original graph. Their difference
is not statistically meaningful. The shifted hidden-state tap introduced for
the newer architecture did not establish a quality or throughput regression,
but neither did this sweep prove a material acceptance improvement.

Acceptance is prefix-based: once draft token `i` mismatches the target, token
`i` and all later proposals in that round are rejected. Improving early-token
agreement therefore has more value than improving independent per-position
accuracy late in the draft.

## Runtime configuration

The graph can emit seven draft tokens, but that capability does not select the
runtime width. Production model packages should include:

```json
{
  "speculative": {
    "max_draft_tokens": 7
  }
}
```

Without this top-level setting, GenAI's generic fallback uses width four.
Correcting width four to width seven improved representative geometric-mean
decode by **9.4%** for shifted INT4, **6.9%** for original INT4, and **3.5%**
for NVFP4. Width seven is the qualified package default, not a recommended
runtime-wide default for every model or workload.

Greedy target sampling is required for deterministic qualification:
`do_sample=false`, `top_k=1`, and temperature 1 or an equivalent greedy
configuration.

## How to improve the drafter

The most important next optimization is better acceptance on representative
agent traffic. Runtime kernel improvements of one or two percent are useful,
but they cannot turn a 65-70% workload into sustained 40+ tok/s. Based on the
measured relationship, the practical target is approximately **77-82%
acceptance** on source-analysis and patch-generation prompts.

Recommended drafter work:

1. Distill on real agent trajectories: repository context, issue text, patch
   planning, unified diffs, tests, tool-result continuation, and long-context
   retrieval.
2. Optimize accepted-prefix length, not only average token cross-entropy.
   Weight the first mismatch heavily because it invalidates the remaining
   proposal.
3. Include the full 4K-262K context distribution. Long-context acceptance
   should be evaluated independently from target attention and KV-cache cost.
4. Re-evaluate hidden-state tap placement using representative acceptance and
   tokens-per-target-forward, rather than synthetic perplexity alone.
5. Preserve the fast drafter graph: shared target embedding/head, compact
   DFlash2 state update, and INT4 drafter MatMuls were all important to the
   qualified latency.

## Deployment policy

Keep width seven as the packaged default for these qualified DFlash2 graphs,
while allowing per-request adaptation later. A useful adaptive policy should
maximize emitted tokens per millisecond, shortening the proposal when recent
accepted-prefix length falls and restoring width seven when acceptance
recovers. It should not react to acceptance alone: target verification cost,
context length, and concurrent batch occupancy also matter.

For product claims, use **approximately 20-35 decode tok/s across 4K-262K** for
representative batch-1 source-analysis workloads. Report **60-80+ tok/s** only
as a clearly labeled high-acceptance or prompt-grounded continuation ceiling.

