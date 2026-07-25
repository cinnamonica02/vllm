# Design: Tensor-Descriptor path for W8A8 block-scaled FP8 matmul

## Status

Draft — kernel edit sketched, not yet correctness-tested or benchmarked.
Branch: `cinnamonica02/w8a8-fp8-block-scaled-td` (proposed, not yet created).

## Context

RFC #42545 ("Tensor descriptor (TD) adoption strategy for vLLM Triton kernels")
proposes migrating Triton kernels from raw-pointer loads to
`tl.make_tensor_descriptor`, which lowers to TMA on Hopper+ and 2D block
reads on Intel Xe. Adoption is gated behind the shared `VLLM_TRITON_USE_TD`
env var (tri-state: unset = platform auto-detect, `1`/`0` = force), per
@xuechendi's feedback against adding a new `*_USE_TD` var per kernel.

@oonyshch (working with @afierka-intel) is driving this RFC and has already
benchmarked several candidate kernels on B200. Their finding, stated
directly in the issue thread:

> The ones that were beneficial are open/draft PRs, mostly GEMM kernels:
> large, contiguous, K-contiguous (stride-1 innermost) 2-D operand tiles
> feeding `tl.dot`, where the gain scales with K/N and amortizes the
> descriptor setup.

They also rejected three decode-attention kernels
(`_fwd_kernel_stage1`/`_fwd_kernel_stage2`/`_fwd_grouped_kernel_stage1` in
`triton_decode_attention.py`) we had originally proposed, on two structural
grounds:

1. No `tl.dot` at all (decode is GEMV: `qk = tl.sum(q*k)`) — nothing for TD
   to attach to.
2. Where `tl.dot` does exist, the KV operand comes from a paged gather
   (`Req_to_tokens -> page*stride`), which can't be expressed as a TD load
   (`make_tensor_descriptor` requires innermost stride==1 addressed by block
   coordinates; a gather isn't block-coordinate addressable).

Measured result: TD was *slower* on all three, by 7.6%–32.5%, and these
kernels only back a CUDA fallback path (FA/FlashInfer are default), so even
a win wouldn't have touched real decode traffic.

## Why this kernel

We traced the 5 named follow-up PRs against actual kernel/file names in the
tree to find what's still open in the same "GEMM feeding tl.dot" category
she described:

| PR | Kernel | File | Status |
|---|---|---|---|
| #47152 | `_w8a8_block_int8_matmul` | `int8_utils.py` | claimed (INT8 only — title is explicit) |
| #47084 | `_bmm_chunk_fwd_kernel` | `ssd_bmm.py` | claimed |
| #45816 | `_chunk_cumsum_fwd_kernel` | `ssd_chunk_state.py` | claimed (note: no `tl.dot` — TD used for the `dt_out` *store* side only, not an operand load) |
| #46340 | `moe_mmk` | not found in this checkout | claimed (likely introduced by that PR's own unmerged diff) |
| #42436 | fused_moe TD path | `fused_moe.py` | claimed |

`_w8a8_triton_block_scaled_mm` in
`vllm/model_executor/layers/quantization/utils/fp8_utils.py` is the **FP8**
sibling of the claimed INT8 kernel. #47152's title is explicitly
"W8A8 block-**INT8** matmul" — the FP8 version was never named. Verified:

- Has a real `tl.dot`: `accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]`.
- Zero gather/paging markers (`page`, `gather`, `block_table`,
  `Req_to_tokens`) — every load is a dense affine-offset tile from a
  contiguous tensor, not an indirect lookup.
- Same autotuned `BLOCK_SIZE_M/N/K` shape (32–256) as `_bmm_chunk_fwd_kernel`,
  the sibling kernel her own team already validated benefits from TD.
- Backs FP8-quantized linear layers broadly — runs on plain CUDA, no XPU
  dependency, testable on Modal.

Two other unclaimed candidates surfaced in the same sweep and are documented
here for later, but are not this PR's scope:

- `_chunk_state_fwd_kernel` (`model_executor/layers/mamba/ops/ssd_chunk_state.py`) — `acc += tl.dot(x, b)`, same block-config family as the claimed `_bmm_chunk_fwd_kernel`.
- `_chunk_scan_fwd_kernel` (`model_executor/layers/mamba/ops/ssd_chunk_scan.py`) — `tl.dot(C, prev_states)` / `tl.dot(cb, x)`, chunk_size is the K dimension (typically 256, tunable) — direct match for "gain scales with K/N."

## Duplicate-work check (per AGENTS.md)

```bash
gh pr list --repo vllm-project/vllm --state open --search "_w8a8_triton_block_scaled_mm in:body"
# no pull requests match

gh pr list --repo vllm-project/vllm --state open --search "block_scaled_mm fp8 in:title,body"
# no pull requests match

gh pr list --repo vllm-project/vllm --state open --search "moe_mmk in:body"
# 2 matches: #46871 (unrelated, all-to-all EP), #46340 (already accounted for above)

gh pr list --repo vllm-project/vllm --state open --search "chunk_state_fwd_kernel OR chunk_scan_fwd_kernel in:body"
# no pull requests match
```

No overlap found. Re-run before opening any PR, since these PRs are all
still unmerged and moving.

## Kernel design

`_w8a8_triton_block_scaled_mm` gets a new `USE_TD: tl.constexpr = False`
parameter. Because it's `constexpr`, Triton specializes the compiled kernel
per value and the unused branch is fully dead-code-eliminated — this is the
same dual-path pattern already used in `triton_unified_attention.py`
(`USE_TD`/`USE_TD_QO`).

Raw-pointer path (unchanged, still the default):

```python
offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
# ... masked tl.load + manual pointer increment per K-tile
```

TD path:

```python
a_desc = tl.make_tensor_descriptor(
    base=A, shape=(M, K), strides=(stride_am, stride_ak),
    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
)
b_desc = tl.make_tensor_descriptor(
    base=B, shape=(K, N), strides=(stride_bk, stride_bn),
    block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
)
# per K-tile:
a = a_desc.load([pid_m * BLOCK_SIZE_M, k * BLOCK_SIZE_K])
b = b_desc.load([k * BLOCK_SIZE_K, pid_n * BLOCK_SIZE_N])
```

Notes:

- Descriptors are built once per program and reused across the whole K
  loop — this is exactly the "amortizes the descriptor setup" case she
  described, and the gain should scale with `K / BLOCK_SIZE_K` (more reuse
  per descriptor).
- TD auto-zero-pads out-of-bounds reads, so the TD path does **not** need
  the raw path's `% M` / `% N` wraparound trick or the per-K-tile
  boundary mask (`offs_k[None, :] < K - k * BLOCK_SIZE_K`). This is a
  behavior difference (zero-pad vs. wrap-and-mask-at-store) that is
  equivalent in the current code because both are masked away at the final
  `tl.store`, but it's worth calling out explicitly in the PR description
  since it's a real semantic change to how boundary tiles are handled.
- The per-block dequant scale loads (`As`/`Bs`) are untouched — they're 1-D
  vector loads, not the 2-D operand feeding `tl.dot`, so they're out of
  scope per her own stated criterion.

Wrapper (`w8a8_triton_block_scaled_mm`) resolves `use_td` from the shared
env var, matching `triton_attn.py`'s pattern — no new per-kernel flag:

```python
td_override = envs.VLLM_TRITON_USE_TD
use_td = current_platform.is_xpu() if td_override is None else td_override
```

## Correctness plan

Extend `tests/kernels/quantization/test_block_fp8.py::test_w8a8_block_fp8_matmul`
(already sweeps M/N/K/block_size/out_dtype against a PyTorch reference) with
a `use_td` parametrize case. Must match the existing tolerance before any
performance number is trusted — this mirrors her own methodology (she
reported the rejected kernels were "bit-exact" before reporting they were
slower).

## Performance validation plan

`benchmarks/kernels/benchmark_block_fp8_gemm.py` already sweeps M/N/K
through the public wrapper — no new benchmark script needed, just add TD as
a third comparison arm (currently Triton vs. Cutlass) or run it twice with
`VLLM_TRITON_USE_TD=0` / `=1`.

**Hardware constraint:** `tl.make_tensor_descriptor` lowers to TMA, which
only exists on Hopper (sm_90) and newer. On Modal this means **H100, H200,
or B200 only** — A100/L40S/L4/T4 predate TMA hardware and would either fail
to compile the TD path or silently take a non-representative fallback.
Matches the hardware tier she benchmarked on (B200), so no mismatch to
account for.

## Decision criteria

Per her explicit offer: "if you wire it to `VLLM_TRITON_USE_TD` and an e2e
run on any kernel we currently don't have a PR for ... shows real benefit
... that's worth a PR." Per-kernel micro-benchmark alone is necessary but
not sufficient — she was explicit that per-kernel micro isn't e2e. Plan:

1. Correctness (bit-exact / tolerance match, TD vs raw path).
2. Micro-benchmark via the existing script, across the M/N/K range it
   already covers — confirm gain does scale with K/N as hypothesized, not
   just a single lucky shape.
3. If micro-benchmark shows a real win, run an e2e serving benchmark
   (`vllm bench`) on a model that actually exercises FP8 block-scaled
   linear layers, per AGENTS.md's "run model evals for model-affecting
   changes" requirement.
4. Only with both numbers in hand, post to the issue thread / open the PR.
   AGENTS.md requires the PR description to state AI assistance was used,
   include test commands + results, and explain why this doesn't duplicate
   the listed PRs (this doc is the source material for that explanation).

## Open risks

- FP8 is a 1-byte dtype; TMA descriptor alignment/tile-size constraints
  are stricter on some dtypes than others (the unified-attention pilot
  kernel has an explicit static assert that `BLOCK_SIZE % TILE_SIZE == 0`
  for its case) — no equivalent assertion exists yet here and needs to be
  derived/verified, not assumed safe.
- Zero-pad-on-OOB (TD) vs. wrap-and-mask-at-store (raw path) is only
  provably equivalent for *this* kernel's current usage; don't assume it
  generalizes without the correctness test passing.
- oonyshch/afierka-intel offered to guide/run XPU + Battlemage validation
  if an e2e win shows up — worth looping them in once we have a number,
  not before.

## Next steps

- [ ] Add `use_td` parametrize case to `test_w8a8_block_fp8_matmul`.
- [ ] Run correctness test on Modal (any NVIDIA GPU — doesn't need Hopper
      for the raw path, but the TD path does).
- [ ] Run `benchmark_block_fp8_gemm.py` with `VLLM_TRITON_USE_TD=0/1` on
      Modal H100 or newer.
- [ ] Re-run the duplicate-check `gh pr list` commands (state may have
      moved).
- [ ] If results are favorable, draft e2e benchmark plan before commenting
      on the issue thread.
