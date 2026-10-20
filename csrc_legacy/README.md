# csrc_legacy — the kernels upstream dropped

A frozen snapshot, derived mechanically and then left alone. Nothing upstream tracks this code any more, so it is edited in place from here on; see **Frozen** below for why no regeneration is expected.

## What this is

Upstream's 2026.09.30 release (`deepseek-ai/FlashMLA` #229) rewrote the DeepSeek-V4.1 kernels and dropped everything else: NVIDIA Hopper, the V3.2 / V3.2-no-RoPE / V4 KV cache formats, and dense MLA decoding. SGLang still serves all of them, so they are kept here while `csrc/` tracks upstream.

The two halves build as separate extensions with separate op namespaces (`flash_mla` and `flash_mla_legacy`) and share no source. That is what lets `csrc/` stay byte-identical to upstream — a sync is a directory replacement, not a merge.

## Provenance

    csrc_legacy == upstream 07a1089 ("Add kernels for DeepSeek v4.1", #221)
                 + the fork's own commits below
                 + a `legacy::` namespace prefix
                 + the conversion to PyTorch's stable C++ ABI

The fork's commits on top of `07a1089`, touching 22 files under `csrc/kernels/`:

| commit | what it added |
|---|---|
| `c288a04` | V3.2-no-RoPE sparse decoding on SM90 and SM100 (#10) |
| `c1dee56` | SM90 dense decode with `HEAD_DIM_K=512`, a zero RoPE tail (#9) |
| `10a8f98` | a low-shared-memory fallback in the decode scheduler, for large batches |
| `e97950d` | an accuracy fix for the `flashmla_kv` NSA backend on B200 (#5) |
| `d28f7fe` | the DSv4-compatible API shape |
| `1bb3ac2` | the SGLang compat shim and the SM90 dense fp8 extension |
| `3e18517` | two `static_assert`s hoisted out of discarded `if constexpr` branches, for nvcc 13.0 |
| `3620b63` | 528 B/token caches read as V4.1 by default (#15) |
| `1759917` | sparse-prefill statistics emitted in the requested units, or skipped (#17) |

Everything else is upstream's code at `07a1089`.

## Frozen

Upstream deleted these kernels, so there is nothing left to track: no new version of them will appear, and the only thing that could arrive is a backport that nobody is going to write. The tree is a snapshot and stays one. In practice that means an upstream sync touches `csrc/` only, and this directory is not expected to move again.

## The snapshot is complete

Nothing is carved out. The V4.1 instantiations and the pre-release (block-split) V4.1 layout are here too, even though `csrc/` serves V4.1 today — the API layer dispatches to them, and removing them once left the extension linking cleanly and then failing at `dlopen` with an undefined symbol. A shared library resolves symbols at load time, so "it builds" does not mean "it loads": `torch.ops.load_library` on both extensions is part of the acceptance check.

One thing is built differently rather than removed: the SM100 dense prefill (CUTLASS FMHA) under `kernels/sm100/prefill/dense/`. Its sources are here so the snapshot stays complete, but `setup_legacy_sources.py` leaves them out of the build. They are the heaviest translation units on either side -- half the CUDA extension's object code -- and nothing in SGLang calls the op. The current side leaves its identical copy out for the same reason, behind `FLASH_MLA_BUILD_DENSE_PREFILL=1`; a build that wants the op turns it on there, not here.

## Deliberate deviations

- **The stable ABI.** The fork wrote `at::Tensor`, `TORCH_CHECK`, `torch::empty` and pybind registration. All of it is converted, because a wheel's compatibility is the minimum over its extensions: one abi3 extension beside a CPython-locked one still locks the wheel. The conversion is mechanical; where upstream converted the same file in its own #230, upstream's version was taken rather than re-derived.
- **Scalar parameter types match the schema.** Every boxed entry point here takes `int64_t` for a schema `int` and `double` for a schema `float`, and narrows explicitly in the body where a kernel's parameter struct keeps a smaller field. `TORCH_BOX` reinterprets the boxed slot by the C++ parameter type rather than converting it, so a narrow parameter reads half the slot. For integers that happens to work on a little-endian host when the value fits, which is why `int d_v` and `uint32_t rope_dim / n_wv_group / num_per_channels` survived the conversion unnoticed. For floating point it does not work at all: a `const float softmax_scale` in `dense_decode.cpp` received the low 32 bits of a double's bit pattern, so `1/sqrt(576)` (`0x3FA5555555555555`) arrived as `1.466e13`, every logit overflowed, and SM90 dense decoding returned all-NaN. Both kinds are fixed; if an argument ever reads back as garbage, check its C++ type against the schema before anything else.

- **No compatibility shim.** The fork compiled `csrc/python_api.cpp` into the extension to give sgl-kernel the signatures it had been calling. Those signatures are gone: every kernel here is reached through the ordinary entry points with `use_legacy=True`, which is one surface instead of two that drift. Dense fp8 decoding keeps its own scheduler op, because the kernel wants its buffers up front rather than allocating them.
