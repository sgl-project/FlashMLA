# tests/legacy — the tests for the kernels upstream dropped

These came from this fork's own test suite, for the kernels that now live in `csrc_legacy/`: NVIDIA Hopper, the V3.2 / V3.2-no-RoPE / V4 KV cache formats, dense MLA decoding, and the MegaAttention kernels over the pre-release block-split V4.1 layout. Upstream deleted all of it in its 2026.09.30 release, so its own `tests/` cannot cover any of it.

They are a separate directory for the same reason `csrc_legacy/` is: the two sides describe a KV cache differently, so one shared `quant.py` would be half true for each. `quant.py`, `lib.py` and `ref.py` here are this fork's, and `kernelkit` is shared from `tests/` unchanged.

## What changed from the fork's originals

- Every entry point is called with `use_legacy=True`. That is the only way to reach these kernels: the two extensions register the same op names in separate namespaces, and the Python layer picks one per call.
- `kv_format` is passed by name. It used to be the last positional parameter of `flash_mla_with_kvcache`; `enable_batch_invariant` and `use_legacy` now come before it.
- `kk.bench_kineto` is `kk.bench`, which is what upstream renamed it to.

Nothing else: the shapes, the tolerances and the reference implementations are the fork's.

## Running them

These are scripts, not pytest modules -- that is how this fork wrote them, and upstream writes its own the same way (it spells them `test-*.py`, with a dash, so that pytest leaves them alone). Run one directly:

```
PYTHONPATH=.:tests python tests/legacy/test_flash_mla_sparse_decoding.py
```

`PYTHONPATH` is needed because `sys.path[0]` is `tests/legacy/`, not the repo root: without it the script cannot find `flash_mla` or the shared `kernelkit`. The `conftest.py` beside these files appends `tests/` for the same reason, but it only runs under pytest.

`pytest tests/legacy` collects only `test_sparse_prefill_stats.py`, which is a real pytest module. The scripts' own entry points are named `run_case`, so pytest leaves them alone: they take a dataclass positional, which pytest would read as a fixture request and error on.

`test_fused_norm_rope_attn_rope_cast.py` builds its reference out of `tile_kernels` and `deep_gemm`, which are not dependencies of this repository. They are imported lazily, so collecting this directory does not need them; running that script without them exits with a message.

The V3.2 / V3.2-no-RoPE / V4 sparse decoding paths and dense decoding run on SM90 as well as SM100 -- SM90 is the reason this fork still builds for Hopper at all. The MegaAttention tests are SM100-family only.
