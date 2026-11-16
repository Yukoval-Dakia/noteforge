# Backends and hardware support

NoteForge runs the same r5 weights through one of two execution backends and one
of four devices. This page states what each combination is, how it is selected,
what has actually been measured, and what has not.

| backend | device | status | how it runs |
|---|---|---|---|
| `torch` | `cuda` | implemented, not measured here | bf16 autocast (explicit), static-cache decode, CUDA Graph replay |
| `torch` | `mps` | implemented, measured (short fixture) | PyTorch MPS, fp32 decode |
| `torch` | `cpu` | implemented, measured (x86_64 pending) | PyTorch CPU via oneDNN (x86) or Accelerate (arm64) |
| `mlx` | `mlx` | implemented, measured | `mlx.core` fp16 static-cache decode; MERT front end on PyTorch MPS |

`mlx`, `mps`, `cuda` and `cpu` are different execution paths. A result on one is
not evidence about another. In particular, Apple ARM numbers and x86_64 numbers
must be reported separately, because the BLAS/oneDNN/Accelerate kernels differ.

## Selection

`noteforge.backends.resolve_backend` is the single selection contract.

- `NOTEFORGE_BACKEND=auto` (default) picks `mlx` only when the host is Apple
  arm64, `mlx` is importable, **and** both exported weight folders exist
  (`NOTEFORGE_FAST_WEIGHTS`, `NOTEFORGE_FAST_MERT`). Otherwise it picks `torch`.
- `NOTEFORGE_BACKEND=mlx` asks for MLX and fails closed with a clear error when
  any precondition is missing; it never silently falls back to torch.
- `NOTEFORGE_BACKEND=torch` with `NOTEFORGE_DEVICE=auto` resolves to
  `cuda`, then `mps`, then `cpu` by availability.
- An explicit `cuda` or `mps` on a host that lacks it fails closed with
  `resolve_torch_device` raising, surfaced through `/v1/health` as `backend_error`.

`GET /v1/health` reports the requested backend and device, the resolved backend
and device, the error when unusable, and host availability for PyTorch, MLX and
CPU. `GET /v1/capabilities` adds the same availability block under `available`.

## CPU threads

CPU decode follows the caller's thread settings instead of a hard-coded value.
The previous import-time `torch.set_num_threads(3)` in
`noteforge.placement.model` was removed; importing that module no longer changes
the thread count.

```bash
NOTEFORGE_BACKEND=torch NOTEFORGE_DEVICE=cpu \
NOTEFORGE_CPU_THREADS=8 NOTEFORGE_CPU_INTEROP_THREADS=1 ...
```

The CLI equivalents are `--cpu-threads` and `--cpu-interop-threads`. When only
`--cpu-threads` is given, the interop count stays at the PyTorch default.

On the Apple host measured below, the thread count turns out not to matter for
decode: 1, 2, 4, 6 and 8 threads all produced whole-song generation within 2%.
The setting is honored and reported; it is not a speed knob on that path.

## MLX on Apple Silicon

MLX does not load the `.pt` file directly. Export both folders once, then point
the environment at them:

```bash
python -m noteforge.mlx.fast_model export models/noteforge-r5.pt ./models/mlx-fast
python -m noteforge.mlx.mert_fast export /path/to/pinned-mert2 ./models/mlx-mert
```

The MLX decode plan (`balanced`, `fast`, `sequential`) changes only scheduling;
it does not change the weights or the consensus rule. Batch shape matters: the
MLX path is fast at `batch-size 64` and slow at `batch-size 1`, which the
measurements below show directly.

The decode entry points use `torch.inference_mode()` rather than `torch.no_grad()`.
On the 100 s song that was 40.995 s vs 41.755 s median (three runs each, about
1.8% faster) with the same events.

`FullEventModel` is constructed on the meta device and the published weights are
assigned into it (`assign=True`). The model learns nothing at load time, but the
plain constructor randomly initialises about 223 `Linear` layers that
`load_state_dict` then overwrites. Skipping that work cut loading in isolation
from 1.338 s to about 0.52 s, and the harness `model_load_s` from 1.344 s to
0.992 s, with all 395 parameters and buffers bit-identical (worst absolute
difference 0.0) and the fixed-input events unchanged.
`noteforge.scripts.draft_song.build_model_on_meta` falls back to the ordinary
construct-and-load path if a PyTorch build rejects meta construction or
`assign=True`.

## NVIDIA CUDA

```bash
NOTEFORGE_BACKEND=torch NOTEFORGE_DEVICE=cuda ...
```

CUDA selects the static-cache, continuous-scheduler decode and captures the
decode step as a CUDA graph. `NOTEFORGE_DECODE_CUDA_GRAPH=0` (or the CLI
`--no-cuda-graph`) disables graph capture. `--autocast bf16` switches the decode
numerics to bfloat16 and is an explicit, separate path from the fp32 decode; it
is not bit-identical to fp32 and must not be described as such.

No NVIDIA device was available in the environment where the numbers below were
taken, so this repository publishes **no CUDA performance number**. The code
paths and the fail-closed capability checks exist; real CUDA figures require a
run on an NVIDIA host.

## x86_64 CPU

x86_64 and ARM64 share the PyTorch inference contract but not the kernel
implementations. `torch` CPU uses oneDNN on x86_64 and Accelerate on Apple
ARM64. `.github/workflows/cpu-x86-benchmark.yml` runs the measurement on a real
x86_64 GitHub runner, downloads the published r5 weights, checks their SHA-256,
and uploads `host.txt` and `results.jsonl`. One such run is in the measured
section below.

## Measured results

Raw rows are in `reports/backend-benchmark-20261116.json`. All rows use the
published r5 weights
(`1b1f2e8dee82101690778ef667b661fa1d88421f50034c0bd97572f3067d405b`) and one
fixed seed. The output is a note-multiset SHA-256, so an identical SHA across
rows means the backends produced identical events.

All SHAs in this page and in the JSON report are produced by the repository's
`noteforge.scripts.bench_backends.note_hash`. The short 8 s fixture hashes to
`36959f5b...` (11 notes) and the 100 s song to `fd95fcdc...` (281 notes).

Host: Apple Silicon (arm64, 8 cores), 16 GiB, macOS 27.3, PyTorch 2.15.0, MLX
0.32.4, Python 3.12.13, `torch.backends.mkldnn.is_available() == False`.

### 8 s fixture, torch CPU, thread sweep

Fixture: the first 8 s of a real licensed song with cached real MERT features
(audio SHA `ac723cf9...`), so the values are real but the clip is short. 5 runs
per thread count, `batch-size 1`, continuous scheduler.

| threads | runs | generation median s | range s | tokens | notes | output SHA |
|---|---|---|---|---|---|---|
| 1 | 5 | 1.5325 | 1.5295..1.5468 | 78 | 11 | `36959f5b...` |
| 2 | 5 | 1.5748 | 1.5592..1.6205 | 78 | 11 | `36959f5b...` |
| 4 | 5 | 1.5601 | 1.5450..1.5766 | 78 | 11 | `36959f5b...` |
| 8 | 5 | 1.5545 | 1.5443..1.5642 | 78 | 11 | `36959f5b...` |

Thread count does not move generation time on this short fixture; the spread
across 1/2/4/8 threads is under 0.05 s.

### 8 s fixture, other paths

| path | runs | generation median s | range s | output SHA |
|---|---|---|---|---|
| torch CPU, fast decode | 5 | 1.5623 | 1.5185..1.5906 | `36959f5b...` |
| torch CPU, reference decode | 5 | 1.9048 | 1.8860..1.9197 | `36959f5b...` |
| torch CPU, `--autocast bf16` | 3 | 3.3678 | 3.3451..3.4023 | `36959f5b...` |
| torch MPS, fp32 | 4 warm + 1 cold | 1.6106 | 1.6095..1.6117 (5.2528 cold) | `36959f5b...` |
| MLX, warm loop | 5 | 2.3205 | 2.3123..2.3492 | `36959f5b...` |

Fast decode is a stable ~1.20x speedup over reference decode with identical
events. MPS needs a warm-up: the first run was 5.2528 s, the rest 1.61 s. MLX at
`batch-size 1` is slower than torch CPU at that size.

On this Apple CPU path, `--autocast bf16` is about 2.2x slower than fp32
(3.37 s vs 1.56 s) and produced the same events. It is an explicit numerical
change that bought nothing here, so CPU keeps fp32.

### torch.compile on CPU

`--compile-decode` on the Apple CPU path is slower, not faster: about 6.56-6.61 s
after compilation versus 1.56 s eager, with a 46.2 s first (compile) run. This is
a real negative result; `--compile-decode` is not a CPU speedup here.

### int8 dynamic quantization on ARM CPU

`torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=qint8)` fails
on this host with `RuntimeError: Didn't find engine for operation
quantized::linear_prepack NoQEngine`. The Apple ARM CPU build of PyTorch 2.15.0
ships no quantized backend, so int8 CPU inference is not available on this
platform at all. It is a capability limit of the toolchain, not a property of
the model, and it is another reason the x86_64 number has to come from x86_64.

### 100.1 s song, batch-size 64

Real licensed song (audio SHA-256
`535386bc98ebabc03b66e7035d62447c0b8cffc530abb1107be593b4b77073a5`), kept out of
this repository. 1 seed, `phira-level 10`, `profile standard`, `batch-size 64`,
`consensus 1`.

| backend | generation s | wall s | tokens | notes | output SHA |
|---|---|---|---|---|---|
| torch CPU, 4 threads | 41.847 | 45.588 | 2029 | 281 | `fd95fcdc...` |
| mlx `balanced` | 28.345 | 29.948 | 2029 | 281 | `fd95fcdc...` |
| mlx `fast` | 18.581 | 20.004 | 2029 | 281 | `fd95fcdc...` |

At `batch-size 64` MLX is 1.5-2.2x faster than torch CPU and all three produced
byte-identical events. The same MLX path at `batch-size 1` took 91.3 s on this
song, slower than torch CPU, so batch shape is part of the MLX contract.

### 100.1 s song, torch CPU thread sweep

Two runs per setting, same song and contract:

| cpu threads | interop threads | generation median s |
|---|---|---|
| 1 | 1 | 40.825 |
| 2 | 1 | 41.433 |
| 4 | 1 | 41.150 |
| 6 | 1 | 40.599 |
| 8 | 1 | 40.790 |
| 4 | 4 | 41.106 |
| 8 | 8 | 41.031 |

The spread across all of these is under 2%. Thread configuration is available
for hosts where it helps, but on this Apple ARM CPU path decode is limited by
per-token latency and memory traffic rather than by parallel matmul throughput.
This also means the x86_64 result must be measured on x86_64 instead of
extrapolated from here.

### 8 s fixture, x86_64 CPU (GitHub-hosted runner)

Host: GitHub-hosted `ubuntu-latest`, x86_64, AMD EPYC 9V74 with 4 vCPU
allocated (2 cores x 2 threads), AVX-512 including `avx512_vnni` and
`avx512_bf16`, torch 2.15.0+cpu with oneDNN enabled, Python 3.12.16. Synthetic
8 s fixture, `batch-size 64`, grouped scheduler, 3 runs per setting. The runner
is shared cloud hardware, so these are that runner's numbers.

| cpu threads | interop threads | runs | generation median s | range s | tokens | notes | output SHA |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 3 | 18.5528 | 18.5060..18.6251 | 260 | 37 | `fd6563dd...` |
| 2 | 1 | 3 | 17.4855 | 17.2300..17.5448 | 260 | 37 | `fd6563dd...` |
| 4 | 1 | 3 | 20.1282 | 19.8279..20.1659 | 260 | 37 | `fd6563dd...` |

This is the opposite shape from the Apple ARM result: on x86_64 the thread count
does matter and 2 threads won on this 4-vCPU runner, while 4 threads were worse
than 1. The lesson is that the CPU thread count is a host-specific tuning knob
and cannot be copied between architectures. All nine x86_64 runs produced one
identical event SHA, the same events logic as every other backend.

### Same fixture on both architectures

To compare the two CPU architectures on one input, both hosts ran the same
generated 8 s fixture (audio SHA
`dc8dcdd504b9662707e59fa9f2dc568526a1fcb4ecff12609d33dd44c475d28d`),
`batch-size 64`, grouped scheduler, 2 runs per thread count:

| host | cpu threads | generation median s |
|---|---|---|
| Apple ARM64, 8 cores | 1 | 4.756 |
| Apple ARM64, 8 cores | 2 | 4.875 |
| Apple ARM64, 8 cores | 4 | 4.826 |
| Apple ARM64, 8 cores | 8 | 4.860 |
| x86_64 AMD EPYC 9V74, 4 vCPU | 1 | 18.553 |
| x86_64 AMD EPYC 9V74, 4 vCPU | 2 | 17.486 |
| x86_64 AMD EPYC 9V74, 4 vCPU | 4 | 20.128 |

Both produced the same event SHA, `fd6563dd...`, on the same fixture. The Apple
ARM64 host was about 3.6-4.2x faster here. These are different hardware classes
— a recent Apple SoC against a shared 4-vCPU cloud slice — so read this as a
same-input sanity check that the two CPU paths agree on the output, not as a
like-for-like CPU benchmark.

## Reporting rules

Any published backend performance statement must include the machine and CPU
architecture, thread count, torch/MLX version, device, batch size, model-load
time, generation time, whole-song wall time, the fixed seed output SHA, and the
failure boundary. Numbers from one backend or architecture are not evidence for
another.
