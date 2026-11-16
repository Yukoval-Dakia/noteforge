# Changelog

## 1.0.1

- Added `python -m noteforge.scripts.doctor` (and the `noteforge-doctor`
  console script): it checks ffmpeg/ffprobe, the runtime dependencies, the
  committed weight SHA-256, the MERT source and the resolved backend, exits 1
  when the host cannot draft, and supports `--json` and `--fix`.
- `--fix` builds the conventional MLX exports on Apple Silicon
  (`models/mlx-fast`, `models/mlx-mert`); those directories are now gitignored.
- `auto` now discovers those conventional export folders itself, so an Apple
  Silicon user who ran `doctor --fix` gets MLX with no environment variables.
  An explicit `NOTEFORGE_FAST_WEIGHTS` / `NOTEFORGE_FAST_MERT` still wins.
- Renamed the Apple Silicon backend to `mlx` everywhere: package,
  environment variables, API backend value, dependency extra, requirements file
  and test entry. The old names are gone rather than kept as aliases.
- Added `noteforge.backends` with one selection contract: `auto`, `torch`,
  `mlx`, and an `auto|cuda|mps|cpu` device resolver that fails closed when an
  explicit device is unavailable.
- `NOTEFORGE_BACKEND` and `NOTEFORGE_DEVICE` default to `auto`. `/v1/health` and
  `/v1/capabilities` now report the requested values, the resolved backend and
  device, the backend error when unusable, and host availability for PyTorch,
  MLX and CPU.
- Added `--cpu-threads` / `--cpu-interop-threads` and
  `NOTEFORGE_CPU_THREADS` / `NOTEFORGE_CPU_INTEROP_THREADS`. Removed the
  import-time `torch.set_num_threads(3)` in `noteforge.placement.model`, which
  silently overrode the caller's choice. Importing that module now leaves the
  thread count untouched.
- Decode receipts now record the active backend, requested and resolved device,
  thread overrides, a torch identity (version, selected device, CUDA name and
  capability when present, CPU threads) and, for MLX, the MLX version and
  default device.
- Fixed Phira placement on an empty draft: `layout` returns no notes instead of
  indexing a 1-D array with 2 indices.
- Added `python -m noteforge.scripts.bench_backends` and
  `python -m noteforge.scripts.make_bench_fixture` for reproducible, hard-gated
  backend measurement that never rewrites the chart.
- Added `torchaudio` to core requirements (the pinned MERT modeling file imports
  it), a cross-platform unit-test workflow, and a manual GitHub Actions
  x86_64 CPU benchmark workflow.

The r5 weights are unchanged: `noteforge-r5.pt`
`1b1f2e8dee82101690778ef667b661fa1d88421f50034c0bd97572f3067d405b`.

Measured on Apple Silicon (arm64, 8 cores) with the published r5 weights, a 100.1 s Phira
song, one seed, `phira-level 10`, `profile standard`, `batch-size 64`:

| backend | generation s | notes | token | output SHA |
|---|---|---|---|---|
| `torch` CPU, 4 threads | 41.8 | 281 | 2029 | `fd95fcdc...` |
| `mlx` plan `balanced` | 28.3 | 281 | 2029 | `fd95fcdc...` |
| `mlx` plan `fast` | 18.6 | 281 | 2029 | `fd95fcdc...` |

The same fixed input under `torch` CPU produced identical events to both MLX
plans. This is an Apple ARM result under MLX and PyTorch CPU; it is not an
x86_64 or NVIDIA number, and MLX at `batch-size 1` was slower than CPU, so the
batch shape is part of the contract.

Also on that host:

- The decode entry points use `torch.inference_mode()` instead of
  `torch.no_grad()`: 40.995 s vs 41.755 s median whole-song generation (three
  runs each, about 1.8% faster) with identical events.
- Model construction now runs on the meta device and the published weights are
  assigned in, skipping the random initialisation that loading immediately
  overwrote: weight loading fell from 1.338 s to about 0.52 s with bit-identical
  parameters and the same events.
- CPU thread count is not a speed knob on this path: 1/2/4/6/8 threads and
  1/4/8 interop threads all land within 2% on the 100.1 s song.
- `--autocast bf16` on CPU is about 2.2x slower than fp32 (3.37 s vs 1.56 s) and
  produced the same events on the short fixture.
- `--compile-decode` on the CPU path is markedly slower (about 6.6 s vs 1.56 s
  eager after the first compile run).

Full rows, including every raw timing and the canonical note-multiset SHA, are
in `reports/backend-benchmark-20261116.json` and `docs/BACKENDS.md`.
