# Changelog

## 1.2.0

- Changed current and future NoteForge releases to Apache-2.0. The license
  covers the source code, CLI, HTTP API, documentation, planner and placement
  weights, and NoteForge's distributed model weights. Apache-2.0 adds an
  explicit patent grant and patent-retaliation terms.
- Applied Apache-2.0 from v1.2.0 to the unchanged r5 weight bytes by their
  published SHA-256. The download remains hosted as the v1.0.0 Release asset,
  whose original MIT grant also remains valid.
- Kept releases through v1.1.1 under their original MIT grant. A later license
  change cannot revoke rights already granted for those released copies.
- Kept the upstream MERT-v2-FullSong dependency unchanged. Its checkpoint is
  not distributed here and remains CC BY-NC 4.0, so the MERT runtime path
  remains limited to non-commercial use without separate permission.
- Added `LICENSING.md` to separate current NoteForge licensing, historical MIT
  releases, upstream MERT terms, and closed training materials.

## 1.1.1

- Added `noteforge mert status`, `noteforge mert fetch`, and
  `noteforge mert export-mlx`. The pinned MERT source is shared by torch
  CPU/CUDA/MPS and MLX; `mlx-mert` is only an optional derived bf16 cache for
  the Apple MLX path. `noteforge export mlx-mert` remains as a compatibility
  alias.
- `doctor --fix` no longer downloads the pinned 2.4 GB MERT checkpoint just
  because `--mert-source hf` is present. Downloads require an explicit
  `noteforge mert fetch --source hf --out DIR` command or `doctor --fix
  --download`.
- Moved backend-neutral MERT source resolution out of `noteforge.mlx` so the
  CLI and product docs no longer imply that only MLX may obtain or use MERT.

## 1.1.0

- Added the unified `noteforge` console script with subcommands `doctor`,
  `draft`, `extract`, `plan`, `auto`, `bench`, `fixture`, `export`, `serve`,
  and `version`. It is a thin forwarder, so every subcommand keeps its module's
  own parser and option list (`noteforge draft --help` equals
  `draft_song --help`). `noteforge help <cmd>` and `noteforge <cmd> --help` both
  work.
- `noteforge extract` is `draft_song --extract-only` with the flag injected when
  absent; `noteforge serve` wraps `uvicorn noteforge_api.asgi:app` and exits 2
  with a clear message when `NOTEFORGE_API_TOKEN` or `uvicorn` is missing.
- `noteforge plan` exposes the density planner, `noteforge auto` runs the
  plan-draft-BPM pipeline, and `noteforge export mlx|mlx-mert` performs the
  one-time MLX exports. A re-export of the published r5 weights reproduces the
  existing `models/mlx-fast/weights.safetensors` byte for byte.
- `noteforge inspect` reads a chart (`.osu`, RPE/official JSON, or a Phira
  package) and prints note count, model-usable timing segments, and the opening
  type shares as JSON.
- `plan_song.py` and `auto_draft.py` now accept an argument list, default
  `--device` to `auto` and the planner folder to the published
  `models/planner_f`, so they work on Apple Silicon without CUDA or a `planner`
  symlink.
- Fixed `--flick-ratio` / `--drag-ratio` on every non-continuous decode path.
  The `kind_bias` decode steer existed only in `ContinuousDecoder`, so the MLX
  backend and `--scheduler grouped|legacy` raised
  `TypeError: generate_batch() got an unexpected keyword argument 'kind_bias'`
  instead of applying the caller's type ratio. `generate_batch` now accepts the
  bias and applies it in both the fast and reference loops, and the scheduler
  restriction is gone. On one real 8 s Apple Silicon input, `torch` CPU and
  `mlx` returned the identical event hash with `--flick-ratio 0.9`
  (`70c8ff39464dbebc`); the same input changed from 33 taps/5 drags/2 holds to
  14 flicks/16 taps/3 holds. The receipt now records the applied bias.
- `noteforge version` now reports the package source version rather than stale
  installed metadata from an editable install.
- Fixed `noteforge export mlx-mert --mert-source hf`, which raised
  `TypeError: argument should be a str or an os.PathLike object ... not 'NoneType'`:
  `resolve_mert_source('hf')` returns a `None` directory to mean "use the pinned
  Hugging Face revision", and that `None` was passed straight into `Path`. The
  export now downloads that exact revision with `huggingface_hub` and validates
  the four pinned SHA-256 files before writing anything. `doctor --fix` uses the
  same path.
- Corrected the MERT-v2-FullSong license in `NOTICE` and the README: the pinned
  checkpoint weights are CC BY-NC 4.0 (non-commercial), not Apache-2.0. The
  repository distributes loading code only and no MERT weight file, so runs that
  need MERT are non-commercial. The MIT r5 license does not relicense MERT.

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
