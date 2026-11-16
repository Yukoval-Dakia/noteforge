# NoteForge

NoteForge 给一首歌生成可编辑的节奏谱草稿：Phira / Phigros 风格的时间、类型、持续信息，以及 osu!mania 4K 的时间、类型和四键位置。输出是草稿，不是成品谱。

本仓库开源 **r5 模型权重、推理代码、HTTP 接口和谱面导出**。训练代码、训练数据、训练方法与内部评测面板保留在私有研究仓库。

## 仓库内容

| 部分 | 位置 | 说明 |
|---|---|---|
| 推理引擎 | `noteforge/` | MERT-v2 + 10 ms mel 特征、整曲滚动生成、共识聚合、谱面解析与导出 |
| HTTP 接口 | `noteforge_api/` | 上传音频或已有谱、异步生成、分段选择、修订、导出；OpenAPI 见 `openapi.json` |
| Apple MLX 路径 | `noteforge/mlx/` | Apple Silicon 上的独立 fp16 图执行实现；MERT 仍使用共享来源 |
| 辅助权重 | `models/planner_f/`、`noteforge/placement/` | 密度曲线规划器和 Phira 位置摆放模型 |
| 推理合同 | `models/config/` | 冻结的 mel 归一化系数和 r5 推理参数 |
| 权重清单 | `models/MANIFEST.json` | 每个权重文件的字节数和 SHA-256 |

主模型是约 3.22 亿参数的自回归事件生成器，支持 `mania_4k` 与 `phira`。整曲按 8 秒核心滚动解码；产品默认 8 次采样、T=1、anchor 共识、3/8 票。产品接口允许调用者提供密度、风格、节奏和用户已有谱；用户自己的音符原样保留。

## 权重下载

`models/noteforge-r5.pt` 约 1.2 GB，作为 [v1.0.0 Release 资产](https://github.com/Yukoval-Dakia/noteforge/releases/download/v1.0.0/noteforge-r5.pt) 发布：

```bash
curl -L -o models/noteforge-r5.pt \
  https://github.com/Yukoval-Dakia/noteforge/releases/download/v1.0.0/noteforge-r5.pt
shasum -a 256 models/noteforge-r5.pt
# 1b1f2e8dee82101690778ef667b661fa1d88421f50034c0bd97572f3067d405b
```

小权重和推理系数已随仓分发；`models/MANIFEST.json` 给出全部 SHA。`noteforge.finalmodelv1` 文件只含 config、vocabulary、EMA 权重和推理合同，不含 optimizer、训练计划或语料信息。公开 mel 系数只含 128 维 mean/std 及其 SHA。

MERT-v2 编码器权重不在本仓库。代码固定 `m-a-p/MERT-v2-FullSong` revision `7001f7657276238629e61c7746ce70735e0604c0`。同一份 MERT 来源供 torch CPU、CUDA、MPS 和 MLX 共用；可用 `--mert-source hf` 按需读取，或先显式保存到本地：

```bash
noteforge mert status --source hf
noteforge mert fetch --source hf --out models/mert2-source
```

`mert fetch --source hf` 会下载约 2.4 GB 权重，因此只在你主动执行时发生。`models/mlx-mert` 不是另一份上游模型，也不是其他后端的限制条件；它是 Apple MLX 路径可选使用的 bf16 派生缓存。首次真实特征提取需要 MERT、`torchaudio` 和 `ffmpeg`/`ffprobe`。

## 安装与命令行

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[api]'
# 系统依赖：ffmpeg / ffprobe
```

## 一个入口：noteforge CLI

安装后只有一个命令，所有能力都在它下面。每个子命令把自己的参数原样交给对应模块，所以 `noteforge draft --help` 与 `draft_song --help` 完全一致，不会丢掉任何选项。

```bash
noteforge doctor      # 自检并说明缺什么（--fix 构建 MLX 导出）
noteforge extract     # 只提取可复用特征，然后停下（= draft --extract-only）
noteforge plan        # 只规划分段密度曲线，不出谱
noteforge draft       # 生成谱面草稿（默认 auto：Apple Silicon 走 MLX，否则 PyTorch）
noteforge auto        # 计划 + 出谱 + 可选 BPM 对齐，一条命令跑完
noteforge inspect     # 读一个谱面文件，打印音符数、时间分段和开头类型份额
noteforge bench       # 在固定输入上测一个后端，输出事件 SHA 便于比对
noteforge fixture     # 写合成特征夹具，用于不含歌曲/编码器的基准
noteforge mert        # 检查/获取共享 MERT 来源，或生成可选的 MLX bf16 缓存
noteforge export      # 把 r5 导出为 MLX fp16（`mlx-mert` 为兼容别名）
noteforge serve       # 启动 HTTP API（= uvicorn noteforge_api.asgi:app）
noteforge version     # 打印版本
```

典型流程：

```bash
noteforge doctor --fix                 # 首次：检查环境，Apple Silicon 顺带建好 MLX 导出
noteforge extract --audio song.mp3 --features-dir ./features/song --out-dir ./work/feat \
  --mert-source hf
noteforge draft --audio song.mp3 --features-dir ./features/song --out-dir ./work/song \
  --model models/noteforge-r5.pt --plan models/config/r5-inference.json \
  --mode phira --phira-level 10 --consensus 8
noteforge serve                        # 需要先设置 NOTEFORGE_API_TOKEN
```

`noteforge auto` 适合不想分两步的用法：它先用规划器给出密度曲线（以及 Phira 的类型份额），再用单样本、history 和自己前一段出谱；带 `--bpm` 时先跑一遍取得相位，再回到最终一次。

Phira 的 `--flick-ratio` / `--drag-ratio` 是解码期偏好（`kind_bias`），不是模型输出；`--kind-as-band` 则把份额作为请求带给模型。两种后端、`continuous`/`grouped`/`legacy` 三种调度都应用同一条规则，回执的 `decoding.kind_bias` 记录实际生效值。实测：同一 8 秒输入、同一 seed，torch CPU 与 mlx 在 `--flick-ratio 0.9` 下输出完全相同的事件哈希。

```bash
noteforge auto --audio song.mp3 --mode phira --level 6 --bpm 174 \
  --features-dir ./features/song --model models/noteforge-r5.pt \
  --plan models/config/r5-inference.json --out-dir ./work/song
```

`noteforge serve` 在缺少 `NOTEFORGE_API_TOKEN` 或未安装 `uvicorn` 时以退出码 2 报错并说明原因，不会静默起一个没有鉴权的服务。`noteforge export mlx` 的输出与 `doctor --fix` 写出的目录逐字节一致（同一权重会得到同一 `weights.safetensors` SHA-256）。`noteforge help <command>` 与 `noteforge <command> --help` 等价。原有的 `python -m noteforge.scripts.<name>`、`python -m noteforge.planner.<name>` 与 `noteforge-doctor` 入口继续可用。

## 环境自检：doctor

安装后先跑一次自检，它会检查 ffmpeg/ffprobe、依赖、权重 SHA-256、MERT 源和可用后端，并说明缺什么：

```bash
python -m noteforge.scripts.doctor          # 人类可读报告
python -m noteforge.scripts.doctor --json   # 同样内容，JSON（可用于 CI）
python -m noteforge.scripts.doctor --fix    # 使用已在磁盘上的来源构建 MLX 导出（Apple Silicon）
```

退出码 0 表示这台机器可以直接出谱，1 表示还缺必需项；`warn`/`info` 不影响退出码。`--fix` 只使用已经在磁盘上的来源，在 Apple Silicon 上构建 `models/mlx-fast` 与可选的 `models/mlx-mert`，然后重新解析后端。它不会在后台下载 MERT；需要下载时必须显式运行 `noteforge mert fetch`，或给 `doctor --fix` 同时传入 `--download`。

## 推理后端

| `NOTEFORGE_BACKEND` | 设备 | 实现 |
|---|---|---|
| `auto` | 自动 | Apple arm64 且能用 MLX 导出时选 `mlx`，否则选 `torch` |
| `mlx` | Apple Silicon | `mlx.core` fp16 静态缓存解码；MERT 前端使用 PyTorch MPS |
| `torch` + `NOTEFORGE_DEVICE=auto` | CUDA / MPS / CPU | PyTorch，按 NVIDIA、Apple Metal、CPU 的顺序选择 |
| `torch` + `NOTEFORGE_DEVICE=cuda` | NVIDIA | bf16 autocast（显式开启）、静态缓存和 CUDA Graph |
| `torch` + `NOTEFORGE_DEVICE=cpu` | x86_64 / arm64 CPU | PyTorch CPU/oneDNN/Accelerate 路径，可控制线程数 |

`mlx`、`mps`、`cuda` 和 `cpu` 是不同后端，不共用性能结论。`auto` 只有在 Apple arm64、已安装 `mlx` 且能用 MLX 导出时才会选 MLX；导出权重可以放在约定目录（`models/mlx-fast`、`models/mlx-mert`，`doctor --fix` 就写这里），也可以用 `NOTEFORGE_FAST_WEIGHTS` / `NOTEFORGE_FAST_MERT` 或 CLI 参数显式指定。显式指定的路径优先，`auto` 不会覆盖它。

### Apple Silicon MLX

```bash
pip install -e '.[api,mlx]'

# 一条命令导出两份 MLX 权重并自动检测
python -m noteforge.scripts.doctor --fix

# 主模型 fp16 MLX 导出
python -m noteforge.mlx.fast_model export models/noteforge-r5.pt ./models/mlx-fast

# 可选：从所有后端共用的 MERT 来源生成 MLX bf16 缓存
noteforge mert export-mlx --source /path/to/pinned-mert2 --out ./models/mlx-mert

export NOTEFORGE_BACKEND=mlx
export NOTEFORGE_FAST_WEIGHTS=$PWD/models/mlx-fast
export NOTEFORGE_FAST_MERT=$PWD/models/mlx-mert
NOTEFORGE_API_TOKEN=secret uvicorn noteforge_api.asgi:app --port 8787
```

MLX 生成分支的默认计划是 `balanced`。API 请求里的 `plan` 可取 `balanced`、`fast` 或 `sequential`；这些是解码调度差异，不改变 r5 权重或共识规则。`models/mlx-mert` 只服务 MLX 路径的加载速度；torch CPU/CUDA/MPS 直接读取同一份 MERT 来源，不需要它。

### x86_64 / ARM64 CPU

```bash
NOTEFORGE_BACKEND=torch NOTEFORGE_DEVICE=cpu \
NOTEFORGE_CPU_THREADS=8 NOTEFORGE_CPU_INTEROP_THREADS=1 \
NOTEFORGE_API_TOKEN=secret uvicorn noteforge_api.asgi:app --port 8787
```

CLI 同样支持 `--device cpu --cpu-threads N --cpu-interop-threads N`。x86_64 和 ARM64 共用 PyTorch 推理合同，但 BLAS/oneDNN/Accelerate 实现不同，发布性能时必须分别注明 CPU 架构、线程数和 torch/BLAS 版本。

### NVIDIA CUDA

```bash
NOTEFORGE_BACKEND=torch NOTEFORGE_DEVICE=cuda \
NOTEFORGE_API_TOKEN=secret uvicorn noteforge_api.asgi:app --port 8787
```

默认启用静态缓存、连续调度和 CUDA Graph；`NOTEFORGE_DECODE_CUDA_GRAPH=0` 可关闭图捕获。CLI 的 `--autocast bf16` 是显式数值路径变更，不能和 fp32 输出称为逐位等价。`--reference-decode` 可回退到参考解码循环用于对照。

`GET /v1/health` 返回实际选中的 `backend`、`device`，以及请求值和后端错误；`GET /v1/capabilities` 还会列出当前可用的 PyTorch/MLX 能力。

从仓库根目录执行：

```bash
# 提取可复用特征
python -m noteforge.scripts.draft_song --extract-only \
  --audio song.mp3 --features-dir ./features/song --out-dir ./work/song-features \
  --mert-source hf

# 生成谱面草稿
python -m noteforge.scripts.draft_song \
  --audio song.mp3 --mode phira --model models/noteforge-r5.pt \
  --plan models/config/r5-inference.json \
  --features-dir ./features/song --out-dir ./work/song \
  --density 6.5 --consensus 8
```

`--tier low|medium|high`、`--density`、`--phira-level`、`--mania-star` 控制整体密度；`--timing`、`--timing-segments`、`--beat-grid` 提供可见节奏输入；`--history-reference` 让每个核心看到模型自己的前一段。完整参数：

```bash
python -m noteforge.scripts.draft_song --help
```

输出包括 `draft-events.json`、`.pez`/`.osz` 或 `.osu` 谱面和 `draft-receipt.json`；回执记录模型、音频、特征、条件、时钟与导出身份。

## HTTP 接口

```bash
NOTEFORGE_API_TOKEN=secret \
uvicorn noteforge_api.asgi:app --host 0.0.0.0 --port 8787
```

默认模型、推理合同、规划器和 mel 系数都在仓库内；可用 `NOTEFORGE_MODEL`、`NOTEFORGE_PLAN`、`NOTEFORGE_PLANNER_DIR`、`NOTEFORGE_MEL_STATS`、`NOTEFORGE_BACKEND`、`NOTEFORGE_DEVICE`、`NOTEFORGE_FAST_WEIGHTS`、`NOTEFORGE_FAST_MERT` 覆盖。流程是 `POST /v1/audio` → `POST /v1/drafts` → 轮询 `GET /v1/drafts/{id}` → `GET /v1/drafts/{id}/events` 或 `/chart`。交互文档在 `/docs`，机器描述在 `openapi.json`。

`NOTEFORGE_FAKE_ENGINE=1` 用确定性假音符跑通接口、修订和导出，仅用于测试，不能代表模型质量。

## 验证

```bash
pip install -e '.[dev,api]'
NOTEFORGE_API_TOKEN=secret python -m pytest tests/test_api.py -q
python -m pytest tests/test_runtime.py -q
```

该测试覆盖鉴权、上传、草稿状态、分段、修订和两种谱面导出；生成步骤使用假引擎，后处理使用真实代码。真实整曲生成还需下载主模型和 MERT-v2 编码器，且需要足够显存或 Apple Silicon。

## 基准测量

同一音频、seed 和解码合同下比较后端：

```bash
python -m noteforge.scripts.make_bench_fixture --out ./bench --seconds 8
python -m noteforge.scripts.bench_backends \
  --backend torch --device cpu --cpu-threads 4 \
  --audio ./bench/silence.wav --features-dir ./bench \
  --mode phira --level 10 --batch-size 64 --runs 3
```

`bench_backends` 输出模型加载、生成、总墙钟、token 数、音符数和事件 SHA，不删音符、不改共识、不跳过合法性检查。合成夹具只测运行时，不代表谱面质量。x86_64 数字必须来自 x86_64 主机；仓库提供 `.github/workflows/cpu-x86-benchmark.yml` 手动跑真实 x86 runner。

已在 Apple Silicon（arm64，8 核）、`torch` 2.15.0、`mlx` 0.32.4 上实测的结论见 [docs/BACKENDS.md](docs/BACKENDS.md) 和原始行 [reports/backend-benchmark-20261116.json](reports/backend-benchmark-20261116.json)。要点：100.1 s 整曲 `batch-size 64` 下 torch CPU 约 41 s、MLX `balanced` 28.3 s、MLX `fast` 18.6 s，三者事件完全一致；CPU 线程数（1–8）在这条路径上差异 <2%；`--compile-decode` 和 CPU 上的 `--autocast bf16` 都更慢。CUDA 与 x86_64 尚无本机数字，没有实测就不发布数值。

发布任何后端性能结论时，需要同时给出：机器与 CPU 架构、线程数、torch/MLX 版本、设备、批大小、模型加载时间、生成时间、整曲墙钟、RSS、固定 seed 输出 SHA 和失败边界。

## 许可证与归属

仓库代码、随仓小权重和发布的 r5 主权重采用 MIT License，见 `LICENSE`。第三方组件及许可见 `NOTICE`：PupuM2D（MIT）、librosa（ISC）、timm 参考实现（Apache-2.0）。MERT-v2-FullSong 是例外：仓库只包含加载它的代码，不分发其权重；上游固定 revision 的权重许可是 **CC BY-NC 4.0（非商用）**，不是 Apache-2.0，需要 MERT 的运行因此仅限非商用。第三方谱面和音频不在本仓库，也不因代码许可获得再分发或训练授权。

训练方法、训练代码、语料和内部评测不属于本开源发布范围。
