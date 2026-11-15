# NoteForge

NoteForge 给一首歌生成可编辑的节奏谱草稿：Phira / Phigros 风格的时间、类型、持续信息，以及 osu!mania 4K 的时间、类型和四键位置。输出是草稿，不是成品谱。

本仓库开源 **r5 模型权重、推理代码、HTTP 接口和谱面导出**。训练代码、训练数据、训练方法与内部评测面板保留在私有研究仓库。

## 仓库内容

| 部分 | 位置 | 说明 |
|---|---|---|
| 推理引擎 | `noteforge/` | MERT-v2 + 10 ms mel 特征、整曲滚动生成、共识聚合、谱面解析与导出 |
| HTTP 接口 | `noteforge_api/` | 上传音频或已有谱、异步生成、分段选择、修订、导出；OpenAPI 见 `openapi.json` |
| 可选 MLX 路径 | `noteforge/m3/` | Apple Silicon 上的 fp16 推理镜像 |
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

MERT-v2 编码器权重不在本仓库。代码固定 `m-a-p/MERT-v2-FullSong` revision `7001f7657276238629e61c7746ce70735e0604c0`，可用 `--mert-source hf` 下载，或提供包含固定 SHA 文件的本地快照。首次真实特征提取需要该编码器和 `ffmpeg`/`ffprobe`。

## 安装与命令行

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[api]'
# 系统依赖：ffmpeg / ffprobe
```

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

默认模型、推理合同、规划器和 mel 系数都在仓库内；可用 `NOTEFORGE_MODEL`、`NOTEFORGE_PLAN`、`NOTEFORGE_PLANNER_DIR`、`NOTEFORGE_MEL_STATS`、`NOTEFORGE_DEVICE` 覆盖。流程是 `POST /v1/audio` → `POST /v1/drafts` → 轮询 `GET /v1/drafts/{id}` → `GET /v1/drafts/{id}/events` 或 `/chart`。交互文档在 `/docs`，机器描述在 `openapi.json`。

`NOTEFORGE_FAKE_ENGINE=1` 用确定性假音符跑通接口、修订和导出，仅用于测试，不能代表模型质量。

## 验证

```bash
pip install -e '.[dev,api]'
NOTEFORGE_API_TOKEN=secret python -m pytest tests/test_api.py -q
```

该测试覆盖鉴权、上传、草稿状态、分段、修订和两种谱面导出；生成步骤使用假引擎，后处理使用真实代码。真实整曲生成还需下载主模型和 MERT-v2 编码器，且需要足够显存或 Apple Silicon。

## 许可证与归属

仓库代码、随仓小权重和发布的 r5 主权重采用 MIT License，见 `LICENSE`。第三方组件及许可见 `NOTICE`：PupuM2D（MIT）、MERT-v2-FullSong（Apache-2.0）、librosa（ISC）、timm 参考实现（Apache-2.0）。第三方谱面和音频不在本仓库，也不因代码许可获得再分发或训练授权。

训练方法、训练代码、语料和内部评测不属于本开源发布范围。
