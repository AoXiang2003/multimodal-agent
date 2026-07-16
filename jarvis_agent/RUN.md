# Jarvis 多模态智能应答系统 — 运行指南

## 目录
1. [环境安装](#1-环境安装)
2. [三种运行模式](#2-三种运行模式)
3. [远程大模型 API 配置](#3-远程大模型-api-配置)
4. [本地小模型配置](#4-本地小模型配置)
5. [完整运行示例](#5-完整运行示例)
6. [常见问题](#6-常见问题)

---

## 1. 环境安装

### 1.1 基础依赖 (必需)

```bash
pip install faster-whisper mediapipe openai pyyaml numpy opencv-python
```

### 1.2 声纹分离 (必需 — 区分谁在说话)

```bash
pip install resemblyzer
# 如果 resemblyzer 安装失败 (Python 3.10+), 用下面这个:
pip install librosa scikit-learn
```

### 1.3 本地小模型 (可选 — 需要本地 Qwen 2B 时)

```bash
# 方案A: 用 Ollama (推荐, 最简单)
# 先去 https://ollama.com 下载安装, 然后:
ollama pull qwen2.5:2b

# 方案B: 用 transformers 直接加载 (内存占用大)
pip install transformers torch accelerate
```

### 1.4 一次性全部安装

```bash
pip install faster-whisper mediapipe openai pyyaml numpy opencv-python resemblyzer librosa scikit-learn transformers torch accelerate
```

---

## 2. 三种运行模式

### 模式A: Demo 模式 (无需数据, 3秒体验)

```bash
cd jarvis_agent
export OPENAI_API_KEY="sk-your-key-here"   # 或用 DeepSeek 的 key
python main.py --demo
```

这会用 IEMOCAP 第一段对话的模拟数据跑一遍完整流程。你会看到：
- 5 个时间窗口的实时分析
- Jarvis 风格的社交建议
- 最终的对话总结

### 模式B: 分析 IEMOCAP 视频+音频 (真实数据)

```bash
cd jarvis_agent
# 数据目录结构: Session1/Session1/dialog/avi/DivX/Ses01F_impro01.avi
#                                        /wav/Ses01F_impro01.wav
python main.py \
    --session 1 \
    --dialog Ses01F_impro01 \
    --data ../Session1/Session1 \
    --use-local   # 可选: 用本地Qwen2B做简单推理
```

**这个命令做了什么：**
```
1. 加载 Ses01F_impro01.avi → 每 20s 窗口分析面部表情
2. 加载 Ses01F_impro01.wav → Whisper转写 + 声纹分离说话人
3. 融合模块 → 对齐时间轴, 区分谁说了什么
4. Jarvis Agent → LLM推理, 生成建议
5. 对话结束 → 远程大模型生成总结
6. 保存到 output/conversation_history.db
```

### 模式C: 批量处理整个 Session

```bash
python main.py --session 1 --batch --data ../Session1/Session1
```

---

## 3. 远程大模型 API 配置

### 3.1 用 DeepSeek (推荐, 国内便宜)

编辑 `config.yaml`:

```yaml
remote_model:
  provider: "openai_compatible"
  api_base: "https://api.deepseek.com/v1"
  api_key: "${DEEPSEEK_API_KEY}"          # 环境变量, 或直接写 "sk-xxx"
  model_name: "deepseek-chat"             # DeepSeek-V3
  max_tokens: 2048
  temperature: 0.7
```

然后：
```bash
export DEEPSEEK_API_KEY="sk-your-deepseek-key"
```

### 3.2 用 OpenAI

```yaml
remote_model:
  provider: "openai_compatible"
  api_base: "https://api.openai.com/v1"
  api_key: "${OPENAI_API_KEY}"
  model_name: "gpt-4o"                    # 或 gpt-4o-mini (更便宜)
  max_tokens: 2048
  temperature: 0.7
```

### 3.3 用通义千问 (阿里)

```yaml
remote_model:
  provider: "openai_compatible"
  api_base: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  api_key: "${DASHSCOPE_API_KEY}"
  model_name: "qwen-plus"                 # 或 qwen-max, qwen-turbo
  max_tokens: 2048
  temperature: 0.7
```

### 3.4 用 SiliconFlow (国内中转, 便宜)

```yaml
remote_model:
  provider: "openai_compatible"
  api_base: "https://api.siliconflow.cn/v1"
  api_key: "${SILICONFLOW_API_KEY}"
  model_name: "deepseek-ai/DeepSeek-V3"   # 或 Qwen/Qwen2.5-7B-Instruct
  max_tokens: 2048
  temperature: 0.7
```

### 3.5 不用 API, 纯本地

编辑 `config.yaml`, 把 `local_model.provider` 改为 `ollama`, 系统会自动全部走本地模型。

---

## 4. 本地小模型配置

系统设计为**本地 Qwen 2B 做简单推理, 远程大模型做复杂推理**:

| 任务 | 路由到 | 延迟 |
|------|--------|------|
| 实时情感分析 (tension < 5) | 本地 Qwen 2B | ~1-2s |
| 复杂冲突消解 (tension > 7) | 远程 API | ~2-3s |
| 对话总结 | 远程 API | ~3-5s |
| 本地不可用 | 全部远程 API | ~2-3s |

### 4.1 方案A: Ollama (极力推荐)

```bash
# 安装 Ollama
# Windows: https://ollama.com/download/windows
# Mac: brew install ollama
# Linux: curl -fsSL https://ollama.com/install.sh | sh

# 下载 Qwen 2B
ollama pull qwen2.5:2b

# 验证
ollama run qwen2.5:2b "Hello, who are you?"
```

然后在 `config.yaml` 中:
```yaml
local_model:
  name: "qwen2.5:2b"
  provider: "ollama"               # ← 改这里
  device: "cpu"
  max_tokens: 512
  temperature: 0.7
```

### 4.2 方案B: transformers 直接加载

```bash
pip install transformers torch accelerate
```

`config.yaml`:
```yaml
local_model:
  name: "Qwen/Qwen2.5-2B-Instruct"
  provider: "transformers"          # ← 默认
  device: "cpu"                     # cpu / cuda
  max_tokens: 512
  temperature: 0.7
  quantization: null                # null / int8 / int4
```

**注意**: 2B 模型 FP16 约 4GB 内存, INT8 约 2GB。

### 4.3 方案C: 不用本地模型, 全走远程 API

把 `--use-local` 参数去掉即可, 系统会自动全部使用远程 API。

或者直接不改 config, 启动时不传 `--use-local`:
```bash
python main.py --demo   # 全部走远程 API
```

---

## 5. 完整运行示例

### 5.1 本地模型 + DeepSeek API

```bash
# 1. 设置 API Key
export DEEPSEEK_API_KEY="sk-xxxxxxxxxxxx"

# 2. 启动 Ollama
ollama serve

# 3. 确认模型可用
ollama list
# NAME            ID              SIZE      MODIFIED
# qwen2.5:2b      xxxxxxxxxxxx    1.6 GB    2 days ago

# 4. 编辑 config.yaml:
#    remote_model.api_base = "https://api.deepseek.com/v1"
#    remote_model.model_name = "deepseek-chat"
#    remote_model.api_key = "${DEEPSEEK_API_KEY}"
#    local_model.provider = "ollama"
#    local_model.name = "qwen2.5:2b"

# 5. Demo 模式
python main.py --demo

# 6. 真实数据
python main.py \
    --session 1 \
    --dialog Ses01F_impro01 \
    --data ../Session1/Session1 \
    --use-local
```

### 5.2 期望输出

```
================================================================
Initializing Jarvis Multimodal Agent System...
================================================================
✓ VideoAnalyzer initialized
✓ AudioAnalyzer initialized
  ✓ faster-whisper loaded: tiny
  ✓ SpeakerDiarizer loaded: resemblyzer
✓ FusionEngine initialized
✓ JarvisAgent initialized (local=ollama, remote=deepseek-chat)
✓ ContextManager initialized

Processing: Ses01F_impro01 (Session 1)
================================================================
Transcription: ../Session1/Session1/dialog/transcriptions/Ses01F_impro01.txt
Parsed 30 utterances, duration: 107.9s
Starting sliding window analysis: 6 windows of 20s each

--- Window 1/6 [0.0s - 20.0s] ---
  Video: frustrated (faces: 25/25)
  Speaker diarization: 8 segments, 2 speakers (SPK_0=female, SPK_1=male)
  Jarvis response generated: generated_by=local

┌──────────────────────────────────────────────────────────────┐
│  Window #01  |  Phase: opening      |  Tension: 3.5/10       │
├──────────────────────────────────────────────────────────────┤
│  👤 You 😐 (neutral    ): Excuse me...
│  🗣️  Partner 😤 (frustrated ): Who told you to get in this line?
├──────────────────────────────────────────────────────────────┤
│  🤖 JARVIS (local):
│  Speaker B is showing clear signs of frustration —
│  their voice carries tension and their expression is guarded.
│  Strategic advice: Acknowledge their concern directly rather
│  than becoming defensive...
└──────────────────────────────────────────────────────────────┘

... (5 more windows) ...

╔══════════════════════════════════════════════════════════════╗
║                 SESSION SUMMARY                              ║
║  Duration: 1.8 min | Turns: 6 | Avg Tension: 5.2/10          ║
║  📋 Summary: This was a tense conversation at a DMV office   ║
║  where Speaker B (male clerk) became frustrated...           ║
╚══════════════════════════════════════════════════════════════╝

✅ Session summary saved to: output/summaries/
✅ Full conversation history saved to: output/conversation_history.db
```

---

## 6. 常见问题

### Q: resemblyzer 安装失败 (Python 3.12+)
```bash
pip install librosa scikit-learn
# 然后 config.yaml 不用改, 系统自动降级
```

### Q: Whisper 下载模型很慢
```bash
# 设置镜像
export HF_ENDPOINT=https://hf-mirror.com
# 或者手动下载模型放到 ~/.cache/huggingface/hub/
```

### Q: 我不想分析视频, 只想跑音频
```bash
python main.py --session 1 --dialog Ses01F_impro01 --data ../Session1/Session1 --no-video
```

### Q: 本地 Qwen 2B 推理太慢
```bash
# 用 Ollama 代替 transformers, 快很多
ollama pull qwen2.5:2b
# 然后 config.yaml 中 provider 改为 "ollama"
```

### Q: 如何用其他本地模型 (Llama / Mistral / MiniCPM)
```bash
# Ollama 方式
ollama pull llama3.2:3b
# 修改 config.yaml: local_model.name = "llama3.2:3b"

# transformers 方式
# 修改 config.yaml: local_model.name = "meta-llama/Llama-3.2-3B-Instruct"
```

### Q: API key 不想写环境变量
直接在 `config.yaml` 中写:
```yaml
remote_model:
  api_key: "sk-xxxxxxxxxxxx"   # 不用 ${ENV_VAR} 格式
```
