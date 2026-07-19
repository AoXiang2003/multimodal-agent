"""
Jarvis Agent — 核心智能体编排器
====================================
职责: 接收多模态观测数据, 使用 LLM (本地 Qwen 2B + 远程大模型 API)
      进行推理, 生成个性化应答建议

架构:
  ┌──────────────────────────────────────────┐
  │           JarvisAgent                     │
  │  ┌──────────────┐  ┌──────────────────┐  │
  │  │ Local Model   │  │  Remote Model    │  │
  │  │ (Qwen 2B)    │  │  (GPT-4o / etc.) │  │
  │  │ 轻量级推理    │  │  复杂推理/总结    │  │
  │  └──────┬───────┘  └───────┬──────────┘  │
  │         │                  │              │
  │         └────────┬─────────┘              │
  │                  ▼                        │
  │         ┌──────────────┐                  │
  │         │ Model Router  │                 │
  │         │ (任务分流)     │                 │
  │         └──────────────┘                  │
  └──────────────────────────────────────────┘

任务分流策略:
  - 简单任务 (情感解读、单轮建议) → 本地 Qwen 2B (低延迟)
  - 复杂任务 (多轮推理、冲突消解、对话总结) → 远程大模型 API
  - 本地不可用时 → 全部走远程 API
"""

import os
import json
import logging
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
import time
import numpy as np
from dataclasses import dataclass,field

@dataclass
class MultimodalObservation:
    timestamp: float = 0.0; window_start: float = 0.0; window_end: float = 0.0
    user_emotion: str = "neutral"; partner_emotion: str = "neutral"
    user_emotion_conf: float = 0.5; partner_emotion_conf: float = 0.5
    user_valence: float = 2.5; user_arousal: float = 2.5
    partner_valence: float = 2.5; partner_arousal: float = 2.5
    user_speech: str = ""; partner_speech: str = ""
    full_transcript: str = ""
    scene_context: str = "conversation"; dialog_type: str = "improvisation"
    emotion_trend: str = "stable"
    # ===== 窗口内每句话的概率向量（用于 JSD 计算） =====
    user_voice_probs_list: list = field(default_factory=list)  # 每项是 dict（9类概率）
    partner_voice_probs_list: list = field(default_factory=list)
    user_face_probs_list: list = field(default_factory=list)  # 每项是 7维 list
    partner_face_probs_list: list = field(default_factory=list)
    user_face_valid_ratios: list = field(default_factory=list)
    partner_face_valid_ratios: list = field(default_factory=list)
    # ===== 自适应决策门限：JSD + 不确定性 + 候选冲突句 =====
    jsd_user: float = 0.0
    jsd_partner: float = 0.0
    face_uncertainty_user: float = 0.0
    face_uncertainty_partner: float = 0.0
    top_conflicts_user: list = field(default_factory=list)
    top_conflicts_partner: list = field(default_factory=list)
    # JSD ≤ 0.2 但模态不可靠的句子（无冲突，仅标记质量）
    low_conflicts_user: list = field(default_factory=list)
    low_conflicts_partner: list = field(default_factory=list)

    def to_dict(self):
        return {
            "user_emotion": self.user_emotion,
            "user_emotion_conf": self.user_emotion_conf,
            "user_emotion_trend": self.emotion_trend,
            "user_speech": self.user_speech,
            "partner_emotion": self.partner_emotion,
            "partner_emotion_conf": self.partner_emotion_conf,
            "partner_emotion_trend": self.emotion_trend,
            "partner_speech": self.partner_speech,
            "scene_context": self.scene_context,
            "full_transcript": self.full_transcript,
            "dialog_type": self.dialog_type,
        }
from .context_manager import ContextManager, SessionSummary
from prompts.system_prompts import (
    JARVIS_PERSONA,
    JARVIS_PERSONA_LOCAL,
    LOCAL_EMOTION_SUMMARY_PROMPT,
    REALTIME_ANALYSIS_PROMPT,
    REALTIME_ANALYSIS_PROMPT_WITH_SUMMARY,
    REALTIME_ANALYSIS_PROMPT_LOCAL,
    SESSION_SUMMARY_PROMPT,
    PERSONALIZATION_INJECTION,
    EMOTION_LABELS,
    STAGE1_ANALYSIS_PROMPT,
    STAGE2_DECISION_PROMPT,
)

logger = logging.getLogger(__name__)


@dataclass
class JarvisResponse:
    """Jarvis Agent 的响应"""
    session_id: str
    timestamp: float
    analysis: str                       # 核心分析建议
    user_emotion_label: str
    partner_emotion_label: str
    suggestion: str = ""                # 具体的行动建议
    risk_alert: Optional[str] = None    # 风险警告
    generated_by: str = "remote"        # 由哪个模型生成
    tension_level: float = 0.0          # 已废弃，仅 SQLite 兼容


class JarvisAgent:
    """
    Jarvis 智能体 — 多模态社交对话教练

    使用方式:
        agent = JarvisAgent(config)
        agent.initialize()

        # 每 20 秒调用一次
        response = agent.analyze_and_respond(observation, session_id)

        # 对话结束
        summary = agent.generate_session_summary(session_id)
    """

    def __init__(self, config: Dict):
        self.config = config
        self.agent_config = config.get("agent", {})
        self.local_config = config.get("local_model", {})
        self.remote_config = config.get("remote_model", {})
        self.utt_per_window = config.get("agent", {}).get("utterances_per_window", 14)

        # 模型引用 (延迟加载)
        self.local_model = None
        self.local_tokenizer = None
        self.remote_client = None

        # 上下文管理器
        self.context_manager = ContextManager(config)

        # 状态
        self.is_initialized = False
        self.current_session_id: Optional[str] = None

        logger.info("JarvisAgent created (not yet initialized)")

    def initialize(self):
        """
        初始化: 加载本地模型 + 配置远程 API

        本地模型 (Qwen 2B): 使用 transformers 或 API 加载
        - 如果配置了 ollama, 优先使用 (更简单)
        - 否则使用 transformers 直接加载
        """
        # --- 尝试加载本地 Qwen 2B ---
        local_provider = self.local_config.get("provider", "transformers")

        if local_provider == "ollama":
            self._init_ollama()
        elif local_provider == "transformers":
            self._init_transformers()
        elif local_provider == "vllm":
            self._init_vllm()

        # --- 配置远程 API ---
        self._init_remote_api()

        self.is_initialized = True
        logger.info(f"JarvisAgent initialized: local={local_provider}, "
                    f"remote={self.remote_config.get('provider')}")

    def _init_ollama(self):
        """通过 Ollama 使用本地 Qwen 2B"""
        try:
            import requests
            # 检查 Ollama 服务是否运行
            resp = requests.get("http://localhost:11434/api/tags", timeout=2)
            if resp.status_code == 200:
                self.local_model = "ollama"  # 标记为可用
                self.local_model_name = self.local_config.get(
                    "name", "qwen2.5:2b"
                )
                logger.info(f"Ollama connected, model: {self.local_model_name}")
            else:
                logger.warning("Ollama not running, will use remote API only")
                self.local_model = None
        except Exception as e:
            logger.warning(f"Ollama not available: {e}, will use remote API only")
            self.local_model = None

    def _init_transformers(self):
        """通过 transformers 加载本地 Qwen 2B"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            model_name = self.local_config.get(
                "model_path"
            ) or self.local_config.get("name", "Qwen/Qwen2.5-2B-Instruct")

            device = self.local_config.get("device", "cpu")
            quant = self.local_config.get("quantization")

            logger.info(f"Loading local model: {model_name} (device={device})")

            load_kwargs = {"trust_remote_code": True}

            if quant == "int8":
                load_kwargs["load_in_8bit"] = True
            elif quant == "int4":
                load_kwargs["load_in_4bit"] = True

            if device == "cpu":
                load_kwargs["device_map"] = "cpu"
                load_kwargs["torch_dtype"] = torch.float32
            elif device == "cuda":
                load_kwargs["device_map"] = "auto"
                load_kwargs["torch_dtype"] = torch.float16

            self.local_tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.local_model = AutoModelForCausalLM.from_pretrained(
                model_name, **load_kwargs
            )
            self.local_model_name = model_name

            logger.info(f"Local model loaded: {model_name}")

        except ImportError:
            logger.warning(
                "transformers/torch not installed. "
                "Install: pip install transformers torch. "
                "Will use remote API only."
            )
            self.local_model = None
        except Exception as e:
            logger.warning(f"Failed to load local model: {e}. Will use remote API only.")
            self.local_model = None

    def _init_vllm(self):
        """通过 vLLM API 使用本地模型"""
        try:
            import requests
            vllm_url = self.local_config.get("api_base", "http://localhost:8000/v1")
            resp = requests.get(f"{vllm_url}/models", timeout=2)
            if resp.status_code == 200:
                self.local_model = "vllm"
                self.local_model_name = self.local_config.get("name", "qwen")
                self.local_api_base = vllm_url
                logger.info(f"vLLM connected at {vllm_url}")
            else:
                logger.warning("vLLM not available")
                self.local_model = None
        except Exception as e:
            logger.warning(f"vLLM not available: {e}")
            self.local_model = None

    def _init_remote_api(self):
        """初始化远程 API 客户端 (OpenAI 兼容格式)"""
        try:
            from openai import OpenAI

            api_base = self.remote_config.get("api_base", "https://api.openai.com/v1")
            api_key = self.remote_config.get("api_key", os.environ.get("OPENAI_API_KEY", ""))

            if api_key:
                self.remote_client = OpenAI(
                    base_url=api_base,
                    api_key=api_key,
                )
                self.remote_model_name = self.remote_config.get("model_name", "gpt-4o")
                logger.info(f"Remote API configured: {self.remote_model_name}")
            else:
                logger.warning("No API key for remote model. Set OPENAI_API_KEY env var.")
                self.remote_client = None
        except ImportError:
            logger.warning("openai package not installed. Install: pip install openai")
            self.remote_client = None

    # ================================================================
    # 核心推理接口
    # ================================================================

    def analyze_and_respond(
        self,
        observation: MultimodalObservation,
        session_id: str,
        use_local: bool = False,
    ) -> JarvisResponse:
        """
        核心推理: 本地小模型先总结情绪特征 → 远程大模型再决策

        流水线:
        1. 本地模型统计窗口内情绪分布 (最多/最少/矛盾/趋势)
        2. 远程模型接收本地摘要 + 原始对话 → 场景/策略/工具

        Args:
            observation: 融合后的多模态观测
            session_id: 会话标识
            use_local: 是否启用本地模型 (用于第一步情绪摘要)

        Returns:
            JarvisResponse
        """
        if not self.is_initialized:
            raise RuntimeError("JarvisAgent not initialized. Call initialize() first.")

        self.current_session_id = session_id
        timestamp = time.time()

        # 1. 获取历史上下文
        history_turns = self.context_manager.get_recent_history(session_id, n_turns=10)
        history_text = self.context_manager.format_history_for_prompt(session_id, n_turns=10)

        # 2. 获取用户画像 (已关闭 — 短对话测试不需要，L3 跨会话持久化时再启用)
        # TODO: L3 实现后取消注释以下代码块
        # user_profile = self.context_manager.get_user_profile("Speaker_A")
        # if not user_profile:
        #     user_profile = self.agent_config.get("user_profile", {})
        # if user_profile and not (user_profile.get("partner_name") or user_profile.get("relationship")):
        #     user_profile = {}
        user_profile = {}
        past_context = ""  # 跨会话记忆已禁用

        # === 步骤 1: 本地小模型统计情绪特征 ===
        local_emotion_summary = ""
        generated_by = "remote"

        if use_local and self.local_model is not None:
            local_summary_prompt = self._build_local_summary_prompt(observation)
            local_emotion_summary = self._call_local_model(
                "You are an emotion statistics analyzer. Output only bullet points. No intro.",
                local_summary_prompt,
            )
            if local_emotion_summary:
                logger.info("Local emotion summary generated")
                generated_by = "local+remote"
            else:
                logger.warning("Local summary failed, will send raw data to remote")
                local_emotion_summary = "(local model unavailable — raw data below)"

        # === 步骤 2: 远程大模型综合决策 ===
        system_prompt = JARVIS_PERSONA
        # TODO: L3 实现后取消注释以下代码块，启用个性化注入
        # if user_profile:
        #     personalization = self._build_personalization_injection(user_profile)
        #     system_prompt += "\n" + personalization
        #     logger.info("=== PERSONALIZATION INJECTED ===\n%s", personalization)
        # else:
        #     logger.info("=== NO PERSONALIZATION (profile empty) ===")
        # ===== 两阶段推理：阶段1（理解）→ 阶段2（决策）=====
        # 阶段1: SCENE + EMOTIONS + CONFLICT ATTRIBUTION
        stage1_prompt = self._build_analysis_prompt_with_summary(
            observation=observation,
            history_text=history_text,
            local_emotion_summary=local_emotion_summary or "(no local summary)",
            history_turns_count=len(history_turns),
            mode="stage1",
        )
        logger.info(f"=== STAGE1 PROMPT ===\n{stage1_prompt}")
        stage1_output = self._call_remote_model(system_prompt, stage1_prompt)

        if stage1_output is None:
            # 降级：阶段1失败 → 回退单 prompt 模式
            logger.warning("Stage 1 failed, falling back to single-prompt mode")
            fallback_prompt = self._build_analysis_prompt_with_summary(
                observation=observation,
                history_text=history_text,
                local_emotion_summary=local_emotion_summary or "(no local summary)",
                history_turns_count=len(history_turns),
                mode="full",
            )
            logger.info(f"=== FALLBACK FULL PROMPT ===\n{fallback_prompt}")
            response_text = self._call_remote_model(system_prompt, fallback_prompt)
        else:
            # 阶段2: STRATEGY + TASKS + TOOLS + PARTNER
            stage2_prompt = self._build_analysis_prompt_with_summary(
                observation=observation,
                history_text=history_text,
                local_emotion_summary=local_emotion_summary or "(no local summary)",
                history_turns_count=len(history_turns),
                mode="stage2",
                stage1_output=stage1_output,
            )
            logger.info(f"=== STAGE2 PROMPT ===\n{stage2_prompt}")
            stage2_output = self._call_remote_model(system_prompt, stage2_prompt)

            if stage2_output is not None:
                response_text = stage1_output + "\n" + stage2_output
                generated_by = "remote-2stage"
            else:
                logger.warning("Stage 2 failed, keeping stage 1 analysis only")
                response_text = stage1_output
                generated_by = "remote-stage1-only"

        # 远程不可用 → 本地模型兜底 (用精简 prompt)
        if response_text is None and self.local_model is not None:
            logger.warning("Remote failed, falling back to local-only")
            fallback_prompt = self._build_local_prompt(observation, history_text)
            response_text = self._call_local_model(JARVIS_PERSONA_LOCAL, fallback_prompt)
            generated_by = "local_fallback"

        # 全部不可用 → 规则引擎
        if response_text is None:
            response_text = self._generate_rule_based_response(observation)
            generated_by = "rule_based"

        # ===== 应用自适应决策门限 =====
        gate_prefix = ""
        if response_text and generated_by != "rule_based":
            gate_prefix, response_text = self._apply_decision_gate(response_text, observation)

        # 3. 构建响应
        response = JarvisResponse(
            session_id=session_id,
            timestamp=timestamp,
            analysis=response_text,
            user_emotion_label=observation.user_emotion,
            partner_emotion_label=observation.partner_emotion,
            suggestion=self._extract_suggestion(response_text),
            risk_alert=self._extract_risk_alert(response_text, observation),
            generated_by=generated_by,
        )

        # 4. 保存到历史
        from .context_manager import ConversationTurn
        turn = ConversationTurn(
            session_id=session_id,
            timestamp=timestamp,
            window_start=observation.window_start,
            window_end=observation.window_end,
            user_emotion=observation.user_emotion,
            partner_emotion=observation.partner_emotion,
            user_speech=observation.user_speech,
            partner_speech=observation.partner_speech,
            user_speech_top2=getattr(observation, 'user_speech_top2', ''),
            partner_speech_top2=getattr(observation, 'partner_speech_top2', ''),
            jarvis_response=response_text,
            gate_prefix=gate_prefix,
            tension_level=0.0,
            conversation_phase="development",
        )
        self.context_manager.save_turn(turn)

        logger.info(
            f"Jarvis response: session={session_id}, "
            f"by={generated_by}"
        )

        return response

    def generate_session_summary(self, session_id: str) -> SessionSummary:
        """
        对话结束后生成总结

        使用远程大模型进行深度总结 (本地2B模型不足以完成此任务)
        """
        turns = self.context_manager.get_full_conversation(session_id)
        if not turns:
            logger.warning(f"No conversation data for session {session_id}")
            return SessionSummary(
                session_id=session_id,
                summary="No data available.",
                key_insights="",
                partner_profile="",
                user_feedback="",
                next_session_prep="",
            )

        stats = self.context_manager.get_conversation_stats(session_id)
        emotion_timeline = self.context_manager.get_emotion_timeline(session_id)

        # 构建完整对话文本
        full_conversation = self._format_full_conversation(turns)

        # 构建情感时间线文本
        emotion_text = self._format_emotion_timeline(emotion_timeline)

        # 构建总结 Prompt
        summary_prompt = SESSION_SUMMARY_PROMPT.format(
            session_id=session_id,
            duration_minutes=stats.get("duration_seconds", 0) / 60,
            total_turns=stats.get("total_turns", 0),
            conversation_type="conversation",
            full_conversation=full_conversation[:8000],  # 截断以防超 token 限制
            emotion_timeline=emotion_text[:3000],
        )

        # 使用远程大模型生成总结
        system_prompt = (
            "You are JARVIS. Generate a comprehensive conversation session summary. "
            "Be analytical, insightful, and professional. Respond in English."
        )

        summary_text = self._call_remote_model(
            system_prompt, summary_prompt, max_tokens=2048
        )

        if summary_text is None:
            summary_text = self._generate_rule_based_summary(turns, stats)

        # 解析总结中的各个部分
        sections = self._parse_summary_sections(summary_text)

        summary = SessionSummary(
            session_id=session_id,
            summary=sections.get("summary", summary_text[:500]),
            key_insights=sections.get("key_insights", ""),
            partner_profile=sections.get("partner_profile", ""),
            user_feedback=sections.get("user_feedback", ""),
            next_session_prep=sections.get("next_session_prep", ""),
        )

        # 持久化保存
        self.context_manager.save_summary(summary)
        logger.info(f"Session summary generated: {session_id}")

        return summary

    # ================================================================
    # LLM 调用接口
    # ================================================================

    def _call_local_model(
        self, system_prompt: str, user_prompt: str
    ) -> Optional[str]:
        """
        调用本地 Qwen 2B 模型
        """
        try:
            local_type = self.local_config.get("provider", "transformers")

            if local_type == "ollama" or self.local_model == "ollama":
                return self._call_ollama(system_prompt, user_prompt)
            elif local_type == "vllm" or self.local_model == "vllm":
                return self._call_vllm(system_prompt, user_prompt)
            elif self.local_model is not None and self.local_tokenizer is not None:
                return self._call_transformers(system_prompt, user_prompt)
        except Exception as e:
            logger.error(f"Local model inference failed: {e}")
            return None

    def _call_transformers(
        self, system_prompt: str, user_prompt: str
    ) -> Optional[str]:
        """使用 transformers 调用 Qwen 2B"""
        import torch

        # Qwen 的 Chat 模板
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        text = self.local_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.local_tokenizer(text, return_tensors="pt")
        if self.local_config.get("device") == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.local_model.generate(
                **inputs,
                max_new_tokens=self.local_config.get("max_tokens", 512),
                temperature=self.local_config.get("temperature", 0.7),
                do_sample=True,
                pad_token_id=self.local_tokenizer.eos_token_id,
            )

        response = self.local_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return response.strip()

    def _call_ollama(
        self, system_prompt: str, user_prompt: str
    ) -> Optional[str]:
        """
        通过 Ollama API 调用 (使用 /api/chat 端点, 兼容 Qwen/Llama 等chat模型)
        """
        import requests
        import json

        # 短 system prompt + 去重换行避免 token 浪费
        system_short = system_prompt.strip()[:600]
        user_short = user_prompt.strip()[:1200]

        try:
            resp = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": self.local_model_name,
                    "messages": [
                        {"role": "system", "content": system_short},
                        {"role": "user", "content": user_short},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": self.local_config.get("temperature", 0.7),
                        "num_predict": self.local_config.get("max_tokens", 512),
                    },
                },
                timeout=120,
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data.get("message", {}).get("content", "").strip()
                if content:
                    return content
                # 空响应 → 用 thinking 字段兜底
                thinking = data.get("message", {}).get("thinking", "").strip()
                if thinking:
                    # 取 thinking 的最后一段作为回答
                    logger.warning("Ollama returned thinking only, extracting last part")
                    lines = thinking.split("\n")
                    return lines[-1].strip() if lines else thinking[-300:]

            logger.warning(f"Ollama returned empty response (status={resp.status_code})")
            return None
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            return None

    def _call_vllm(
        self, system_prompt: str, user_prompt: str
    ) -> Optional[str]:
        """通过 vLLM OpenAI 兼容 API 调用"""
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url=self.local_api_base,
                api_key="not-needed",
            )
            resp = client.chat.completions.create(
                model=self.local_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.local_config.get("max_tokens", 512),
                temperature=self.local_config.get("temperature", 0.7),
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"vLLM call failed: {e}")
            return None

    def _call_remote_model(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        """
        调用远程大模型 API (OpenAI 兼容格式)
        """
        if self.remote_client is None:
            logger.warning("Remote client not available")
            return None

        if max_tokens is None:
            max_tokens = self.remote_config.get("max_tokens", 1024)

        try:
            resp = self.remote_client.chat.completions.create(
                model=self.remote_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=self.remote_config.get("temperature", 0.7),
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Remote model call failed: {e}")
            return None

    # ================================================================
    # 自适应决策门限
    # ================================================================

    def _conflict_level(self, jsd, face_valid, voice_unclear):
        """返回 (level, reasons) 或 None。level ∈ {'candidate', 'moderate', 'high_unreliable'}"""
        if jsd <= 0.2:
            return None

        reasons = []
        if face_valid < 0.5:
            reasons.append('Face unreliable')
        if voice_unclear >= 0.5:
            reasons.append('Voice unclear')
        reason_str = ' + '.join(reasons) if reasons else None

        if jsd <= 0.35:
            return ('moderate', reason_str)
        else:
            if face_valid >= 0.5 and voice_unclear < 0.5:
                return ('candidate', None)
            else:
                return ('high_unreliable', reason_str)

    def _detect_emotion_shifts(self, voice_labels, face_labels):
        """
        跨句连续情绪突变检测 — 区分真实趋势转折与单句噪声。

        算法:
          1. 将窗口分为前半段和后半段
          2. 比较两半段的主导情绪（众数）
          3. 若主导情绪变化，且新情绪在后半段连续出现 ≥ min_run 次 → 判定为真实突变
          4. 单句噪声（仅出现1-2次即恢复）不会被误判

        返回: list[dict], 每个 dict 包含 modality / from / to / from_count / to_count / shift_at
              无突变时返回空列表
        """
        shifts = []
        min_run = 3  # 新情绪至少连续出现 3 次才判定为真实突变

        for modality_name, labels in [("voice", voice_labels), ("face", face_labels)]:
            n = len(labels)
            if n < 6:
                continue  # 窗口太小，不足以检测趋势

            mid = n // 2
            first_half = labels[:mid]
            second_half = labels[mid:]

            from collections import Counter
            first_dom = Counter(first_half).most_common(1)
            second_dom = Counter(second_half).most_common(1)
            if not first_dom or not second_dom:
                continue
            from_emo, from_cnt = first_dom[0]
            to_emo, to_cnt = second_dom[0]

            if from_emo == to_emo:
                continue  # 无变化

            # 检查新情绪在后半段是否有连续 min_run 次出现
            max_run = 0
            cur_run = 0
            shift_idx = None
            for i, label in enumerate(second_half):
                if label == to_emo:
                    if cur_run == 0:
                        run_start = mid + i
                    cur_run += 1
                    max_run = max(max_run, cur_run)
                else:
                    if cur_run >= min_run and shift_idx is None:
                        shift_idx = run_start
                    cur_run = 0
            if cur_run >= min_run and shift_idx is None:
                shift_idx = run_start

            if max_run < min_run:
                continue  # 新情绪不够持久 → 噪声

            shifts.append({
                'modality': modality_name,
                'from_emotion': from_emo,
                'to_emotion': to_emo,
                'from_count': from_cnt,
                'to_count': to_cnt,
                'longest_run': max_run,
                'shift_utterance': shift_idx + 1 if shift_idx is not None else mid + 1,
            })

        return shifts

    def _build_emotion_shift_section(self, user_shifts, partner_shifts):
        """将情绪突变检测结果格式化为 Prompt 段落。无突变时返回空字符串。"""
        all_shifts = [(f"User ({mod})", s) for s in user_shifts
                      for mod in [s['modality']]] + \
                     [(f"Partner ({mod})", s) for s in partner_shifts
                      for mod in [s['modality']]]

        if not all_shifts:
            return ""

        lines = [
            "**EMOTION SHIFT DETECTED** (pre-computed, cross-utterance trend analysis):",
            "The following sustained emotion shifts were detected in this window.",
            "Use this information in your EMOTIONS section — identify the likely trigger for each shift.",
            "",
        ]
        for who, s in all_shifts:
            lines.append(
                f"  - {who}: {s['from_emotion']} → {s['to_emotion']} "
                f"(near utterance #{s['shift_utterance']}, "
                f"new emotion sustained for {s['longest_run']} consecutive utterances)"
            )
        return "\n".join(lines)

    def _apply_decision_gate(self, response_text: str, observation: MultimodalObservation):
        """
        Safety net: only warn when LLM completely misses the CONFLICT ATTRIBUTION section.
        No longer prefixes every response — attribution is handled by the prompt itself.
        """
        import re

        def _extract_unclear_from_top2(voice_top2_str):
            match = re.search(r'unclear\(([\d.]+)\)', voice_top2_str)
            return float(match.group(1)) if match else 0.0

        # Count candidate-level conflicts
        candidate_count = 0
        for conflict in observation.top_conflicts_user + observation.top_conflicts_partner:
            jsd, text, voice_top2, face_top2, face_valid, unclear_prob = conflict
            voice_unclear = _extract_unclear_from_top2(voice_top2)
            result = self._conflict_level(jsd, face_valid, voice_unclear)
            if result and result[0] == 'candidate':
                candidate_count += 1

        if candidate_count == 0:
            return ("", response_text)

        # LLM produced the section — trust its analysis (case-insensitive)
        if re.search(r"conflict attribution", response_text, re.IGNORECASE):
            return ("", response_text)

        # LLM missed it entirely — add safety prefix
        prefix = f"[Warning: {candidate_count} conflict(s) not attributed] "
        return (prefix, prefix + response_text)

    # ================================================================
    # Prompt 构建
    # ================================================================

    def _compute_window_jsd(self, voice_probs_list, face_probs_list, utterance_texts, face_valid_ratios_list=None):
        """
        计算窗口内语音和面部概率分布的 Jensen-Shannon 散度（对称），并提取关键样本。

        参数:
            voice_probs_list: 每句话的语音概率字典列表 (9类)
            face_probs_list: 每句话的面部概率列表 (7维)
            utterance_texts: 每句话的原始文本列表
            face_valid_ratios_list: 每句话的面部有效帧占比 (与 face_probs_list 一一对应)

        返回:
            {
                'avg_jsd': float,
                'max_jsd': float,
                'num_valid': int,
                'top_conflicts': [(jsd, text, voice_top2_str, face_top2_str, valid_ratio, unclear_prob), ...],
                'emotion_counts': {'voice': {}, 'face': {}}
            }
        """
        if not voice_probs_list or not face_probs_list:
            return {'avg_jsd': 0.0, 'max_jsd': 0.0, 'num_valid': 0,
                    'top_conflicts': [], 'low_quality': [], 'emotion_counts': {}}

        # ---- 1. 维度映射（语音 9→7） ----
        VOICE_TO_FACE = {
            'frustrated': 'angry', 'neutral': 'neutral', 'angry': 'angry',
            'sad': 'sad', 'excited': 'surprised', 'happy': 'happy',
            'disgust': 'disgust', 'fear': 'fear', 'unclear': None
        }
        FACE_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']

        # ---- 2. 逐句计算 JSD + 收集情绪标签 + 语音不确定性 ----
        voice_labels_all = []
        face_labels_all = []
        jsd_entries = []

        min_len = min(len(voice_probs_list), len(face_probs_list), len(utterance_texts))
        eps = 1e-10

        for i in range(min_len):
            vp = voice_probs_list[i]
            fp = face_probs_list[i]
            text = utterance_texts[i] if i < len(utterance_texts) else ""

            # ---- 单句语音不确定性 ----
            unclear_prob = vp.get('unclear', 0.0)

            # ---- 映射语音到7维 ----
            voice_7d = []
            for label in FACE_LABELS:
                total = sum(v_prob for v_label, v_prob in vp.items() if VOICE_TO_FACE.get(v_label) == label)
                voice_7d.append(total)
            sum_v = sum(voice_7d)
            if sum_v == 0:
                continue
            voice_7d = [v / sum_v for v in voice_7d]

            # ---- 归一化面部概率 ----
            sum_f = sum(fp)
            if sum_f == 0:
                continue
            fp_norm = [v / sum_f for v in fp]
            
            valid_ratio = 1.0
            if face_valid_ratios_list and i < len(face_valid_ratios_list):
                valid_ratio = face_valid_ratios_list[i]

            # ---- 计算 JSD (Jensen-Shannon Divergence, 对称) ----
            # JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), M = (P+Q)/2
            m = [(p + q) / 2 for p, q in zip(voice_7d, fp_norm)]
            jsd = 0.0
            for p, q, mid in zip(voice_7d, fp_norm, m):
                p = max(p, eps)
                q = max(q, eps)
                mid = max(mid, eps)
                jsd += 0.5 * p * np.log(p / mid) + 0.5 * q * np.log(q / mid)

            # ---- 提取 Top-2 标签（9类→8类映射，与 _format_utterance_with_top2 一致） ----
            v_mapped = {}
            v_unclear = 0.0
            for label, prob in vp.items():
                mapped_label = VOICE_TO_FACE.get(label)
                if mapped_label:
                    v_mapped[mapped_label] = v_mapped.get(mapped_label, 0) + prob
                else:
                    v_unclear += prob
            if v_unclear > 0:
                v_mapped['unclear'] = v_unclear
            sorted_v = sorted(v_mapped.items(), key=lambda x: x[1], reverse=True)
            top_v1, top_v2 = sorted_v[0], (sorted_v[1] if len(sorted_v) > 1 else sorted_v[0])
            voice_top2_str = f"{top_v1[0]}({top_v1[1]:.2f})/{top_v2[0]}({top_v2[1]:.2f})"

            sorted_f = sorted([(FACE_LABELS[idx], fp[idx]) for idx in range(len(FACE_LABELS))], key=lambda x: x[1],
                              reverse=True)
            top_f1, top_f2 = sorted_f[0], (sorted_f[1] if len(sorted_f) > 1 else sorted_f[0])
            face_top2_str = f"{top_f1[0]}({top_f1[1]:.2f})/{top_f2[0]}({top_f2[1]:.2f})"

            jsd_entries.append({
                'jsd': jsd,
                'text': text,
                'voice_top2': voice_top2_str,
                'face_top2': face_top2_str,
                'voice_label': top_v1[0],
                'face_label': top_f1[0],
                'valid_ratio': valid_ratio,
                'unclear_prob': unclear_prob,
            })

            voice_labels_all.append(top_v1[0])
            face_labels_all.append(top_f1[0])

        # ---- 3. 排序取极端样本 ----
        if not jsd_entries:
            return {'avg_jsd': 0.0, 'max_jsd': 0.0, 'num_valid': 0,
                    'top_conflicts': [], 'low_quality': [], 'emotion_counts': {}}

        sorted_entries = sorted(jsd_entries, key=lambda x: x['jsd'], reverse=True)

        # 冲突最大的2句 (JSD 最大 = 模态不一致)
        top_conflicts = [(e['jsd'], e['text'], e['voice_top2'], e['face_top2'],
                          e['valid_ratio'], e['unclear_prob']) for e in sorted_entries if e['jsd'] > 0.2]

        # JSD ≤ 0.2 但模态不可靠的句子（无冲突，仅标记质量）
        low_quality = []
        for e in sorted_entries:
            if e['jsd'] <= 0.2:
                reasons = []
                if e['unclear_prob'] > 0.5:
                    reasons.append('Voice unclear')
                if e['valid_ratio'] < 0.5:
                    reasons.append('Face unreliable')
                if reasons:
                    low_quality.append((e['jsd'], e['text'], e['voice_top2'], e['face_top2'],
                                       e['valid_ratio'], e['unclear_prob'], ' + '.join(reasons)))

        # ---- 4. 情绪计数 ----
        from collections import Counter
        emotion_counts = {
            'voice': dict(Counter(voice_labels_all)),
            'face': dict(Counter(face_labels_all))
        }

        jsd_values = [e['jsd'] for e in jsd_entries]

        return {
            'avg_jsd': float(np.mean(jsd_values)),
            'max_jsd': float(np.max(jsd_values)),
            'num_valid': len(jsd_entries),
            'top_conflicts': top_conflicts,
            'low_quality': low_quality,
            'emotion_counts': emotion_counts,
            'voice_labels': voice_labels_all,
            'face_labels': face_labels_all,
        }

    def _format_utterance_with_top2(self, text: str, voice_probs: dict, face_probs: list) -> str:
        """
        将一句话格式化为带 Top‑2 概率的标注。
        voice_probs: 9类概率字典 → 自动映射到8类（7情绪 + unclear）
        face_probs: 7类概率列表
        """
        VOICE_9TO8 = {
            'frustrated': 'angry', 'neutral': 'neutral', 'angry': 'angry',
            'sad': 'sad', 'excited': 'surprised', 'happy': 'happy',
            'disgust': 'disgust', 'fear': 'fear',
        }
        # ---- 语音 Top‑2 (9类→8类：7情绪映射合并 + unclear保留) ----
        if voice_probs:
            mapped = {}
            unclear_prob = 0.0
            for label, prob in voice_probs.items():
                mapped_label = VOICE_9TO8.get(label)
                if mapped_label:
                    mapped[mapped_label] = mapped.get(mapped_label, 0) + prob
                else:
                    unclear_prob += prob
            if unclear_prob > 0:
                mapped['unclear'] = unclear_prob
            sorted_v = sorted(mapped.items(), key=lambda x: x[1], reverse=True)
            top1_v, top2_v = sorted_v[0], (sorted_v[1] if len(sorted_v) > 1 else sorted_v[0])
            voice_str = f"{top1_v[0]}({top1_v[1]:.2f})/{top2_v[0]}({top2_v[1]:.2f})"
        else:
            voice_str = "?"

        # ---- 面部 Top‑2 ----
        if face_probs and sum(face_probs) > 0:
            labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
            sorted_idx = np.argsort(face_probs)[::-1]
            top1_idx, top2_idx = sorted_idx[0], (sorted_idx[1] if len(sorted_idx) > 1 else sorted_idx[0])
            top1_f = (labels[top1_idx], face_probs[top1_idx])
            top2_f = (labels[top2_idx], face_probs[top2_idx])
            face_str = f"{top1_f[0]}({top1_f[1]:.2f})/{top2_f[0]}({top2_f[1]:.2f})"
        else:
            face_str = "?"

        return f"[voice: {voice_str} | face: {face_str}] {text}"
    def _build_analysis_prompt(
        self,
        observation: MultimodalObservation,
        history_text: str,
        past_context: str,
        history_turns_count: int,
    ) -> str:
        """构建实时分析的 Prompt"""
        obs = observation.to_dict()

        # 用户/对方 的情感标签转可读文本
        user_emotion_label = EMOTION_LABELS.get(
            obs["user_emotion"], obs["user_emotion"]
        )
        partner_emotion_label = EMOTION_LABELS.get(
            obs["partner_emotion"], obs["partner_emotion"]
        )

        prompt = REALTIME_ANALYSIS_PROMPT.format(
            window_size=self.agent_config.get("response_interval", 20),
            user_emotion=user_emotion_label,
            user_emotion_conf=obs["user_emotion_conf"],
            user_emotion_trend=obs["user_emotion_trend"],
            user_speech=obs["user_speech"] or "(no speech detected)",
            partner_emotion=partner_emotion_label,
            partner_emotion_conf=obs["partner_emotion_conf"],
            partner_emotion_trend=obs["partner_emotion_trend"],
            partner_speech=obs["partner_speech"] or "(no speech detected)",
            scene_context=obs["scene_context"],
            history_turns=history_turns_count,
            conversation_history=history_text,
        )

        if past_context:
            prompt += f"\n\n### Context from Previous Sessions:\n{past_context[:500]}"

        return prompt

    def _build_local_prompt(
        self,
        observation: MultimodalObservation,
        history_text: str,
    ) -> str:
        """构建本地小模型专用精简 Prompt"""
        obs = observation.to_dict()
        user_emotion_label = EMOTION_LABELS.get(
            obs["user_emotion"], obs["user_emotion"]
        )
        partner_emotion_label = EMOTION_LABELS.get(
            obs["partner_emotion"], obs["partner_emotion"]
        )
        return REALTIME_ANALYSIS_PROMPT_LOCAL.format(
            user_emotion=user_emotion_label,
            user_speech=obs["user_speech"] or "(silence)",
            partner_emotion=partner_emotion_label,
            partner_speech=obs["partner_speech"] or "(silence)",
            conversation_history=history_text[:300] if history_text else "(new conversation)",
        )

    def _build_local_summary_prompt(self, observation: MultimodalObservation) -> str:
        """构建本地小模型的情绪统计 Prompt"""
        obs = observation.to_dict()
        # 统计窗口内大致句子数
        user_utts = obs["user_speech"].count("|") + 1 if "|" in (obs["user_speech"] or "") else 1
        partner_utts = obs["partner_speech"].count("|") + 1 if "|" in (obs["partner_speech"] or "") else 1
        return LOCAL_EMOTION_SUMMARY_PROMPT.format(
            n_utterances=user_utts + partner_utts,
            user_speech=obs["user_speech"] or "(silence)",
            partner_speech=obs["partner_speech"] or "(silence)",
        )


    def _build_conflict_attribution_section(self, observation: MultimodalObservation, total_utterances: int = 0) -> str:
        """Build the CONFLICT ATTRIBUTION section with three tiers + window stats.

        Output layers:
          High Conflict       — JSD > 0.35 + both modalities reliable  → 5-type LLM attribution
          High Conflict/Noisy — JSD > 0.35 + one modality noisy        → text vs reliable modality
          Mild Divergence     — 0.2 < JSD ≤ 0.35 + both reliable       → 1-line count
          Signal Quality Note — modality flagged (any JSD range)         → quality warning
          Unusable            — both modalities unreliable               → skipped
        Plus window utterance count for zero-drop guarantee.
        """
        candidates = []   # (speaker, text, v_t2, f_t2, jsd_str)
        partial = []      # (speaker, text, v_t2, f_t2, jsd_str, reasons, use_instruction)
        skipped = []      # (speaker, text, reasons) — both modalities unreliable
        low_conflicts = []  # (speaker, text, v_t2, f_t2, reasons) — JSD≤0.2 quality or 0.2-0.35 noise
        mild_count = 0    # 0.2<JSD≤0.35 + both reliable → counted, not listed

        all_conflicts = (
            [("User", c) for c in (observation.top_conflicts_user or [])] +
            [("Partner", c) for c in (observation.top_conflicts_partner or [])]
        )

        for speaker, conflict in all_conflicts:
            jsd, text, voice_top2, face_top2, face_valid, unclear_prob = conflict
            voice_unclear = unclear_prob
            result = self._conflict_level(jsd, face_valid, unclear_prob)
            if result is None:
                continue
            level, reasons = result
            if level == "candidate":
                candidates.append((speaker, text[:60], voice_top2, face_top2, f"{jsd:.3f}", f"{face_valid:.2f}"))
            elif level == "moderate":
                # Triage: #5→count, #6-7→Signal Quality, #8→Unusable
                if face_valid >= 0.5 and voice_unclear < 0.5:
                    mild_count += 1
                elif face_valid < 0.5 and voice_unclear >= 0.5:
                    skipped.append((speaker, text[:60], reasons if reasons else "both unreliable"))
                else:
                    low_conflicts.append((speaker, text[:60], voice_top2, face_top2,
                                         f"JSD={jsd:.3f} + {'Voice unclear' if voice_unclear >= 0.5 else 'Face unreliable'}"))
            elif level == "high_unreliable":
                # Determine which modality is still usable
                if face_valid >= 0.5:
                    use = "face+text only (voice noisy, JSD may be noise)"
                elif voice_unclear < 0.5:
                    use = "voice+text only (face unreliable, JSD may be noise)"
                else:
                    skipped.append((speaker, text[:60], reasons if reasons else "both unreliable"))
                    continue
                partial.append((speaker, text[:60], voice_top2, face_top2, f"{jsd:.3f}", reasons, use))

        # ================================================================
        # Append JSD≤0.2 entries with unreliable modalities (no conflict, quality note only)
        # ================================================================
        for speaker, entries in [("User", observation.low_conflicts_user or []),
                                  ("Partner", observation.low_conflicts_partner or [])]:
            for entry in entries:
                jsd, text, v_t2, f_t2, fv, up, reasons = entry
                low_conflicts.append((speaker, text[:60], v_t2, f_t2, f"JSD={jsd:.3f} + {reasons}"))

        # ================================================================
        # Window coverage stats (replaces FULL-SCAN — analysis already done by Python)
        # ================================================================
        window_stats = f"\nWindow: {total_utterances} utterances — all covered in analysis above."

        # ================================================================
        # Window-level alignment summary (used by both branches)
        # ================================================================
        avg_u = getattr(observation, 'jsd_user', 0) or 0
        avg_p = getattr(observation, 'jsd_partner', 0) or 0

        # ================================================================
        # Case: no candidates
        # ================================================================
        if not candidates:
            lines = [
                "**CONFLICT ATTRIBUTION**:",
                f"Window voice-face alignment: User avg JSD={avg_u:.3f} | Partner avg JSD={avg_p:.3f}",
                "No High Conflict items in this window (no JSD>0.35 with both modalities reliable).",
            ]

            if mild_count > 0:
                lines.append(f"Mild divergence: {mild_count} utterance(s) (0.2<JSD≤0.35, both reliable), see window stats above.")

            if partial:
                lines.append("")
                lines.append("High Conflict \u2014 Noisy Signal (JSD>0.35 but one modality unreliable, use reliable modality only):")
                for speaker, text, v_t2, f_t2, jsd, reasons, use in partial:
                    tag = f" ({reasons})" if reasons else ""
                    lines.append(f'  - [{speaker}{tag}] "{text}..." \u2014 {use}')
                    lines.append(f'    Voice: {v_t2} | Face: {f_t2} | JSD={jsd}')

            if skipped:
                lines.append("")
                lines.append("Unusable (both modalities unreliable, no comparison possible):")
                for speaker, text, reasons in skipped:
                    lines.append(f'  - [{speaker} ({reasons})] "{text}..."')

            if low_conflicts:
                lines.append("")
                lines.append("Signal Quality Note (modality quality flagged, voice-face comparison may be unreliable):")
                for speaker, text, v_t2, f_t2, reasons in low_conflicts:
                    lines.append(f'  - [{speaker} ({reasons})] "{text}..." \u2014 Voice: {v_t2} | Face: {f_t2}')

            lines.append(window_stats)
            return "\n".join(lines)

        # ================================================================
        # Case: candidates exist — High Conflict + Noisy Signal + FULL-SCAN
        # ================================================================
        lines = [
            "**CONFLICT ATTRIBUTION**:",
            f"Window voice-face alignment: User avg JSD={avg_u:.3f} | Partner avg JSD={avg_p:.3f}",
            f"The system flagged {len(candidates)} utterance(s) with voice-face mismatch (both modalities reliable).",
            f"You MUST address EXACTLY the {len(candidates)} items listed in High Conflict below — no more, no less. Do NOT reference utterances from Recent History. Address each INDIVIDUALLY with its number. Do NOT summarize or group them. Pick ONE explanation per item.",
            "",
            "High Conflict (both signals reliable):",
        ]

        for i, (speaker, text, v_t2, f_t2, jsd, fv) in enumerate(candidates, 1):
            lines.append(
                f'  {i}. [{speaker}] "{text}..."'
                f' \u2014 Voice: {v_t2} | Face: {f_t2} | JSD={jsd} | FV={fv}'
            )

        lines.append("")
        lines.append("For each numbered item, respond with ONE of:")
        lines.append('  1. Genuine masking \u2192 "I notice your voice and expression differ in \'...\'."')
        lines.append('  2. Social display \u2192 "Note: \'...\' may reflect social convention."')
        lines.append('  3. Sarcasm/irony \u2192 "Note: \'...\' appears sarcastic."')
        lines.append('  4. Suppressed emotion \u2192 "I sense you may be holding back some emotion."')
        lines.append("  5. Cannot determine \u2192 ask for clarification.")
        lines.append("")
        lines.append("ATTRIBUTION GUIDANCE \u2014 read the utterance TEXT first, then check voice-face data:")
        lines.append('  \u2022 If the utterance contains expletives, aggression, confrontation, or rhetorical questions')
        lines.append('    \u2192 the emotion is being EXPRESSED. Consider Genuine masking or Social display, NOT Suppressed emotion.')
        lines.append('  \u2022 If the utterance is restrained, polite, or indirect while voice-face shows strong emotion')
        lines.append('    \u2192 the emotion is being HELD BACK. Suppressed emotion is the correct choice.')
        lines.append('  \u2022 If the utterance is sarcastic in tone ("Oh, that\'s just great") OR the wording')
        lines.append('    contradicts the likely true feeling \u2192 Sarcasm/irony.')
        lines.append("")
        lines.append("CRITICAL: For [Partner] items, user CANNOT answer for partner.")
        lines.append('  Use ONLY: "Note: Partner\'s statement \'...\' shows voice-face conflict."')
        lines.append("")
        lines.append('For each [User] item, pick EXACTLY ONE type (1-5). Write the type name explicitly (e.g., "Suppressed emotion → I sense..."). Do NOT use generic descriptions like "shows voice-face conflict." Output in this exact format per item: "N. [TYPE] → "specific response text"" (include the item number, type name, and quoted utterance text).')

        if mild_count > 0:
            lines.append("")
            lines.append(f"Mild divergence: {mild_count} utterance(s) (0.2<JSD≤0.35, both reliable), see window stats above.")

        if partial:
            lines.append("")
            lines.append("High Conflict \u2014 Noisy Signal (JSD>0.35 but one modality unreliable, use reliable modality only):")
            for speaker, text, v_t2, f_t2, jsd, reasons, use in partial:
                tag = f" ({reasons})" if reasons else ""
                lines.append(f'  - [{speaker}{tag}] "{text}..." \u2014 {use}')
                lines.append(f'    Voice: {v_t2} | Face: {f_t2} | JSD={jsd}')

        if skipped:
            lines.append("")
            lines.append("Unusable (both modalities unreliable, no comparison possible):")
            for speaker, text, reasons in skipped:
                lines.append(f'  - [{speaker} ({reasons})] "{text}..."')

        if low_conflicts:
            lines.append("")
            lines.append("Signal Quality Note (modality quality flagged, voice-face comparison may be unreliable):")
            for speaker, text, v_t2, f_t2, reasons in low_conflicts:
                lines.append(f'  - [{speaker} ({reasons})] "{text}..." \u2014 Voice: {v_t2} | Face: {f_t2}')

        lines.append(window_stats)
        return "\n".join(lines)
    def _build_analysis_prompt_with_summary(
            self,
            observation: MultimodalObservation,
            history_text: str,
            local_emotion_summary: str,
            history_turns_count: int,
            mode: str = "full",
            stage1_output: str = None,
    ) -> str:
        obs = observation.to_dict()

        # 拆分出原始文本列表（用于 JSD 计算）
        user_texts = obs["user_speech"].split(" | ") if obs["user_speech"] != "(silence)" else []
        partner_texts = obs["partner_speech"].split(" | ") if obs["partner_speech"] != "(silence)" else []
        user_valid_ratios = getattr(observation, 'user_face_valid_ratios', [])
        partner_valid_ratios = getattr(observation, 'partner_face_valid_ratios', [])

        # ---- 构建带 Top‑2 的对话文本（每句一行，编号，便于 LLM 解析） ----
        def build_annotated_text(voice_list, face_list, speech_str, texts_list):
            if not speech_str or speech_str == "(silence)":
                return "(silence)"
            num = min(len(texts_list), len(voice_list), len(face_list))
            annotated = []
            for idx in range(num):
                text = texts_list[idx] if idx < len(texts_list) else ""
                voice = voice_list[idx] if idx < len(voice_list) else {}
                face = face_list[idx] if idx < len(face_list) else [0.0] * 7
                annotated.append(
                    f"  [{idx + 1}] {self._format_utterance_with_top2(text, voice, face)}"
                )
            return "\n".join(annotated)
        logger.info(f"DEBUG user_face_probs_list: {observation.user_face_probs_list[:2] if observation.user_face_probs_list else 'EMPTY'}")
        logger.info(f"DEBUG user_voice_probs_list: {observation.user_voice_probs_list[:2] if observation.user_voice_probs_list else 'EMPTY'}")
        
        user_speech_annotated = build_annotated_text(
            observation.user_voice_probs_list,
            observation.user_face_probs_list,
            obs["user_speech"],
            user_texts
        )
        partner_speech_annotated = build_annotated_text(
            observation.partner_voice_probs_list,
            observation.partner_face_probs_list,
            obs["partner_speech"],
            partner_texts
        )

        # ---- 计算 JSD 摘要（传入原始文本列表） ----
        jsd_user = self._compute_window_jsd(
            observation.user_voice_probs_list,
            observation.user_face_probs_list,
            user_texts,
            user_valid_ratios
        )
        jsd_partner = self._compute_window_jsd(
            observation.partner_voice_probs_list,
            observation.partner_face_probs_list,
            partner_texts,
            partner_valid_ratios
        )

        # ---- 格式化冲突摘要（使用 _conflict_level，与归因段一致） ----
        def format_jsd_summary(jsd_data, speaker_name):
            if not jsd_data.get('top_conflicts'):
                return ""  # 无冲突句子，不显示
            lines = [f"**{speaker_name}**:"]
            for jsd, txt, v_t2, f_t2, valid_ratio, unclear_prob in jsd_data['top_conflicts']:
                result = self._conflict_level(jsd, valid_ratio, unclear_prob)
                if result is None:
                    continue  # JSD <= 0.2, skip
                level, reasons = result
                if level == 'candidate':
                    tag = 'High conflict'
                elif level == 'high_unreliable':
                    tag = f'High conflict ({reasons})'
                else:  # moderate
                    tag = f'Moderate conflict{(" (" + reasons + ")") if reasons else ""}'
                lines.append(
                    f"  - [{tag}] | JSD={jsd:.3f} | Voice: {v_t2} | Face: {f_t2}"
                    f" | FaceValid={valid_ratio:.2f} | Text: \"{txt[:60]}...\""
                )
            return "\n".join(lines)

        user_conflicts = format_jsd_summary(jsd_user, 'User')
        partner_conflicts = format_jsd_summary(jsd_partner, 'Partner')
        conflict_parts = [p for p in [user_conflicts, partner_conflicts] if p]
        conflict_summary = (
            "### Conflict & Reliability Analysis:\n" + "\n\n".join(conflict_parts)
        ) if conflict_parts else ""

        # ---- 构建 Prompt ----
        window_size = self.utt_per_window

        # Store ALL computed fields to observation BEFORE building attribution section
        observation.top_conflicts_user = jsd_user.get('top_conflicts', [])
        observation.top_conflicts_partner = jsd_partner.get('top_conflicts', [])
        observation.low_conflicts_user = jsd_user.get('low_quality', [])
        observation.low_conflicts_partner = jsd_partner.get('low_quality', [])
        observation.jsd_user = jsd_user.get('avg_jsd', 0.0)
        observation.jsd_partner = jsd_partner.get('avg_jsd', 0.0)

        total_utterances = len(user_texts) + len(partner_texts)

        # Build dynamic CONFLICT ATTRIBUTION section (now sees correct jsd_user/jsd_partner)
        conflict_attribution_section = self._build_conflict_attribution_section(observation, total_utterances)

        # ---- 情绪突变检测（跨句时间维度，与 JSD 空间维度正交） ----
        user_shifts = self._detect_emotion_shifts(
            jsd_user.get('voice_labels', []),
            jsd_user.get('face_labels', [])
        )
        partner_shifts = self._detect_emotion_shifts(
            jsd_partner.get('voice_labels', []),
            jsd_partner.get('face_labels', [])
        )
        emotion_shift_section = self._build_emotion_shift_section(user_shifts, partner_shifts)

        if mode == "stage1":
            prompt = STAGE1_ANALYSIS_PROMPT.format(
                window_size=window_size,
                user_speech_annotated=user_speech_annotated,
                partner_speech_annotated=partner_speech_annotated,
                conversation_history=history_text,
                conflict_attribution_section=conflict_attribution_section,
                emotion_shift_section=emotion_shift_section,
            )
        elif mode == "stage2":
            # Build brief numbered conversation text for stage 2 reference
            def _format_brief(texts):
                if not texts:
                    return "(silence)"
                return "\n".join(f"  [{i}] {t}" for i, t in enumerate(texts, 1))
            user_texts_brief = _format_brief(user_texts)
            partner_texts_brief = _format_brief(partner_texts)

            prompt = STAGE2_DECISION_PROMPT.format(
                stage1_analysis=stage1_output or "",
                user_texts_brief=user_texts_brief,
                partner_texts_brief=partner_texts_brief,
            )
        else:
            prompt = REALTIME_ANALYSIS_PROMPT_WITH_SUMMARY.format(
                window_size=window_size,
                user_speech_annotated=user_speech_annotated,
                partner_speech_annotated=partner_speech_annotated,
                conversation_history=history_text,
                conflict_attribution_section=conflict_attribution_section,
                emotion_shift_section=emotion_shift_section,
            )

        # 追加冲突摘要（仅当有冲突时）
        # if conflict_summary: (removed — redundant with Flagged above)
            # Conflict Summary removed from prompt (redundant with Flagged/Partial/Context/Skipped)

        # 日志：完整冲突摘要
        logger.info(f"=== CONFLICT SUMMARY ===\n{conflict_summary if conflict_summary else '(no conflicts)'}")

        # top_conflicts and jsd already stored above (before template format)

        # 计算面部不确定性：从 observation 的 face_valid_ratios 列表取平均有效占比
        user_fvr = getattr(observation, 'user_face_valid_ratios', [])
        if user_fvr:
            observation.face_uncertainty_user = 1.0 - sum(user_fvr) / len(user_fvr)
        else:
            observation.face_uncertainty_user = 1.0

        partner_fvr = getattr(observation, 'partner_face_valid_ratios', [])
        if partner_fvr:
            observation.face_uncertainty_partner = 1.0 - sum(partner_fvr) / len(partner_fvr)
        else:
            observation.face_uncertainty_partner = 1.0

        # ===== Set turn-level dominant emotions from window data (replaces hardcoded "neutral") =====
        def _dominant_face(emotion_counts):
            face_counts = emotion_counts.get('face', {})
            if face_counts:
                return max(face_counts, key=face_counts.get)
            return "neutral"

        def _dominant_voice(emotion_counts):
            voice_counts = emotion_counts.get('voice', {})
            if not voice_counts:
                return "neutral"
            # Skip 'unclear' if any other label exists
            non_unclear = {k: v for k, v in voice_counts.items() if k != 'unclear'}
            if non_unclear:
                return max(non_unclear, key=non_unclear.get)
            return max(voice_counts, key=voice_counts.get)

        observation.user_emotion = _dominant_face(jsd_user.get('emotion_counts', {}))
        observation.partner_emotion = _dominant_face(jsd_partner.get('emotion_counts', {}))

        return prompt

    def _build_personalization_injection(self, user_profile: Dict) -> str:
        """构建个性化注入"""
        return PERSONALIZATION_INJECTION.format(
            preferred_tone=user_profile.get("preferred_tone", "professional"),
            communication_style=user_profile.get("communication_style", "direct"),
            strengths=user_profile.get("strengths", "not yet assessed"),
            weaknesses=user_profile.get("weaknesses", "not yet assessed"),
            past_takeaways=user_profile.get("past_takeaways", "none"),
            partner_name=user_profile.get("partner_name", "Speaker B"),
            relationship=user_profile.get("relationship", "conversation partner"),
            partner_traits=user_profile.get("partner_traits", "not yet profiled"),
            past_patterns=user_profile.get("past_patterns", "not yet observed"),
            effective_strategies=user_profile.get("effective_strategies", "not yet determined"),
            sensitive_topics=user_profile.get("sensitive_topics", "not yet identified"),
        )

    def _format_full_conversation(self, turns) -> str:
        """格式化完整对话用于总结"""
        lines = []
        for i, turn in enumerate(turns):
            emoji_user = {"happy": "😊", "sad": "😢", "angry": "😠", "neutral": "😐",
                         "frustrated": "😤", "surprised": "😲", "fear": "😨",
                         "disgust": "🤢"}.get(turn.user_emotion, "")
            emoji_partner = {"happy": "😊", "sad": "😢", "angry": "😠", "neutral": "😐",
                            "frustrated": "😤", "surprised": "😲", "fear": "😨",
                            "disgust": "🤢"}.get(turn.partner_emotion, "")

            # Use Top‑2 annotated speech when available (has per-utterance [voice: ... | face: ...] labels)
            user_text = (getattr(turn, 'user_speech_top2', '') or turn.user_speech)
            partner_text = (getattr(turn, 'partner_speech_top2', '') or turn.partner_speech)

            lines.append(
                f"[Turn {i+1}]\n"
                f"  User {emoji_user}({turn.user_emotion}): \"{user_text}\"\n"
                f"  Partner {emoji_partner}({turn.partner_emotion}): \"{partner_text}\"\n"
                f"  Jarvis: \"{turn.jarvis_response[:150]}...\"\n"
            )
        return "\n".join(lines)

    def _format_emotion_timeline(self, timeline) -> str:
        """格式化情感时间线"""
        lines = ["Time(s) | User Emotion | Partner Emotion | Phase"]
        lines.append("-" * 55)
        for entry in timeline:
            lines.append(
                f"{entry['time']:.0f}s | {entry['user_emotion']:12s} | "
                f"{entry['partner_emotion']:15s} | {entry['phase']}"
            )
        return "\n".join(lines)

    # ================================================================
    # 响应后处理
    # ================================================================

    def _extract_suggestion(self, response: str) -> str:
        """从 Jarvis 响应中提取核心建议"""
        # 简单启发式: 找 "suggest" / "recommend" / "should" 附近的句子
        if not response:
            return "No specific suggestion."
        sentences = response.split(". ")
        for s in sentences:
            lower = s.lower()
            if any(w in lower for w in ["suggest", "recommend", "should", "try", "consider"]):
                return s.strip()
        return sentences[-1].strip() if sentences else ""

    def _extract_risk_alert(
        self, response: str, observation: MultimodalObservation
    ) -> Optional[str]:
        """检测是否需要风险警告 (基于情绪, 不再使用 tension)"""
        if observation.partner_emotion in ("angry", "frustrated"):
            return "⚠ Partner showing negative emotions — tread carefully"
        if observation.user_emotion in ("angry", "frustrated"):
            return "⚠ You are showing signs of frustration — consider pausing"
        return None

    # ================================================================
    # 规则回退 (无模型可用时)
    # ================================================================

    def _generate_rule_based_response(
        self, observation: MultimodalObservation
    ) -> str:
        """基于规则的简单响应 (当 LLM 不可用时的回退方案)"""
        emotion = observation.partner_emotion

        templates = {
            "happy": "Speaker B appears to be in a positive mood. This is a good "
                     "opportunity to build rapport. Consider matching their energy "
                     "and steering the conversation toward productive topics.",
            "sad": "I'm detecting signs of sadness from Speaker B. They may need "
                   "empathy and support. Consider acknowledging their feelings "
                   "before moving to problem-solving.",
            "angry": "Warning — Speaker B is showing anger signals. I recommend "
                     "de-escalation: lower your voice, acknowledge their perspective, "
                     "and avoid defensive responses. Take a breath before responding.",
            "frustrated": "Speaker B seems frustrated. This may stem from feeling "
                          "misunderstood. Try paraphrasing their concern to show "
                          "you're listening, then offer a concrete next step.",
            "surprised": "Speaker B appears surprised — this could be an opportunity "
                         "to clarify or elaborate. Check if there's been a misunderstanding.",
            "fear": "Speaker B shows signs of anxiety or fear. Create a sense of "
                    "safety by being predictable and calm. Avoid sudden topic changes.",
            "disgust": "Speaker B appears dismissive or displeased. Consider whether "
                       "the current topic should be tabled and revisited later.",
            "neutral": "Speaker B appears neutral. Continue monitoring for emotional "
                       "shifts. This is a good time to establish shared goals.",
        }

        return templates.get(emotion, templates["neutral"])

    def _generate_rule_based_summary(
        self, turns, stats: Dict
    ) -> str:
        """基于规则的简单总结"""
        return (
            f"Session Summary (auto-generated):\n"
            f"- Duration: {stats.get('duration_seconds', 0)/60:.1f} minutes\n"
            f"- Total turns: {stats.get('total_turns', 0)}\n"
            f"- Dominant emotion: {stats.get('dominant_emotion', 'neutral')}\n"
            f"Note: LLM not available for detailed summary. "
            f"Connect to remote API for comprehensive analysis."
        )

    def _parse_summary_sections(self, text: str) -> Dict[str, str]:
        """解析 LLM 输出的总结各部分"""
        if not text:
            return {}

        sections = {}
        current_section = "summary"
        current_content = []

        section_markers = {
            "summary": ["session summary", "summary", "1.", "## summary"],
            "key_insights": ["key insights", "2.", "## key insights"],
            "partner_profile": ["speaker b profile", "partner profile", "3.", "## speaker b"],
            "user_feedback": ["speaker a", "user feedback", "4.", "## speaker a"],
            "next_session_prep": ["next session", "5.", "## next session"],
        }

        for line in text.split("\n"):
            line_lower = line.lower().strip()
            matched = None
            for section_name, markers in section_markers.items():
                if any(m in line_lower for m in markers):
                    matched = section_name
                    break

            if matched and matched != current_section:
                if current_content:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = matched
                current_content = []
            else:
                current_content.append(line)

        if current_content:
            sections[current_section] = "\n".join(current_content).strip()

        return sections
