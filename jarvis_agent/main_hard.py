#!/usr/bin/env python3
"""
============================================================
 Jarvis — 多模态智能应答系统 (赛题五)
 智能终端多模态输入感知与识别及个性化应答
============================================================

使用方法: 
  1. 分析单个 IEMOCAP 视频 + 音频:
     python main.py --session 1 --dialog Ses01F_impro01

  2. 批量处理 Session:
     python main.py --session 1 --batch

  3. 仅使用 transcription 精确时间戳 (更快, 不需要分析视频):
     python main.py --session 1 --dialog Ses01F_impro01 --use-transcription

  4. 使用本地 Qwen 2B 模型 (默认使用远程 API):
     python main.py --session 1 --dialog Ses01F_impro01 --use-local

  5. Demo 模式 (无需数据集, 使用模拟数据):
     python main.py --demo

依赖安装:
  pip install mediapipe faster-whisper openai pyyaml numpy opencv-python
  pip install transformers torch  # 如果需要本地 Qwen 2B
"""

import os
os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')  # 绕过 cuDNN DLL 缺失
import sys
import argparse
import time
import logging
from pathlib import Path
from typing import Optional, Dict, List

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.helpers import (
    load_config,
    setup_logging,
    parse_dialog_name_from_video,
    parse_iemocap_path,
)
from src.audio_analyzer import AudioAnalyzer
from dataclasses import dataclass, field

from src.context_manager import ContextManager, ConversationTurn
from src.jarvis_agent import JarvisAgent, MultimodalObservation
from prompts.system_prompts import EMOTION_ICONS

logger = logging.getLogger("main")

# ====================================================================
# 运行配置 — 直接改这里的路径就能跑
# ====================================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))          # jarvis_agent/
PARENT_DIR  = os.path.dirname(PROJECT_DIR)                        # 项目根目录

# IEMOCAP 数据根目录 (Session1/Session1 里放 dialog/avi, dialog/wav 等)
DATA_ROOT   = os.path.join(PARENT_DIR, "Session1", "Session1")

# 默认处理的对话
DEFAULT_SESSION = 1
DEFAULT_DIALOG  = "Ses01F_impro01"

# 是否默认启用本地模型 (True=本地+远程混合, False=纯远程)
USE_LOCAL = True

# 分析窗口: 按对话轮次(推荐) / 按时间
# 每 N 句 Whisper 转录分析一次 (无论说话人)
UTTERANCES_PER_WINDOW = 14     # 每14句话分析一次 (设0则用时间窗口)

# 单个视频文件路径 (如果用 --video 模式, 直接用这个路径或传参)
DEFAULT_VIDEO = os.path.join(DATA_ROOT, "dialog", "avi", "DivX",
                             "Ses01M_impro03.avi")
DEFAULT_VIDEO = os.path.join(DATA_ROOT, "dialog", "avi", "DivX",
                             "Ses01F_impro04.avi")                             

# ====================================================================


class JarvisPipeline:
    """
    Jarvis 主流水线 — 编排所有模块

    流程:
    1. 加载 IEMOCAP 视频 + 音频
    2. 每 20 秒窗口:
       a. VideoAnalyzer 分析视频情感
       b. AudioAnalyzer 转写语音
       c. FusionEngine 融合多模态数据
       d. JarvisAgent 生成个性化建议
    3. 对话结束 → 生成总结
    """

    def __init__(self, config_path: str = "config.yaml"):
        # 从任意目录运行都能找到 config.yaml
        if not os.path.isabs(config_path) and not os.path.exists(config_path):
            config_path = os.path.join(PROJECT_DIR, config_path)
        self.config = load_config(config_path)
        setup_logging(self.config)

        logger.info("=" * 60)
        logger.info("Initializing Jarvis Multimodal Agent System...")
        logger.info("=" * 60)

        # 初始化各模块 (去耦合, 各模块独立可替换)
        self.audio_analyzer: Optional[AudioAnalyzer] = None
        self.jarvis_agent: Optional[JarvisAgent] = None
        self.context_manager: Optional[ContextManager] = None
        self.window_size = self.config.get("fusion", {}).get("window_size", 20)

    def initialize(self):
        if not self.audio_analyzer:
            try:
                self.audio_analyzer = AudioAnalyzer(self.config)
                logger.info("✓ AudioAnalyzer initialized")
            except Exception as e:
                logger.error(f"✗ AudioAnalyzer failed: {e}")
                self.audio_analyzer = None
        if not self.jarvis_agent:
            self.jarvis_agent = JarvisAgent(self.config)
            self.jarvis_agent.initialize()
            self.context_manager = self.jarvis_agent.context_manager
        logger.info("✓ Pipeline ready")

    def run_on_iemocap_dialog(
        self,
        session_id: int,
        dialog_name: str,
        data_base_path: str,
        use_transcription: bool = False,
        use_local: bool = False,
    ):
        """
        对一个 IEMOCAP 对话运行完整流水线

        Args:
            session_id: Session 编号 (1-5)
            dialog_name: 对话名称 (如 Ses01F_impro01)
            data_base_path: IEMOCAP 数据根目录
            use_transcription: 是否使用精确 transcription 时间戳
            use_local: 是否优先使用本地模型
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {dialog_name} (Session {session_id})")
        logger.info(f"{'='*60}")

        # --- 加载对话数据 ---
        video_path = None
        audio_path = None
        transcription_path = None

        if not use_transcription:
            video_path = parse_iemocap_path(
                session_id, dialog_name, "avi", data_base_path
            )
            audio_path = parse_iemocap_path(
                session_id, dialog_name, "wav", data_base_path
            )

        transcription_path = parse_iemocap_path(
            session_id, dialog_name, "transcriptions", data_base_path
        )

        if transcription_path:
            logger.info(f"Transcription: {transcription_path}")

        # --- 解析 transcription 文件获取时间范围 ---
        transcription_data = []
        total_duration = 0.0

        if transcription_path and os.path.exists(transcription_path):
            transcription_data = self._parse_transcription(transcription_path)
            if transcription_data:
                total_duration = transcription_data[-1]["end"]
                logger.info(
                    f"Parsed {len(transcription_data)} utterances, "
                    f"duration: {total_duration:.1f}s"
                )

        # 如果找不到 transcription, 用视频/音频长度
        if total_duration == 0:
            if video_path:
                import cv2
                cap = cv2.VideoCapture(video_path)
                total_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
                cap.release()
            else:
                total_duration = 120  # 默认假设2分钟

        # --- 分析全段音频 (一次转写, 分段使用) ---
        audio_result = None
        if audio_path and os.path.exists(audio_path) and self.audio_analyzer:
            logger.info(f"Transcribing audio: {audio_path}")
            audio_result = self.audio_analyzer.transcribe_segment(audio_path)

            if audio_result.full_transcript:
                logger.info(f"Transcript preview: {audio_result.full_transcript[:200]}...")

        # --- 按 20s 窗口滑动处理 ---
        session_id_str = f"{dialog_name}_{int(time.time())}"
        window_count = int(total_duration / self.window_size) + 1

        logger.info(
            f"Starting sliding window analysis: {window_count} windows "
            f"of {self.window_size}s each"
        )

        all_responses = []

        for win_idx in range(window_count):
            win_start = win_idx * self.window_size
            win_end = min(win_start + self.window_size, total_duration)

            if win_end - win_start < 5:  # 跳过太短的尾巴
                break

            logger.info(f"\n--- Window {win_idx+1}/{window_count} "
                        f"[{win_start:.1f}s - {win_end:.1f}s] ---")

            # Step 1: 视频分析
            video_result = None
            if video_path and os.path.exists(video_path) and self.video_analyzer:
                try:
                    video_result = self.video_analyzer.analyze_video_segment(
                        video_path, win_start, win_end
                    )
                    logger.info(
                        f"  Video: {video_result.dominant_emotion} "
                        f"(faces: {video_result.faces_detected}/{video_result.frames_processed})"
                    )
                except Exception as e:
                    logger.error(f"  Video analysis error: {e}")

            # Step 2: 融合 (使用 transcription 时间戳或音频结果)
            if use_transcription and transcription_data:
                observation = self.fusion_engine.fuse_with_transcription(
                    video_result=video_result,
                    transcription_data=transcription_data,
                    window_start=win_start,
                    window_end=win_end,
                )
            else:
                observation = self.fusion_engine.fuse(
                    video_result=video_result,
                    audio_result=audio_result,
                    window_start=win_start,
                    window_end=win_end,
                )

            # Step 3: Jarvis Agent 推理
            try:
                response = self.jarvis_agent.analyze_and_respond(
                    observation=observation,
                    session_id=session_id_str,
                    use_local=use_local,
                )
                all_responses.append(response)

                # --- 控制台输出 ---
                self._print_jarvis_output(response, observation, win_idx + 1)

            except Exception as e:
                logger.error(f"  Jarvis Agent error: {e}")
                import traceback
                traceback.print_exc()

        # --- 对话结束 → 生成总结 ---
        logger.info(f"\n{'='*60}")
        logger.info("Conversation ended. Generating session summary...")
        logger.info(f"{'='*60}")

        summary = self.jarvis_agent.generate_session_summary(session_id_str)

        # --- 更新用户画像 ---
        stats = self.context_manager.get_conversation_stats(session_id_str)
        user_profile = {
            "last_session": session_id_str,
            "total_sessions": len(self.context_manager.get_all_summaries()),
            "preferred_tone": "professional",
            "communication_style": "direct",
            "strengths": self._infer_strengths(all_responses),
            "weaknesses": self._infer_weaknesses(all_responses),
            "past_takeaways": summary.summary[:300] if summary else "",
            "partner_traits": summary.partner_profile[:300] if summary else "",
        }
        self.context_manager.update_user_profile("Speaker_A", user_profile)

        # --- 输出总结 ---
        self._print_session_summary(summary, stats)

        logger.info(f"\nAll outputs saved to: output/")
        return all_responses, summary

    def run_on_video(
        self,
        video_path: str,
        use_local: bool = False,
    ):
        """
        从单个 .avi 视频文件提取视频流和音频流, 运行完整流水线

        不需要单独的 .wav 文件 — 音频从视频中自动提取 (ffmpeg)

        Args:
            video_path: .avi 视频文件路径
            use_local: 是否使用本地模型做简单推理
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing video: {video_path}")
        logger.info(f"{'='*60}")

        # --- 从视频提取音频 (尝试多种方式) ---
        import tempfile, subprocess, shutil

        # 持久化音频到 output/ (供声纹F0分析使用)
        os.makedirs(os.path.join(PROJECT_DIR, "output"), exist_ok=True)
        audio_tmp = os.path.join(PROJECT_DIR, "output", f"{os.path.splitext(os.path.basename(video_path))[0]}_audio.wav")
        extracted = False

        # 方式1: 系统 ffmpeg
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            logger.info(f"Extracting audio via ffmpeg...")
            subprocess.run(
                [ffmpeg, "-y", "-i", video_path, "-vn",
                 "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_tmp],
                capture_output=True, check=True,
            )
            extracted = True

        # 方式2: 项目目录下的 ffmpeg.exe
        if not extracted:
            local_ffmpeg = os.path.join(os.path.dirname(__file__), "ffmpeg.exe")
            if os.path.exists(local_ffmpeg):
                logger.info(f"Extracting audio via local ffmpeg...")
                subprocess.run(
                    [local_ffmpeg, "-y", "-i", video_path, "-vn",
                     "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_tmp],
                    capture_output=True, check=True,
                )
                extracted = True

        # 方式3: imageio-ffmpeg (Python 包自带 ffmpeg 二进制)
        if not extracted:
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                logger.info(f"Extracting audio via imageio-ffmpeg...")
                subprocess.run(
                    [ffmpeg_exe, "-y", "-i", video_path, "-vn",
                     "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_tmp],
                    capture_output=True, check=True,
                )
                extracted = True
            except Exception:
                pass

        if not extracted:
            raise RuntimeError(
                "Cannot extract audio from video. Install one of:\n"
                "  pip install imageio-ffmpeg\n"
                "  winget install ffmpeg"
            )

        # --- 获取视频时长 ---
        import cv2
        cap = cv2.VideoCapture(video_path)
        total_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1)
        cap.release()

        logger.info(f"Video duration: {total_duration:.1f}s")

        # --- 初始化模块 ---
        if not self.audio_analyzer:
            self.initialize()

        # --- 匹配 IEMOCAP transcription (精确转写+时间戳+说话人) ---
        from src.audio_analyzer import SpeechSegment, AudioAnalysisResult
        dialog_name = os.path.splitext(os.path.basename(video_path))[0]
        transcription_path = os.path.join(os.path.dirname(video_path), "..", "..",
                                          "transcriptions", f"{dialog_name}.txt")
        transcription_path = os.path.normpath(transcription_path)
        audio_result = None

        if os.path.exists(transcription_path):
            utterances = self._parse_transcription(transcription_path)
            logger.info(f"Loaded {len(utterances)} utterances from IEMOCAP transcription")
            audio_result = AudioAnalysisResult(
                dialog_name=dialog_name, duration=total_duration,
                full_transcript="\n".join(
                    f"[{u['start']:.1f}s-{u['end']:.1f}s] "
                    f"{'You' if u['speaker']=='F' else 'Partner'}: {u['text']}"
                    for u in utterances),
                segments=[SpeechSegment(
                    start_time=u["start"], end_time=u["end"],
                    text=u["text"],
                    speaker_id="SPK_0" if u["speaker"] == "F" else "SPK_1"
                ) for u in utterances],
            )
        else:
            logger.warning(f"No transcription at {transcription_path}, fallback to Whisper")
            audio_result = self.audio_analyzer.transcribe_with_speakers(audio_tmp) if self.audio_analyzer else None

        # --- 分析: 按轮次(推荐) 或 按时间窗口 ---
        session_id_str = f"{dialog_name}_{int(time.time())}"
        self._audio_path = audio_tmp  # 保存音频路径供 run_turn_based 使用
        all_responses = []

        # --- 视频人脸表情 (ONNX模型, 每句采样4帧) ---
        if audio_result and audio_result.segments:
            try:
                from src.face_emotion import FaceEmotionAnalyzer
                fea = FaceEmotionAnalyzer(dialog_name=dialog_name)
                utt_list = [{"start": s.start_time, "end": s.end_time}
                            for s in audio_result.segments]
                face_results = fea.analyze_dialog(video_path, utt_list)
                for seg, fr in zip(audio_result.segments, face_results):
                    seg._face_left = fr.get("left_emotion", "neutral")
                    seg._face_right = fr.get("right_emotion", "neutral")
                    seg._face_result = fr 
                logger.info(f"Face emotion: {len(face_results)} utterances analyzed")
            except Exception as e:
                logger.warning(f"Face emotion failed: {e}")

        # 已有转录精确说话人, 不做声纹分离
        if UTTERANCES_PER_WINDOW > 0 and audio_result and audio_result.segments:
            # === 按对话句分析 (每N句触发) ===
            all_responses, summary = self.run_turn_based(
                video_path=video_path,
                audio_result=audio_result,
                session_id=session_id_str,
                use_local=use_local,
            )
        else:
            # === 按时间窗口分析 (fallback) ===
            window_count = int(total_duration / self.window_size) + 1
            for win_idx in range(window_count):
                win_start = win_idx * self.window_size
                win_end = min(win_start + self.window_size, total_duration)
                if win_end - win_start < 5: break

                logger.info(f"\n--- Window {win_idx+1}/{window_count} "
                            f"[{win_start:.1f}s - {win_end:.1f}s] ---")

                video_result = None
                if self.video_analyzer:
                    try:
                        video_result = self.video_analyzer.analyze_video_segment(
                            video_path, win_start, win_end)
                    except Exception as e:
                        logger.error(f"Video error: {e}")

                observation = self.fusion_engine.fuse(
                    video_result=video_result, audio_result=audio_result,
                    window_start=win_start, window_end=win_end)

                try:
                    response = self.jarvis_agent.analyze_and_respond(
                        observation=observation, session_id=session_id_str,
                        use_local=use_local)
                    all_responses.append(response)
                    self._print_jarvis_output(response, observation, win_idx + 1)
                except Exception as e:
                    logger.error(f"Agent error: {e}")

            # 生成总结 (仅时间窗口模式; 轮次模式已在 run_turn_based 里生成)
            summary = self.jarvis_agent.generate_session_summary(session_id_str)
            stats = self.context_manager.get_conversation_stats(session_id_str)
            self._print_session_summary(summary, stats)
            self._write_clean_log(session_id_str, audio_result, all_responses)

        return all_responses, summary

    
    def run_turn_based(
            self, video_path: str, audio_result, session_id: str, use_local: bool,
        ):
        """
        按对话轮次分析 — 每 N 句触发一次 Jarvis.
        同时收集窗口内每句话的语音和面部概率向量（用于KL散度）。
        """
        n_per_window = UTTERANCES_PER_WINDOW
        segments = sorted(audio_result.segments, key=lambda s: s.start_time)
        total_utts = len(segments)
        logger.info(f"Turn-based: {total_utts} utterances, every {n_per_window}")

        all_responses = []
        window_num = 0

        for i in range(0, total_utts, n_per_window):
            batch = segments[i:i + n_per_window]
            if not batch:
                break
            window_num += 1

            win_start, win_end = batch[0].start_time, batch[-1].end_time

            # 逐句带情绪标注（仅用于显示）
            wav_path = getattr(self, '_audio_path',
                               os.path.join(PROJECT_DIR, "output", f"{session_id}_audio.wav"))
            user_parts, partner_parts = [], []

            # ===== 新增：收集窗口内每句话的概率向量 =====
            voice_probs_user = []          # 每个元素是 dict（9类概率）
            voice_probs_partner = []
            face_probs_user = []           # 每个元素是 7维 list
            face_probs_partner = []
            face_valid_ratios_user = []    # 每个元素是 float (有效帧占比)
            face_valid_ratios_partner = []

            for seg in batch:
                # ---- 语音情绪 + 概率 ----
                v_emo = "?"
                v_all_probs = {}  # 9维概率字典
                if self.audio_analyzer and self.audio_analyzer.emotion_model and os.path.exists(wav_path):
                    try:
                        import librosa
                        dur = max(seg.end_time - seg.start_time, 0.3)
                        wav_seg, sr = librosa.load(wav_path, sr=16000,
                                                offset=max(0, seg.start_time - 0.1),
                                                duration=dur + 0.2)
                        if len(wav_seg) > sr * 0.05:
                            pred = self.audio_analyzer.predict_emotion(wav_seg, sr)
                            v_emo = pred["label"]
                            v_all_probs = pred.get("all", {})  # 获取完整9类概率
                    except:
                        pass

                # ---- 面部情绪 + 概率（从 _face_result 取） ----
                f_emo = "?"
                f_probs = [0.0] * 7  # 7维
                valid_ratio = 0.0    # 默认无有效帧
                face_res = getattr(seg, '_face_result', None)
                if face_res:
                    if seg.speaker_id == "SPK_0":
                        f_emo = face_res.get("left_emotion", "?")
                        f_probs = face_res.get("left_probs", [0.0] * 7)
                        valid_ratio = face_res.get("valid_ratio_left", 0.0)
                    else:
                        f_emo = face_res.get("right_emotion", "?")
                        f_probs = face_res.get("right_probs", [0.0] * 7)
                        valid_ratio = face_res.get("valid_ratio_right", 0.0)

                # ---- 保存概率和有效占比到对应的列表 ----
                if v_all_probs:
                    if seg.speaker_id == "SPK_0":
                        voice_probs_user.append(v_all_probs)
                    else:
                        voice_probs_partner.append(v_all_probs)
                else:
                    # 添加默认均匀分布，避免长度不一致
                    default_probs = {
                        'frustrated': 1.0/9, 'neutral': 1.0/9, 'angry': 1.0/9,
                        'sad': 1.0/9, 'excited': 1.0/9, 'happy': 1.0/9,
                        'disgust': 1.0/9, 'fear': 1.0/9, 'unclear': 1.0/9
                    }
                    if seg.speaker_id == "SPK_0":
                        voice_probs_user.append(default_probs)
                    else:
                        voice_probs_partner.append(default_probs)

                # 面部概率列表：即使无效也添加全零（保持长度一致）
                if seg.speaker_id == "SPK_0":
                    face_probs_user.append(f_probs)
                    face_valid_ratios_user.append(valid_ratio)
                else:
                    face_probs_partner.append(f_probs)
                    face_valid_ratios_partner.append(valid_ratio)

                # ---- 构建 Top2 格式的对话文本 ----
                # 从 v_all_probs 提取 Top2 语音
                if v_all_probs:
                    sorted_v = sorted(v_all_probs.items(), key=lambda x: x[1], reverse=True)
                    top_v1, top_v2 = sorted_v[0], (sorted_v[1] if len(sorted_v) > 1 else sorted_v[0])
                    voice_str = f"{top_v1[0]}({top_v1[1]:.2f})/{top_v2[0]}({top_v2[1]:.2f})"
                else:
                    voice_str = "?"

                # 从 f_probs 提取 Top2 面部
                if any(f_probs):
                    face_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
                    sorted_idx = sorted(range(len(f_probs)), key=lambda i: f_probs[i], reverse=True)
                    top_f1 = (face_labels[sorted_idx[0]], f_probs[sorted_idx[0]])
                    top_f2 = (face_labels[sorted_idx[1]], f_probs[sorted_idx[1]]) if len(sorted_idx) > 1 else top_f1
                    face_str = f"{top_f1[0]}({top_f1[1]:.2f})/{top_f2[0]}({top_f2[1]:.2f})"
                else:
                    face_str = "?"

                seg._top2_text = f"[voice: {voice_str} | face: {face_str}] {seg.text}"

                # user_parts 只存纯文本，Top2 由 _build_analysis_prompt_with_summary 负责
                if seg.speaker_id == "SPK_0":
                    user_parts.append(seg.text)
                else:
                    partner_parts.append(seg.text)

            user_text = " | ".join(user_parts) if user_parts else "(silence)"
            partner_text = " | ".join(partner_parts) if partner_parts else "(silence)"

            # ===== 原有语音综合情绪（兼容旧逻辑） =====
            user_emotion = "neutral"
            partner_emotion = "neutral"
            if self.audio_analyzer and self.audio_analyzer.emotion_model:
                try:
                    import librosa, numpy as np
                    if os.path.exists(wav_path):
                        full_wav, sr = librosa.load(wav_path, sr=16000)
                        user_clips = []
                        partner_clips = []
                        if hasattr(audio_result, '_diar_segments') and audio_result._diar_segments:
                            for ds in audio_result._diar_segments:
                                if ds.start_time > win_end or ds.end_time < win_start:
                                    continue
                                t1 = max(win_start, ds.start_time)
                                t2 = min(win_end, ds.end_time)
                                if t2 - t1 < 0.2:
                                    continue
                                clip = full_wav[int(t1 * sr):int(t2 * sr)]
                                if ds.speaker_id == "SPK_0":
                                    user_clips.append(clip)
                                else:
                                    partner_clips.append(clip)
                        if user_clips:
                            user_audio = np.concatenate(user_clips)[:sr * 5]
                            if len(user_audio) > sr * 0.3:
                                user_emotion = self.audio_analyzer.predict_emotion(user_audio, sr)["label"]
                        if partner_clips:
                            partner_audio = np.concatenate(partner_clips)[:sr * 5]
                            if len(partner_audio) > sr * 0.3:
                                partner_emotion = self.audio_analyzer.predict_emotion(partner_audio, sr)["label"]
                except Exception:
                    pass

            # ===== 原有面部综合情绪（硬标签，仅用于显示） =====
            face_left, face_right = [], []
            for si in range(i, min(i + n_per_window, len(segments))):
                if hasattr(segments[si], '_face_result'):
                    fr = segments[si]._face_result
                    face_left.append(fr.get("left_emotion", "neutral"))
                    face_right.append(fr.get("right_emotion", "neutral"))
            from collections import Counter as _C
            face_l = _C(face_left).most_common(1)[0][0] if face_left else "neutral"
            face_r = _C(face_right).most_common(1)[0][0] if face_right else "neutral"
            user_face_emo = face_l
            partner_face_emo = face_r

            # ===== 构建观测对象 =====
            obs = MultimodalObservation(
                timestamp=(win_start + win_end) / 2,
                window_start=win_start, window_end=win_end,
                user_emotion="neutral",
                partner_emotion="neutral",
                user_emotion_conf=0.65, partner_emotion_conf=0.65,
                user_valence=2.5, user_arousal=3.0,
                partner_valence=2.5, partner_arousal=3.0,
                user_speech=user_text,
                partner_speech=partner_text,
                full_transcript=f"You: {user_text}\nPartner: {partner_text}",
                # ===== 存储窗口内所有句子的概率向量 =====
                user_voice_probs_list=voice_probs_user,
                partner_voice_probs_list=voice_probs_partner,
                user_face_probs_list=face_probs_user,
                partner_face_probs_list=face_probs_partner,
            )
            # ===== 新增：存储有效帧占比列表 =====
            obs.user_face_valid_ratios = face_valid_ratios_user
            obs.partner_face_valid_ratios = face_valid_ratios_partner

            # ===== 构建 Top2 格式的历史文本 =====
            user_text_top2 = " | ".join(
                [getattr(seg, '_top2_text', seg.text) for seg in batch
                 if seg.speaker_id == "SPK_0"]
            ) or "(silence)"
            partner_text_top2 = " | ".join(
                [getattr(seg, '_top2_text', seg.text) for seg in batch
                 if seg.speaker_id != "SPK_0"]
            ) or "(silence)"
            obs.user_speech_top2 = user_text_top2
            obs.partner_speech_top2 = partner_text_top2

            # 将窗口大小传递给 Agent（用于 Prompt 显示）
            self.jarvis_agent.utt_per_window = UTTERANCES_PER_WINDOW

            try:
                response = self.jarvis_agent.analyze_and_respond(
                    observation=obs, session_id=session_id, use_local=use_local)
                all_responses.append(response)
                self._print_jarvis_output(response, obs, window_num)
            except Exception as e:
                logger.error(f"Agent error: {e}")

        summary = self.jarvis_agent.generate_session_summary(session_id)
        stats = self.context_manager.get_conversation_stats(session_id)
        self._print_session_summary(summary, stats)
        self._write_clean_log(session_id, audio_result, all_responses)
        return all_responses, summary
    
    def _dominant_emotion(self, dets) -> str:
        from collections import Counter
        return Counter(d.emotion for d in dets).most_common(1)[0][0] if dets else "neutral"

    def _apply_voice_diarization(self, audio_result):
        """
        用 pyannote 声纹结果标注每个 segment 的说话人 + 识别男女.

        流程:
        1. pyannote 聚类出两个声纹簇 → SPK_A, SPK_B
        2. 每簇采样音频片段, 估计基频 F0
        3. 高 F0 → 女声 → You; 低 F0 → 男声 → Partner
        4. 对每个 Whisper segment, 用声纹时间轴标注说话人
        """
        from src.audio_analyzer import SpeechSegment
        import numpy as np
        import wave

        if not hasattr(audio_result, '_diar_segments'):
            logger.warning("No voice diarization data available")
            return

        diar_segs = audio_result._diar_segments
        if not diar_segs:
            return

        logger.info(f"Voice diarization: {len(diar_segs)} speaker segments")

        # 说话人映射: 先出声的 → SPK_0 (主视角/You)
        # (IEMOCAP 视频镜头对着的人通常先开口)
        first_speaker = diar_segs[0].speaker_id if diar_segs else "SPK_00"
        other_speaker = [s for s in set(ds.speaker_id for ds in diar_segs) if s != first_speaker]
        other_speaker = other_speaker[0] if other_speaker else "SPK_01"
        cluster_map = {first_speaker: "SPK_0", other_speaker: "SPK_1"}
        logger.info(f"Voice map: first={first_speaker}->You, {other_speaker}->Partner")
        for ds in diar_segs:
            ds.speaker_id = cluster_map.get(ds.speaker_id, ds.speaker_id)

        # Step 2: 对每个 Whisper segment, 用声纹时间轴标注说话人
        for seg in audio_result.segments:
            mid = (seg.start_time + seg.end_time) / 2
            best_speaker = "SPK_0"
            best_overlap = 0

            for ds in diar_segs:
                overlap = min(seg.end_time, ds.end_time) - max(seg.start_time, ds.start_time)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = ds.speaker_id

            seg.speaker_id = best_speaker

        # 统计
        spk_counts = {}
        for s in audio_result.segments:
            spk_counts[s.speaker_id] = spk_counts.get(s.speaker_id, 0) + 1
        logger.info(f"Speaker distribution: {spk_counts}")

    def _merge_same_speaker(self, audio_result):
        """
        合并连续同一说话人的 segment.
        声纹确定了说话人后, 相邻相同说话人的片段合并成一句.
        """
        from src.audio_analyzer import SpeechSegment
        if not audio_result.segments:
            return

        merged = []
        current = audio_result.segments[0]

        for seg in audio_result.segments[1:]:
            if seg.speaker_id == current.speaker_id:
                # 同一人继续 → 合并
                current.text += " " + seg.text
                current.end_time = seg.end_time
            else:
                merged.append(current)
                current = seg

        merged.append(current)
        audio_result.segments = merged
        logger.info(f"Speaker merge: → {len(merged)} utterances")

    def _align_speech_to_speaker_by_lip(self, video_path: str, audio_result):
        """(deprecated — replaced by _apply_voice_diarization + _merge_same_speaker)"""
        pass
        import cv2, numpy as np
        import mediapipe as mp
        from src.audio_analyzer import SpeechSegment

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_h, frame_w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        # Step 1: 收集所有词 + 时间戳
        all_words = []
        for seg in audio_result.segments:
            words = getattr(seg, '_words', None)
            if words:
                all_words.extend(words)
            else:
                all_words.append({
                    "word": seg.text, "start": seg.start_time,
                    "end": seg.end_time, "prob": seg.confidence,
                })

        if not all_words:
            cap.release()
            return

        logger.info(f"Word-level lip alignment: {len(all_words)} words")

        # Step 2: 逐词判断说话人 (用中间时间点的一帧)
        word_speakers = []
        for wd in all_words:
            t = (wd["start"] + wd["end"]) / 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ret, frame = cap.read()
            speaker = "SPK_?"
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_bright = cv2.convertScaleAbs(frame_rgb, alpha=1.4, beta=30)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_bright)

                if (hasattr(self.video_analyzer, 'face_detector') and
                    self.video_analyzer.face_detector is not None):
                    fd_result = self.video_analyzer.face_detector.detect(mp_img)
                    scores = {}
                    num_faces = len(fd_result.detections)
                    if num_faces >= 2:
                        # 两张脸都看到 → 纯嘴部运动判断(最准)
                        for det in fd_result.detections[:2]:
                            bbox = det.bounding_box
                            fx, fy, bfw, bfh = (int(bbox.origin_x), int(bbox.origin_y),
                                                int(bbox.width), int(bbox.height))
                            spk_id = "SPK_0" if (fx + bfw/2) < frame_w * 0.55 else "SPK_1"
                            my1 = max(0, fy + int(bfh * 0.55))
                            my2 = min(frame_h, fy + int(bfh * 0.9))
                            mx1 = max(0, fx + int(bfw * 0.2))
                            mx2 = min(frame_w, fx + int(bfw * 0.8))
                            if my2 > my1 and mx2 > mx1:
                                mouth = frame_bright[my1:my2, mx1:mx2]
                                gray = cv2.cvtColor(mouth, cv2.COLOR_RGB2GRAY)
                                scores[spk_id] = cv2.Laplacian(gray, cv2.CV_64F).var()
                        if scores:
                            speaker = max(scores, key=scores.get)
                    else:
                        # 只看到一张脸 → 嘴部判断画面里的人是否在说,
                        # 如果嘴没动 → 画面外的人在说 → 用原来的声纹结果
                        pass  # speaker stays "SPK_?" → will use audio diarization
            word_speakers.append(speaker)

        # Step 2.5: 仍然 SPK_? 的词用音频声纹结果兜底
        audio_speaker_map = {}
        for seg_orig in audio_result.segments:
            mid = (seg_orig.start_time + seg_orig.end_time) / 2
            audio_speaker_map[mid] = getattr(seg_orig, '_audio_speaker', seg_orig.speaker_id)

        for i in range(len(word_speakers)):
            if word_speakers[i] == "SPK_?":
                t = (all_words[i]["start"] + all_words[i]["end"]) / 2
                # 找最近的音频声纹标注
                best_spk = "SPK_0"  # default
                best_dist = 999
                for at, aspk in audio_speaker_map.items():
                    d = abs(at - t)
                    if d < best_dist and aspk != "SPK_?":
                        best_dist, best_spk = d, aspk
                word_speakers[i] = best_spk

        # Step 3: 按说话人分组 → 切分为清晰句子
        new_segments = []
        current_words = []
        current_speaker = None
        current_start = None

        for i, (w, spk) in enumerate(zip(all_words, word_speakers)):
            word_text = w["word"].strip()
            if not word_text:
                continue

            if spk != current_speaker and current_words and spk != "SPK_?":
                # 说话人切换 → 输出上一句
                text = " ".join(current_words)
                new_segments.append(SpeechSegment(
                    start_time=current_start or w["start"],
                    end_time=w["start"],
                    text=text, speaker_id=current_speaker or "SPK_?",
                ))
                current_words = []
                current_start = w["start"]

            current_words.append(word_text)
            if current_speaker is None or spk != "SPK_?":
                current_speaker = spk
            if current_start is None:
                current_start = w["start"]

        # 最后一句
        if current_words:
            new_segments.append(SpeechSegment(
                start_time=current_start or all_words[-1]["start"],
                end_time=all_words[-1]["end"],
                text=" ".join(current_words),
                speaker_id=current_speaker or "SPK_?",
            ))

        if new_segments:
            audio_result.segments = new_segments
            audio_result.full_transcript = "\n".join(
                f"[{s.start_time:.1f}s-{s.end_time:.1f}s] "
                f"{'You' if s.speaker_id=='SPK_0' else 'Partner'}: {s.text}"
                for s in new_segments
            )
            logger.info(
                f"Word-level lip diarization: {len(audio_result.segments)} "
                f"clean utterances (from {len(all_words)} words)"
            )

        cap.release()

    def _merge_split_sentences(self, segments, responses):
        """
        规则合并被切碎的句子:
        1. 同一说话人的连续片段 → 合并
        2. A说话人的片段以逗号/无标点结尾 + 下一片段是B说话人且首字母小写
           → 很可能B的片段其实是A的延续 → 检查并合并到A
        3. 去掉纯标点/单字母的碎片
        """
        if len(segments) < 3:
            return segments

        import re
        from src.audio_analyzer import SpeechSegment
        merged = []
        skip_next = False

        for i, seg in enumerate(segments):
            if skip_next:
                skip_next = False
                continue

            text = seg.text.strip()
            if not text or len(text) <= 1:
                continue  # 跳过碎片

            # 检查下一段是否是当前段的延续 (不同说话人, 但内容明显属于当前段)
            if i + 1 < len(segments):
                next_seg = segments[i + 1]
                # 当前段不以句号/问号/感叹号结尾 → 可能是被切断了
                ends_open = not text.endswith(('.', '?', '!', '."', '?"'))
                # 下一段首字母小写 → 很可能是延续
                next_starts_lower = (next_seg.text.strip()[:1].islower()
                                    if next_seg.text.strip() else False)
                # 下一段很短 (< 5个词) → 可能是碎片
                next_short = len(next_seg.text.split()) < 5

                if ends_open and next_starts_lower and next_seg.speaker_id != seg.speaker_id:
                    # 合并: 把下一段的内容追加到当前段
                    text = text.rstrip().rstrip(',').rstrip('.') + ", " + next_seg.text.strip()
                    skip_next = True
                elif ends_open and next_short and next_seg.speaker_id != seg.speaker_id:
                    text = text.rstrip() + " " + next_seg.text.strip()
                    skip_next = True

            merged.append(SpeechSegment(
                start_time=seg.start_time, end_time=seg.end_time,
                text=text, speaker_id=seg.speaker_id))

        # 第二轮: 合并连续相同说话人
        cleaned = []
        for seg in merged:
            if cleaned and seg.speaker_id == cleaned[-1].speaker_id:
                cleaned[-1].text += " " + seg.text
            else:
                cleaned.append(seg)

        if len(cleaned) != len(segments):
            logger.info(f"Rule merge: {len(segments)} → {len(cleaned)} utterances")
        else:
            logger.info(f"Rule merge: no change ({len(segments)} utterances)")
        return cleaned

    def _write_clean_log(self, session_id: str, audio_result, responses):
        """生成干净的对话+分析日志"""
        log_path = os.path.join(PROJECT_DIR, "output",
                                f"{session_id}_conversation_log.txt")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"{'='*60}\n")
            f.write(f"  Jarvis Conversation Log\n")
            f.write(f"  Session: {session_id}\n")
            f.write(f"{'='*60}\n\n")

            n_per = UTTERANCES_PER_WINDOW
            segs = audio_result.segments
            resp_idx = 0

            for i in range(0, len(segs), n_per):
                batch = segs[i:i + n_per]
                f.write(f"{'─'*60}\n")
                f.write(f"  ROUND {i//n_per + 1}  [{batch[0].start_time:.1f}s - {batch[-1].end_time:.1f}s]\n")
                f.write(f"{'─'*60}\n")

                for seg in batch:
                    speaker = "You    " if seg.speaker_id == "SPK_0" else "Partner"
                    # 语音情绪 (用对话名精确匹配音频文件)
                    voice_emo = "?"
                    if self.audio_analyzer and self.audio_analyzer.emotion_model:
                        try:
                            import librosa
                            wav_path = getattr(self, '_audio_path',
                                               os.path.join(PROJECT_DIR, "output", f"{session_id}_audio.wav"))
                            if os.path.exists(wav_path):
                                dur = max(seg.end_time-seg.start_time, 0.3)
                                offset = max(0, seg.start_time - 0.15)
                                wav, sr = librosa.load(wav_path, sr=16000,
                                                       offset=offset, duration=dur+0.3)
                                if len(wav) > sr * 0.05:
                                    voice_emo = self.audio_analyzer.predict_emotion(wav, sr)["label"]
                        except Exception:
                            pass
                    # 视频人脸情绪 (逐句采样)
                    face_emo = getattr(seg, '_face_left', None) if seg.speaker_id == "SPK_0" else getattr(seg, '_face_right', None)
                    face_emo = face_emo if face_emo else "?"
                    emo_tag = f" [voice:{voice_emo} face:{face_emo}]"
                    f.write(f"  {speaker}{emo_tag}: {seg.text}\n")

                if resp_idx < len(responses):
                    resp = responses[resp_idx]
                    f.write(f"\n  >>> JARVIS ({resp.generated_by}) <<<\n")
                    f.write(f"  Emotion: You={resp.user_emotion_label} | Partner={resp.partner_emotion_label}\n")
                    for line in resp.analysis.split("\n"):
                        f.write(f"  {line}\n")
                    if resp.risk_alert:
                        f.write(f"  !! {resp.risk_alert}\n")
                f.write(f"\n")
                resp_idx += 1

        logger.info(f"Clean log written to: {log_path}")

    def run_demo(self, use_local: bool = False):
        """Demo 模式 — 使用模拟数据演示系统流程"""
        logger.info("\n" + "=" * 60)
        logger.info(f"JARVIS DEMO MODE — use_local={use_local}")
        logger.info("=" * 60)

        self.initialize()

        # 模拟一段对话
        demo_scenes = [
            {
                "user_emotion": "neutral", "partner_emotion": "frustrated",
                "user_speech": "Is there a problem?",
                "partner_speech": "Who told you to get in this line?",
                "phase": "opening",
            },
            {
                "user_emotion": "surprised", "partner_emotion": "angry",
                "user_speech": "You did. You were standing at the beginning.",
                "partner_speech": "Okay. But I didn't tell you to get in this line.",
                "phase": "development",
            },
            {
                "user_emotion": "frustrated", "partner_emotion": "angry",
                "user_speech": "How am I supposed to get an ID without an ID?",
                "partner_speech": "I need an ID to pass this form along.",
                "phase": "climax",
            },
            {
                "user_emotion": "angry", "partner_emotion": "frustrated",
                "user_speech": "That's out of control!",
                "partner_speech": "I don't understand why this is so complicated.",
                "phase": "climax",
            },
            {
                "user_emotion": "frustrated", "partner_emotion": "neutral",
                "user_speech": "Do you have a supervisor?",
                "partner_speech": "Yeah. Do you want to see my supervisor? Fine.",
                "phase": "resolution",
            },
        ]

        session_id = f"demo_{int(time.time())}"
        all_responses = []

        for i, scene in enumerate(demo_scenes):
            logger.info(f"\n--- Demo Scene {i+1}/{len(demo_scenes)} ---")

            observation = MultimodalObservation(
                timestamp=i * 20,
                window_start=i * 20,
                window_end=(i + 1) * 20,
                user_emotion=scene["user_emotion"],
                partner_emotion=scene["partner_emotion"],
                user_speech=scene["user_speech"],
                partner_speech=scene["partner_speech"],
                # user_emotion_trend="stable" if i < 2 else "rising" if i < 4 else "falling",
                scene_context="DMV office — ID verification dispute",
                dialog_type="improvisation",
            )

            response = self.jarvis_agent.analyze_and_respond(
                observation=observation,
                session_id=session_id,
                use_local=use_local,
            )
            all_responses.append(response)

            self._print_jarvis_output(response, observation, i + 1)

        # 生成总结
        summary = self.jarvis_agent.generate_session_summary(session_id)
        stats = self.context_manager.get_conversation_stats(session_id)
        self._print_session_summary(summary, stats)

        logger.info("\nDemo completed! Check output/ for saved data.")

    # ================================================================
    # 辅助方法
    # ================================================================

    def _parse_transcription(self, path: str) -> List[Dict]:
        """解析 IEMOCAP transcription 文件"""
        import re
        utterances = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # 格式: Ses01F_impro01_F000 [006.2901-008.2357]: Excuse me.
                match = re.match(
                    r"(\S+)\s+\[(\d+\.?\d*)-(\d+\.?\d*)\]:\s+(.*)",
                    line,
                )
                if match:
                    sentence_id = match.group(1)
                    start = float(match.group(2))
                    end = float(match.group(3))
                    text = match.group(4)

                    # 从 sentence_id 解析说话人性别
                    parts = sentence_id.rsplit("_", 1)
                    if len(parts) == 2:
                        speaker = parts[1][0]  # F or M
                    else:
                        speaker = "unknown"

                    utterances.append({
                        "sentence_id": sentence_id,
                        "start": start,
                        "end": end,
                        "text": text,
                        "speaker": speaker,
                        "emotion_label": "neutral",
                        "valence": 2.5,
                        "arousal": 2.5,
                        "dominance": 2.5,
                    })

        return utterances

    def _print_jarvis_output(
        self, response, observation, window_num: int
    ):
        """格式化打印 Jarvis 输出"""
        output = (
            f"\n{'='*70}\n"
            f"  Window #{window_num:02d}"
            f"\n"
            f"{'='*70}\n"
            f"  User: {observation.user_speech}\n"
            f"  Partner: {observation.partner_speech}\n"
            f"{'-'*70}\n"
            f"  JARVIS ({response.generated_by}):\n\n"
            f"  {response.analysis}\n"
            f"{'-'*70}\n"
            f"  {'WARNING: ' + response.risk_alert if response.risk_alert else ''}\n"
        )
        try:
            print(output)
        except UnicodeEncodeError:
            # 最后的兜底
            print(output.encode('ascii', errors='replace').decode('ascii'))

    def _print_session_summary(self, summary, stats: Dict):
        """打印会话总结"""
        output = (
            f"\n{'='*70}\n"
            f"  SESSION SUMMARY\n"
            f"{'='*70}\n"
            f"  Duration: {stats.get('duration_seconds', 0)/60:.1f} min"
            f" | Turns: {stats.get('total_turns', 0)}"
            f"{'='*70}\n"
            f"  Summary:\n"
            f"  {summary.summary[:500] if summary else 'N/A'}\n"
            f"\n"
            f"  Key Insights:\n"
            f"  {summary.key_insights[:400] if summary else 'N/A'}\n"
            f"\n"
            f"  Partner Profile:\n"
            f"  {summary.partner_profile[:300] if summary else 'N/A'}\n"
            f"{'='*70}\n"
            f"\n"
            f"  Session summary saved to: output/summaries/\n"
            f"  Full conversation history: output/conversation_history.db\n"
        )
        try:
            print(output)
        except UnicodeEncodeError:
            print(output.encode('ascii', errors='replace').decode('ascii'))

    def _infer_strengths(self, responses) -> str:
        """从响应中推断用户优势"""
        if not responses:
            return "not yet assessed"
        positive_patterns = ["good", "effective", "well", "right"]
        return "Engages in direct communication"

    def _infer_weaknesses(self, responses) -> str:
        """从响应中推断用户弱点"""
        if not responses:
            return "not yet assessed"
        return "May escalate under pressure — consider emotion regulation"


# ================================================================
# CLI 入口
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Jarvis — Multimodal Intelligent Response System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo
  python main.py --session 1 --dialog Ses01F_impro01 --data ../Session1/Session1
  python main.py --session 1 --batch --data ../Session1/Session1
  python main.py --session 1 --dialog Ses01F_impro01 --use-transcription
  python main.py --session 1 --dialog Ses01F_impro01 --use-local
        """,
    )
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--session", type=int, default=DEFAULT_SESSION)
    parser.add_argument("--dialog", type=str, default=DEFAULT_DIALOG)
    parser.add_argument("--data", type=str, default=DATA_ROOT)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--use-transcription", action="store_true")
    parser.add_argument("--use-local", action="store_true", default=USE_LOCAL)
    parser.add_argument("--video", type=str, default=DEFAULT_VIDEO,
                       help="Direct .avi/.mp4 path")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-audio", action="store_true")

    args = parser.parse_args()

    pipeline = JarvisPipeline(config_path=args.config)
    # 切换到软标签模型目录
    pipeline.config["audio"]["emotion_model_dir"] = "finetuned_emotion"  # soft待训练

    # ---- 路由 ----
    if args.demo:
        pipeline.run_demo(use_local=args.use_local)
        return

    # --video 模式: 显式传了 --video 参数就用它
    if args.video and os.path.exists(args.video):
        pipeline.initialize()
        pipeline.run_on_video(args.video, use_local=args.use_local)
        return

    # 默认: IEMOCAP 模式
    pipeline.initialize()

    if args.batch:
        # 批量处理 Session 中所有对话
        base = args.data.replace("Session1", f"Session{args.session}")
        avi_dir = os.path.join(base, "dialog", "avi", "DivX")
        if not os.path.exists(avi_dir):
            print(f"ERROR: Video directory not found: {avi_dir}")
            sys.exit(1)

        import glob
        video_files = sorted(glob.glob(os.path.join(avi_dir, "*.avi")))
        # 去重 (同一个对话有 F 和 M 两个视频)
        dialog_names = sorted(set(
            os.path.splitext(os.path.basename(v))[0].replace("F", "X").replace("M", "X")
            for v in video_files
        ))

        for dname in dialog_names:
            # 用原始的 F 版本
            actual_name = dname.replace("X", "F")
            try:
                pipeline.run_on_iemocap_dialog(
                    session_id=args.session,
                    dialog_name=actual_name,
                    data_base_path=args.data,
                    use_transcription=args.use_transcription,
                    use_local=args.use_local,
                )
            except Exception as e:
                logger.error(f"Failed on {actual_name}: {e}")
                continue
    else:
        pipeline.run_on_iemocap_dialog(
            session_id=args.session,
            dialog_name=args.dialog,
            data_base_path=args.data,
            use_transcription=args.use_transcription,
            use_local=args.use_local,
        )


if __name__ == "__main__":
    main()