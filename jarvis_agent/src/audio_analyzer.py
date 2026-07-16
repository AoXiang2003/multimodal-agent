"""
音频分析模块 — ASR转写 + 声纹分离 + 语音情感识别
===================================================
职责:
  1. ASR转写: Whisper 语音→文字
  2. 声纹分离: pyannote 区分说话人 (音色聚类)
  3. 语音情感: wav2vec2 从语调判断情绪 (取代面部几何规则)
"""

import os
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import time

logger = logging.getLogger(__name__)

# --- ASR 引擎 ---
try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    HAS_FASTER_WHISPER = False

try:
    import whisper
    HAS_OPENAI_WHISPER = True
except ImportError:
    HAS_OPENAI_WHISPER = False

# --- 说话人分离 ---
# SpeakerDiarizer 已删除, lazy import for backward compat
try:
    from src.speaker_diarizer import SpeakerDiarizer, create_diarizer, SpeakerSegment
except ImportError:
    SpeakerDiarizer = None
    create_diarizer = None
    SpeakerSegment = None


@dataclass
class SpeechSegment:
    """单个语音片段 (带说话人标注)"""
    start_time: float
    end_time: float
    text: str = ""                     # ASR 转写文本
    speaker_id: str = "unknown"        # 声纹识别的说话人 ID
    speaker_gender: str = "unknown"    # 推断的性别 (male/female)
    confidence: float = 0.0            # ASR 置信度
    language: str = "en"


@dataclass
class AudioAnalysisResult:
    """完整音频分析结果"""
    dialog_name: str
    duration: float
    full_transcript: str                           # 按说话人和时间排序的完整对话
    segments: List[SpeechSegment] = field(default_factory=list)
    word_count: int = 0
    avg_confidence: float = 0.0
    # 说话人信息
    speaker_ids: List[str] = field(default_factory=list)   # 所有说话人 ID
    speaker_genders: Dict[str, str] = field(default_factory=dict)
    # 统计
    speech_rate: float = 0.0
    voice_emotion: str = "neutral"
    voice_valence: float = 2.5
    voice_arousal: float = 2.5

    def get_speaker_text(self, speaker_id: str) -> str:
        """获取某个说话人的全部文本"""
        return " ".join(
            s.text for s in self.segments if s.speaker_id == speaker_id
        )

    def get_speaker_segments(
        self, speaker_id: str, start_time: float = 0.0, end_time: Optional[float] = None
    ) -> List[SpeechSegment]:
        """获取某说话人在时间范围内的片段"""
        result = [s for s in self.segments if s.speaker_id == speaker_id]
        if end_time is not None:
            result = [
                s for s in result
                if s.start_time >= start_time and s.end_time <= end_time
            ]
        return result


class AudioAnalyzer:
    """
    音频分析器 — ASR + 声纹分离

    去耦合设计:
      - ASR引擎: faster-whisper / openai-whisper (可替换)
      - 说话人分离: resemblyzer / pyannote / librosa (可替换)
      - 两个模块独立运行, 通过时间戳对齐融合

    你只需要: 一个 .wav 文件
    你不用: 任何标签数据、transcription文件、说话人标注

    模型全部预训练, 零微调。
    """

    def __init__(self, config: Dict):
        self.config = config.get("audio", {})
        self.model_size = self.config.get("asr_model", "tiny")
        self.asr_engine = self.config.get("asr_engine", "faster_whisper")
        self.sample_rate = self.config.get("sample_rate", 16000)
        self.language = self.config.get("language", "en")

        # ---- 加载 ASR 模型 ----
        self.asr_model = None
        if self.asr_engine == "faster_whisper":
            if not HAS_FASTER_WHISPER:
                logger.warning("faster-whisper not available. Install: pip install faster-whisper")
                self.asr_engine = "openai_whisper"
            else:
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    compute = "float16" if device == "cuda" else "int8"
                    # 优先本地模型
                    _whisper_local = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        "models", "faster-whisper-small")
                    _whisper_path = _whisper_local if os.path.exists(_whisper_local) else self.model_size
                    self.asr_model = WhisperModel(
                        _whisper_path, device=device,
                        compute_type=compute, num_workers=2)
                    logger.info(f"✓ faster-whisper: {self.model_size} ({device}/{compute})")
                except Exception as e:
                    logger.error(f"faster-whisper failed: {e}")
                    self.asr_engine = "openai_whisper"

        if self.asr_engine == "openai_whisper":
            if not HAS_OPENAI_WHISPER:
                raise ImportError("Install: pip install openai-whisper")
            self.asr_model = whisper.load_model(self.model_size)
            logger.info(f"✓ openai-whisper loaded: {self.model_size}")

        # ---- 声纹分离器 (延迟加载) ----
        self.diarization_method = self.config.get("diarization_method", "resemblyzer")
        self.hf_token = os.environ.get("HF_TOKEN", "") or self.config.get("hf_token", "")
        self.diarizer: Optional[SpeakerDiarizer] = None

        # ---- 加载语音情感识别模型 (wav2vec2-base, IEMOCAP微调可选) ----
        self.emotion_model = None
        self.emotion_extractor = None
        try:
            from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
            import torch
            # 优先用本地微调模型, 其次 IEMOCAP large, 最后通用
            # 优先加载 best 模型, 其次 final
            model_dir_name = self.config.get("emotion_model_dir", "finetuned_emotion")
            emo_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                   "output", model_dir_name)
            local_ft = os.path.join(emo_dir, "wav2vec2_iemocap_best.pt")
            if not os.path.exists(local_ft):
                local_ft = os.path.join(emo_dir, "wav2vec2_iemocap_finetuned.pt")
            if os.path.exists(local_ft):
                logger.info("Loading local finetuned model...")
                # 加载本地 wav2vec2-base (项目 models/ 目录)
                _models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
                _w2v2_path = os.path.join(_models_dir, "wav2vec2-base")
                if os.path.exists(_w2v2_path):
                    self.emotion_extractor = AutoFeatureExtractor.from_pretrained(_w2v2_path)
                else:
                    self.emotion_extractor = AutoFeatureExtractor.from_pretrained(
                        "facebook/wav2vec2-base")  # fallback
                import sys
                _train_dir = os.path.dirname(os.path.dirname(__file__))
                if _train_dir not in sys.path:
                    sys.path.insert(0, _train_dir)
                from train_emotion import Wav2Vec2ForIEMOCAP
                from train_emotion import LABEL_LIST, label2id, id2label
                self.emotion_model = Wav2Vec2ForIEMOCAP(num_labels=len(LABEL_LIST))
                ckpt = torch.load(local_ft, map_location='cpu', weights_only=False)
                self.emotion_model.load_state_dict(ckpt['model_state_dict'])
                if torch.cuda.is_available():
                    self.emotion_model = self.emotion_model.cuda()
                self.emotion_labels = id2label
                logger.info(f"✓ Local finetuned model loaded (device={'cuda' if torch.cuda.is_available() else 'cpu'})")
            else:
                emo_model_name = self.config.get("emotion_model",
                    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
                self.emotion_extractor = AutoFeatureExtractor.from_pretrained(emo_model_name)
                self.emotion_model = AutoModelForAudioClassification.from_pretrained(emo_model_name)
                if torch.cuda.is_available():
                    self.emotion_model = self.emotion_model.cuda()
                self.emotion_labels = self.emotion_model.config.id2label
                logger.info(f"✓ Speech emotion model: {emo_model_name} (device={'cuda' if torch.cuda.is_available() else 'cpu'})")
        except Exception as e:
            logger.warning(f"Speech emotion model not available: {e}")

        logger.info(
            f"AudioAnalyzer ready: ASR={self.asr_engine}/{self.model_size}, "
            f"Diarization={'enabled' if self.diarizer else 'disabled'}, "
            f"Emotion={'enabled' if self.emotion_model else 'disabled'}"
        )

    # ================================================================
    # 主入口: ASR + 声纹分离 → 带说话人标注的完整结果
    # ================================================================

    def transcribe_with_speakers(
        self,
        audio_path: str,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
    ) -> AudioAnalysisResult:
        """
        转写音频并标注每个片段的说话人 (基于声纹)

        这是推荐的主入口, 同时完成 ASR 和说话人分离。

        Args:
            audio_path: 音频文件路径
            start_time: 分析起始时间
            end_time: 分析结束时间

        Returns:
            AudioAnalysisResult (每个 segment 带 speaker_id + text + gender)
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        logger.info(f"Processing: {audio_path}")

        t_start = time.time()

        # ---- Step 1: ASR 转写 (带时间戳) ----
        asr_segments = self._run_asr(audio_path)

        # ---- Step 2: 声纹说话人分离 ----
        if self.diarizer is not None:
            try:
                diar_segments = self.diarizer.diarize(
                    audio_path, sample_rate=self.sample_rate
                )
                logger.info(
                    f"Diarization: {len(diar_segments)} voice segments, "
                    f"{len(set(s.speaker_id for s in diar_segments))} speakers"
                )
            except Exception as e:
                logger.warning(f"Diarization failed: {e}, using fallback")
                diar_segments = None
        else:
            diar_segments = None

        # ---- Step 3: 时间戳对齐 ----
        # 将 ASR 文本段映射到声纹分离的说话人
        annotated_segments = self._align_asr_to_speakers(
            asr_segments, diar_segments
        )

        # ---- Step 4: 构建结果 ----
        speaker_ids = list(set(s.speaker_id for s in annotated_segments))
        speaker_genders = {}
        for s in annotated_segments:
            if s.speaker_id not in speaker_genders and s.speaker_gender != "unknown":
                speaker_genders[s.speaker_id] = s.speaker_gender

        # 构建按时间和说话人组织的完整转写
        transcript_lines = []
        for seg in annotated_segments:
            gender_tag = f"({seg.speaker_gender[0]})" if seg.speaker_gender != "unknown" else ""
            transcript_lines.append(
                f"[{seg.start_time:.1f}s-{seg.end_time:.1f}s] "
                f"{seg.speaker_id}{gender_tag}: {seg.text}"
            )
        full_transcript = "\n".join(transcript_lines)

        duration = (
            annotated_segments[-1].end_time if annotated_segments
            else asr_segments[-1].end_time if asr_segments
            else 0.0
        )

        word_count = sum(len(s.text.split()) for s in annotated_segments)
        avg_conf = np.mean([s.confidence for s in annotated_segments]) if annotated_segments else 0.0

        result = AudioAnalysisResult(
            dialog_name=audio_path,
            duration=duration,
            full_transcript=full_transcript,
            segments=annotated_segments,
            word_count=word_count,
            avg_confidence=avg_conf,
            speaker_ids=speaker_ids,
            speaker_genders=speaker_genders,
            speech_rate=word_count / duration if duration > 0 else 0.0,
        )
        # 保存原始声纹分离结果 (供主流水线使用)
        result._diar_segments = diar_segments

        elapsed = time.time() - t_start
        logger.info(
            f"Audio analysis complete: {word_count} words, "
            f"{len(annotated_segments)} annotated segments, "
            f"speakers={speaker_ids}, genders={speaker_genders}, "
            f"{elapsed:.1f}s"
        )

        return result

    def transcribe_segment(
        self,
        audio_path: str,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
    ) -> AudioAnalysisResult:
        """兼容旧接口 — 内部调用 transcribe_with_speakers"""
        return self.transcribe_with_speakers(audio_path, start_time, end_time)

    # ================================================================
    # ASR 引擎
    # ================================================================

    def _run_asr(self, audio_path: str) -> List[SpeechSegment]:
        """
        运行 ASR 获取带时间戳的文本段

        Returns:
            [SpeechSegment(start=0.5, end=2.1, text="Excuse me.", speaker_id="unknown"), ...]
        """
        if self.asr_engine == "faster_whisper":
            return self._asr_faster_whisper(audio_path)
        else:
            return self._asr_openai_whisper(audio_path)

    def _asr_faster_whisper(self, audio_path: str) -> List[SpeechSegment]:
        segments = []
        raw_segments, info = self.asr_model.transcribe(
            audio_path,
            language=self.language,
            beam_size=5,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 150,    # 150ms 静音就切句
                "speech_pad_ms": 30,
                "min_speech_duration_ms": 100,
            },
            word_timestamps=True,                   # 词级时间戳!
        )
        for seg in raw_segments:
            # 保留词级信息 (用于嘴部运动精确切分)
            words_with_ts = []
            if hasattr(seg, 'words') and seg.words:
                words_with_ts = [
                    {"word": w.word, "start": w.start, "end": w.end,
                     "prob": getattr(w, 'probability', 0.8)}
                    for w in seg.words
                ]
            segments.append(SpeechSegment(
                start_time=seg.start,
                end_time=seg.end,
                text=seg.text.strip(),
                confidence=getattr(seg, 'avg_logprob', 0.8),
                language=info.language,
            ))
            # 把词级信息暂时挂在 segment 上
            segments[-1]._words = words_with_ts
        return segments

    def _asr_openai_whisper(self, audio_path: str) -> List[SpeechSegment]:
        result = self.asr_model.transcribe(audio_path, language=self.language, verbose=False)
        segments = []
        for seg in result.get("segments", []):
            segments.append(SpeechSegment(
                start_time=seg["start"],
                end_time=seg["end"],
                text=seg["text"].strip(),
                confidence=seg.get("confidence", 0.8),
                language=self.language,
            ))
        return segments

    # ================================================================
    # 时间戳对齐: ASR文本段 → 声纹说话人
    # ================================================================

    def _align_asr_to_speakers(
        self,
        asr_segments: List[SpeechSegment],
        diar_segments: Optional[List[SpeakerSegment]],
    ) -> List[SpeechSegment]:
        """
        将 ASR 文本段与声纹分离结果进行时间戳对齐

        对于每个 ASR 片段, 找到时间上重叠最多的声纹片段,
        继承其 speaker_id 和 gender。

        如果没有声纹分离结果, 退化为奇偶交替分配。
        """
        if diar_segments is None or len(diar_segments) == 0:
            return self._fallback_speaker_assignment(asr_segments)

        annotated = []

        for asr_seg in asr_segments:
            best_overlap = 0.0
            best_speaker = "SPK_0"
            best_gender = "unknown"

            # 找时间重叠最大的声纹片段
            for diar_seg in diar_segments:
                overlap = self._compute_overlap(
                    asr_seg.start_time, asr_seg.end_time,
                    diar_seg.start_time, diar_seg.end_time,
                )
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = diar_seg.speaker_id
                    best_gender = diar_seg.speaker_gender

            asr_duration = asr_seg.end_time - asr_seg.start_time
            # 如果重叠超过 50%, 认为是匹配的
            if asr_duration > 0 and best_overlap / asr_duration > 0.3:
                asr_seg.speaker_id = best_speaker
                asr_seg.speaker_gender = best_gender
            else:
                # 低重叠 → 保持 unknown
                asr_seg.speaker_id = "SPK_?"

            annotated.append(asr_seg)

        # 后处理: 如果某些短片段标记为 unknown, 用最近的已知片段填充
        annotated = self._fill_unknown_speakers(annotated)

        return annotated

    def _fallback_speaker_assignment(
        self, segments: List[SpeechSegment]
    ) -> List[SpeechSegment]:
        """
        退化为奇偶交替分配 (当声纹分离不可用时)

        这比纯顺序交替更鲁棒: 用时间间隔辅助判断
        间隔 >2s 大概率是同一人继续, <2s 大概率交替
        """
        if not segments:
            return segments

        current_speaker = "SPK_0"
        segments[0].speaker_id = current_speaker

        for i in range(1, len(segments)):
            gap = segments[i].start_time - segments[i-1].end_time
            if gap > 2.0:
                # 间隔长, 可能是同一个人继续
                pass  # 保持不变
            else:
                # 间隔短, 大概率是交替
                current_speaker = "SPK_1" if current_speaker == "SPK_0" else "SPK_0"
            segments[i].speaker_id = current_speaker

        return segments

    def _fill_unknown_speakers(
        self, segments: List[SpeechSegment]
    ) -> List[SpeechSegment]:
        """用前后已知说话人填充 unknown 片段"""
        # 前向填充
        last_known = None
        for seg in segments:
            if seg.speaker_id != "SPK_?":
                last_known = (seg.speaker_id, seg.speaker_gender)
            elif last_known is not None:
                seg.speaker_id, seg.speaker_gender = last_known

        # 反向填充
        last_known = None
        for seg in reversed(segments):
            if seg.speaker_id != "SPK_?":
                last_known = (seg.speaker_id, seg.speaker_gender)
            elif last_known is not None:
                seg.speaker_id, seg.speaker_gender = last_known

        return segments

    @staticmethod
    def _compute_overlap(
        a_start: float, a_end: float,
        b_start: float, b_end: float,
    ) -> float:
        """计算两个时间段的重叠长度"""
        overlap_start = max(a_start, b_start)
        overlap_end = min(a_end, b_end)
        return max(0.0, overlap_end - overlap_start)

    # ================================================================
    # 便捷方法
    # ================================================================

    def transcribe_sentence_wavs(
        self, wav_dir: str, sentence_ids: List[str],
    ) -> Dict[str, AudioAnalysisResult]:
        """批量转写句子级 wav (每个句子是单人语音, 不需要分离)"""
        results = {}
        for sid in sentence_ids:
            wav_path = os.path.join(wav_dir, f"{sid}.wav")
            if os.path.exists(wav_path):
                results[sid] = self.transcribe_with_speakers(wav_path)
        return results

    def predict_emotion(self, audio_wav: np.ndarray, sample_rate: int = 16000) -> Dict:
        """
        从语音波形预测情绪 (wav2vec2模型)

        Args:
            audio_wav: 1D numpy array, 语音波形
            sample_rate: 采样率

        Returns:
            {"label": "angry", "score": 0.87, "all": {...}}
        """
        if self.emotion_model is None:
            return {"label": "neutral", "score": 0.0, "all": {}}

        try:
            import torch
            # 取最多5秒
            max_samples = sample_rate * 5
            if len(audio_wav) > max_samples:
                audio_wav = audio_wav[:max_samples]
            if len(audio_wav) < sample_rate * 0.02:  # <20ms 太短
                return {"label": "neutral", "score": 0.0, "all": {}}

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            inputs = self.emotion_extractor(
                audio_wav, sampling_rate=sample_rate, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.emotion_model(**inputs)
            # 兼容 dict 和 HuggingFace 输出
            logits = outputs.get("logits") if isinstance(outputs, dict) else outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]
            top_idx = torch.argmax(probs).item()

            all_emotions = {}
            for i in range(len(probs)):
                label = self.emotion_labels.get(i, f'class_{i}')
                all_emotions[label] = float(probs[i].item())

            # 标签映射: wav2vec2标签 → 我们的8类情感
            label_map = {
                "ang": "angry", "anger": "angry", "angry": "angry",
                "hap": "happy", "happy": "happy", "happiness": "happy",
                "sad": "sad", "sadness": "sad",
                "neu": "neutral", "neutral": "neutral",
                "fea": "fear", "fear": "fear", "fearful": "fear",
                "dis": "disgust", "disgust": "disgust",
                "sur": "surprised", "surprise": "surprised", "surprised": "surprised",
                "fru": "frustrated", "frustrated": "frustrated",
                "exc": "excited", "excited": "excited",
                "calm": "neutral",  # IEMOCAP model
            }
            raw_label = self.emotion_labels.get(top_idx, "neutral")
            mapped_label = label_map.get(raw_label, raw_label)

            return {
                "label": mapped_label,
                "score": float(probs[top_idx].item()),
                "all": all_emotions,
            }
        except Exception as e:
            logger.warning(f"Emotion prediction failed: {e}")
            return {"label": "neutral", "score": 0.0, "all": {}}

    def get_transcript_summary(self, result: AudioAnalysisResult) -> Dict:
        """生成摘要, 供 Fusion Engine 使用"""
        # 按说话人分组
        speaker_texts = {}
        for sid in result.speaker_ids:
            speaker_texts[sid] = result.get_speaker_text(sid)

        return {
            "full_transcript": result.full_transcript,
            "word_count": result.word_count,
            "speech_rate": result.speech_rate,
            "avg_confidence": result.avg_confidence,
            "segment_count": len(result.segments),
            "speaker_ids": result.speaker_ids,
            "speaker_genders": result.speaker_genders,
            "speaker_texts": speaker_texts,
            "last_utterance": (
                result.segments[-1].text if result.segments else ""
            ),
        }
