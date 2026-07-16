#!/usr/bin/env python3
"""
视频通话模拟器 — 实时人脸+语音情绪 + 对话记录 + JARVIS场景分析
============================================================
用法: python simulate_video_call.py [视频路径] [--analysis-every N]

左侧: 视频画面 (YOLO人脸框+情绪)
右侧: 对话记录 (逐句显示, 带语音+人脸情绪) + JARVIS分析

控制:
  W / ↑  加速    S / ↓  减速    空格  暂停    Q  退出
  F       跳转到下一个分析窗口
  R       重播
"""
import os, sys, cv2, numpy as np, time
import torch
torch.backends.cudnn.enabled = False
from PIL import Image, ImageDraw
from collections import deque, Counter
import argparse

# ---- 配置 ----
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "models")

parser = argparse.ArgumentParser()
parser.add_argument("video", nargs="?", help="Video path")
parser.add_argument("--analysis-every", type=int, default=14, help="Utterances per JARVIS window")
args = parser.parse_args()

ANALYSIS_EVERY = args.analysis_every
DEFAULT_VIDEO = os.path.join(os.path.dirname(PROJECT_DIR), "Session1", "Session1",
                             "dialog", "avi", "DivX", "Ses01M_impro03.avi")
video_path = args.video or DEFAULT_VIDEO

EMO_COLORS = {
    'angry': (0,0,220), 'disgust': (0,140,0), 'fear': (160,0,160),
    'happy': (0,220,220), 'neutral': (160,160,160), 'sad': (220,80,0),
    'surprised': (0,220,0), 'excited': (0,255,128), 'frustrated': (255,100,0),
    'unclear': (180,180,180),
}

# ---- 音色标记 ----
VOICE_MARKER = "🎤"  # 语音延迟提示

# ================================================================
# Step 1: 预计算全部数据 (同 main_hard 流程)
# ================================================================
print("=" * 60)
print("Pre-computing transcript + emotions + JARVIS analysis...")
print("=" * 60)

sys.path.insert(0, PROJECT_DIR)
from utils.helpers import load_config, setup_logging
import logging as _logging
_logging.basicConfig(level=_logging.WARNING)  # suppress noise

config = load_config(os.path.join(PROJECT_DIR, "config.yaml"))

# 加载 IEMOCAP transcription
dialog_name = os.path.splitext(os.path.basename(video_path))[0]
from main_hard import JarvisPipeline  # reuse the pipeline
pipeline = JarvisPipeline()
pipeline.initialize()

# 匹配 transcription
audio_tmp = os.path.join(PROJECT_DIR, "output", f"{dialog_name}_audio.wav")
from src.audio_analyzer import SpeechSegment, AudioAnalysisResult
import re

transcription_path = os.path.join(os.path.dirname(video_path), "..", "..",
                                  "transcriptions", f"{dialog_name}.txt")
transcription_path = os.path.normpath(transcription_path)

if not os.path.exists(transcription_path):
    print(f"ERROR: Transcription not found at {transcription_path}")
    sys.exit(1)

utterances = pipeline._parse_transcription(transcription_path)
print(f"✓ {len(utterances)} utterances loaded")

# 构建 segments
segments = [SpeechSegment(start_time=u["start"], end_time=u["end"],
            text=u["text"], speaker_id="SPK_0" if u["speaker"]=="F" else "SPK_1")
            for u in utterances]

# 给每个 segment 补上语音+人脸情绪
from src.face_emotion import FaceEmotionAnalyzer
fea = FaceEmotionAnalyzer(dialog_name=dialog_name)
utt_list = [{"start": s.start_time, "end": s.end_time} for s in segments]
# 每句采样2帧 (降计算量)
for ut in utt_list:
    ut["_emo"] = fea.analyze_utterance(video_path, ut["start"], ut["end"], num_samples=3)
face_results = [u["_emo"] for u in utt_list]
for seg, fr in zip(segments, face_results):
    seg._face_left = fr.get("left_emotion", "neutral")
    seg._face_right = fr.get("right_emotion", "neutral")

# 语音情绪: 每句预测
if os.path.exists(audio_tmp):
    import librosa
    for seg in segments:
        try:
            dur = max(seg.end_time - seg.start_time, 0.3)
            wav, sr = librosa.load(audio_tmp, sr=16000,
                                   offset=max(0, seg.start_time-0.1),
                                   duration=dur+0.2)
            if len(wav) > sr * 0.05 and pipeline.audio_analyzer:
                seg._voice_emo = pipeline.audio_analyzer.predict_emotion(wav, sr)["label"]
            else:
                seg._voice_emo = "?"
        except:
            seg._voice_emo = "?"
else:
    for seg in segments:
        seg._voice_emo = "?"

# 生成 JARVIS 分析窗口
print("Generating JARVIS analysis...")
jarvis_analyses = []
for i in range(0, len(segments), ANALYSIS_EVERY):
    batch = segments[i:i + ANALYSIS_EVERY]
    # Build observation
    user_parts, partner_parts = [], []
    for seg in batch:
        v = getattr(seg, '_voice_emo', '?')
        f = getattr(seg, '_face_left', '?') if seg.speaker_id == "SPK_0" else getattr(seg, '_face_right', '?')
        f = f if f else "?"
        spk = "You" if seg.speaker_id == "SPK_0" else "Partner"
        user_parts.append(f"[v:{v} f:{f}] {seg.text}") if seg.speaker_id == "SPK_0" else partner_parts.append(f"[v:{v} f:{f}] {seg.text}")

    from src.jarvis_agent import MultimodalObservation
    obs = MultimodalObservation(
        user_speech=" | ".join(user_parts) if user_parts else "(silence)",
        partner_speech=" | ".join(partner_parts) if partner_parts else "(silence)",
        user_emotion="?", partner_emotion="?",
    )
    try:
        resp = pipeline.jarvis_agent.analyze_and_respond(
            observation=obs, session_id=dialog_name, use_local=False)
        jarvis_analyses.append((batch[-1].end_time, resp.analysis))
        print(f"  Window {len(jarvis_analyses)}: [{batch[0].start_time:.1f}s - {batch[-1].end_time:.1f}s] OK")
    except Exception as e:
        print(f"  Window {len(jarvis_analyses)+1}: ERROR {e}")

print(f"\n✓ Pre-computation complete: {len(segments)} utterances, {len(jarvis_analyses)} analyses")
print("=" * 60)

# ================================================================
# Step 2: 视频播放器 (性能优化版 — 缓存面板 + 时间索引)
# ================================================================
from ultralytics import YOLO
face_model = YOLO(os.path.join(MODEL_DIR, "yolov8n-face.pt"), verbose=False)
if torch.cuda.is_available():
    face_model.to('cuda')

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
mid_x = frame_w / 2
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

speeds = [0.5, 1, 2, 4, 8, 16, 32]
speed_idx = 1
paused = False
emo_history = {'left': deque(maxlen=5), 'right': deque(maxlen=5)}
PANEL_W = 520
total_w = frame_w + PANEL_W

cv2.namedWindow('Video Call Simulator', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Video Call Simulator', total_w, frame_h)

fn = 0
frame_times = deque(maxlen=30)
last_frame_time = time.time()
analysis_idx = 0

# ==== 性能优化: 预计算时间索引 + 面板缓存 ====
_time2emo = {}
for seg in segments:
    for t in range(int(seg.start_time), int(seg.end_time) + 1):
        le = getattr(seg, '_face_left', 'neutral') or 'neutral'
        re = getattr(seg, '_face_right', 'neutral') or 'neutral'
        _time2emo[t] = (le, re)

_cached_panel = None
_cached_vcount = -1
_cached_aidx = -1

def _build_panel(ct):
    p = Image.fromarray(np.ones((frame_h, PANEL_W, 3), dtype=np.uint8) * 245)
    d = ImageDraw.Draw(p)
    y = 8
    vs = [s for s in segments if s.start_time <= ct + 0.3]
    ss = vs[-20:] if len(vs) > 20 else vs
    d.text((4, y), f"--- DIALOGUE ({len(vs)}/{len(segments)}) ---", fill=(0,0,0))
    y += 15
    for seg in ss[-18:]:
        spk = "You" if seg.speaker_id == "SPK_0" else "Ptn"
        v = getattr(seg, '_voice_emo', '?')
        f = getattr(seg, '_face_left', '?') if seg.speaker_id == "SPK_0" else getattr(seg, '_face_right', '?')
        f = f if f else "?"
        vd = "?" if ct < seg.end_time else v
        d.text((4, y), f"{spk}[v:{vd} f:{f}]: {seg.text[:62]}", fill=(30,30,30))
        y += 14
        if y > frame_h - 220: break
    eff = None
    for i, (at, atx) in enumerate(jarvis_analyses):
        if ct >= at + 1.5: eff = (i, atx)
    if eff:
        d.line([(0, y+3), (PANEL_W, y+3)], fill=(0,0,0), width=1)
        y += 8
        d.text((4, y), f"--- JARVIS #{eff[0]+1} ---", fill=(0,0,180))
        y += 16
        for line in eff[1].split("\n")[:15]:
            if line.strip(): d.text((4, y), line.strip()[:80], fill=(20,20,20))
            y += 12
            if y > frame_h - 10: break
    return np.array(p), len(vs), eff[0] if eff else -1

print(f"\nControls: W/↑ speed | S/↓ speed | Space pause | F next analysis | Q quit")
print(f"Analysis every {ANALYSIS_EVERY} utterances | {len(jarvis_analyses)} windows")

while True:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
    ret, frame = cap.read()
    if not ret: fn = 0; continue
    ct = fn / fps
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    canvas = cv2.copyMakeBorder(frame_rgb, 0, 0, 0, PANEL_W, cv2.BORDER_CONSTANT, value=(240,240,240))

    if fn % 30 == 0 or not hasattr(face_model, '_cached_dets'):
        yr = face_model(frame_rgb, verbose=False)
        dets = []
        if yr[0].boxes:
            for box in yr[0].boxes:
                x1,y1,x2,y2 = [int(v) for v in box.xyxy[0]]
                conf = float(box.conf[0]) if box.conf is not None else 0.5
                if conf > 0.3: dets.append({'x1':x1,'y1':y1,'x2':x2,'y2':y2,'conf':conf})
        face_model._cached_dets = dets
    else:
        dets = face_model._cached_dets

    left_d = [d for d in dets if (d['x1']+d['x2'])/2 < mid_x]
    right_d = [d for d in dets if (d['x1']+d['x2'])/2 >= mid_x]
    best = []
    if left_d: best.append(max(left_d, key=lambda d: (d['x2']-d['x1'])*(d['y2']-d['y1'])))
    if right_d: best.append(max(right_d, key=lambda d: (d['x2']-d['x1'])*(d['y2']-d['y1'])))

    for d in best:
        x1,y1,x2,y2 = d['x1'],d['y1'],d['x2'],d['y2']
        side = "left" if (x1+x2)/2 < mid_x else "right"
        le, re = _time2emo.get(int(ct), ('neutral','neutral'))
        emo = le if side == 'left' else re
        emo_history[side].append(emo)
        smooth = Counter(emo_history[side]).most_common(1)[0][0] if emo_history[side] else "neutral"
        color = EMO_COLORS.get(smooth, (255,255,255))
        cv2.rectangle(canvas, (x1,y1), (x2,y2), color, 2)
        cv2.rectangle(canvas, (x1, y1-18), (x1+100, y1), color, -1)
        cv2.putText(canvas, f"{'L' if side=='left' else 'R'} {smooth}", (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

    cv2.line(canvas, (int(mid_x), 0), (int(mid_x), frame_h), (0,255,0), 1)

    vc = len([s for s in segments if s.start_time <= ct + 0.3])
    ai = -1
    for i, at in enumerate([t for t,_ in jarvis_analyses]):
        if ct >= at + 1.5: ai = i
    if _cached_panel is None or vc != _cached_vcount or ai != _cached_aidx or fn % 10 == 0:
        _cached_panel, _cached_vcount, _cached_aidx = _build_panel(ct)
    canvas[:, frame_w:] = _cached_panel

    info = f"T={ct:.1f}s | {speeds[speed_idx]}x | {vc}/{len(segments)} utt"
    cv2.putText(canvas, info, (8, frame_h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 2)
    cv2.putText(canvas, info, (8, frame_h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    cv2.imshow('Video Call Simulator', cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    delay = max(1, int(1000/fps/speeds[speed_idx]))
    key = cv2.waitKey(delay) & 0xFF
    if key == ord('q') or key == 27: break
    elif key == ord(' '): paused = not paused; print("⏸ Paused" if paused else "▶ Playing")
    elif key == ord('f') or key == ord('F'):
        if analysis_idx < len(jarvis_analyses):
            at, _ = jarvis_analyses[analysis_idx]; fn = max(0, int(at*fps)-10); analysis_idx += 1
            print(f"→ Window {analysis_idx}: {at:.1f}s")
    elif key == ord('w') or key in (82, 2490368):
        if speed_idx < len(speeds)-1: speed_idx += 1; print(f"Speed: {speeds[speed_idx]}x")
    elif key == ord('s') or key in (83, 2621440):
        if speed_idx > 0: speed_idx -= 1; print(f"Speed: {speeds[speed_idx]}x")

    if not paused: fn += 1
    now = time.time(); frame_times.append(now - last_frame_time); last_frame_time = now

cap.release()
cv2.destroyAllWindows()
print(f"\nDone: {fn} frames rendered")
