#!/usr/bin/env python3
"""
实时视频人脸情绪播放器
======================
- YOLO 检测人脸 + ViT 识别情绪
- 实时画框显示
- 键盘控制播放速度

用法: python play_face_emotion.py [视频路径]
默认: Ses01F_impro01.avi

控制:
  ↑  / →  加速 (1x→2x→4x→8x)
  ↓  / ←  减速 (8x→4x→2x→1x)
  空格     暂停/播放
  q / ESC  退出
"""
import os, sys, cv2, numpy as np
import torch
torch.backends.cudnn.enabled = False
from PIL import Image, ImageDraw, ImageFont
from collections import deque
import time

# ---- 配置 ----
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "models")
DEFAULT_VIDEO = os.path.join(os.path.dirname(PROJECT_DIR), "Session1", "Session1",
                             "dialog", "avi", "DivX", "Ses01F_impro01.avi")
video_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO

# 颜色
EMO_COLORS = {
    'angry': (0, 0, 220), 'disgust': (0, 140, 0), 'fear': (160, 0, 160),
    'happy': (0, 220, 220), 'neutral': (160, 160, 160), 'sad': (220, 80, 0),
    'surprised': (0, 220, 0), 'unclear': (180, 180, 180),
}

# ---- 加载模型 ----
print("Loading models...")
from ultralytics import YOLO
face_model = YOLO(os.path.join(MODEL_DIR, "yolov8n-face.pt"), verbose=False)
if torch.cuda.is_available():
    face_model.to('cuda')
print("✓ YOLO face model")

from transformers import pipeline
emo_pipe = pipeline('image-classification',
                    model=os.path.join(MODEL_DIR, 'vit-face-expression'),
                    device=0 if torch.cuda.is_available() else -1)
print("✓ ViT emotion model")

# ---- 视频 ----
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
mid_x = frame_w / 2
print(f"Video: {frame_w}x{frame_h}, {fps:.0f}fps, {total_frames} frames")

# 播放速度
speeds = [0.5, 1, 2, 4, 8, 16, 32]
speed_idx = 1  # 默认 1x
paused = False

# 情绪平滑 (最近5帧取众数)
emo_history = {'left': deque(maxlen=5), 'right': deque(maxlen=5)}

cv2.namedWindow('Face Emotion (q=quit, space=pause, up/down=speed)', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Face Emotion (q=quit, space=pause, up/down=speed)', 960, 640)

fn = 0
frame_times = deque(maxlen=30)
last_frame_time = time.time()

print("\nControls: ↑↓ speed | Space pause | q quit")
print(f"Speed: {speeds[speed_idx]}x")

while True:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
    ret, frame = cap.read()
    if not ret:
        fn = 0
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)

    # ---- YOLO 检测 ----
    yolo_r = face_model(frame_rgb, verbose=False)
    dets = []
    if yolo_r[0].boxes:
        for box in yolo_r[0].boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            conf = float(box.conf[0]) if box.conf is not None else 0.5
            if conf > 0.3 and x2 > x1 and y2 > y1:
                dets.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'conf': conf})

    # 左右各取最大
    left_dets = [d for d in dets if (d['x1'] + d['x2']) / 2 < mid_x]
    right_dets = [d for d in dets if (d['x1'] + d['x2']) / 2 >= mid_x]
    best = []
    if left_dets:
        best.append(max(left_dets, key=lambda d: (d['x2'] - d['x1']) * (d['y2'] - d['y1'])))
    if right_dets:
        best.append(max(right_dets, key=lambda d: (d['x2'] - d['x1']) * (d['y2'] - d['y1'])))

    # ---- 情绪识别 + 画框 ----
    for d in best:
        x1, y1, x2, y2 = d['x1'], d['y1'], d['x2'], d['y2']
        face = frame_rgb[y1:y2, x1:x2]
        if face.size == 0: continue

        try:
            face_pil = Image.fromarray(cv2.resize(face, (224, 224)))
            result = emo_pipe(face_pil)
            emo = result[0]['label']
            score = result[0]['score']
        except:
            emo, score = 'neutral', 0.0

        side = "left" if (x1 + x2) / 2 < mid_x else "right"
        emo_history[side].append(emo)
        # 平滑后的情绪
        from collections import Counter
        smooth_emo = Counter(emo_history[side]).most_common(1)[0][0] if emo_history[side] else emo

        color = EMO_COLORS.get(smooth_emo, (255, 255, 255))
        label = f"{'L' if side=='left' else 'R'} {smooth_emo} {score:.2f}"

        # 画框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        # 标签背景
        tw = len(label) * 9
        draw.rectangle([x1, y1 - 22, x1 + tw, y1], fill=color)
        draw.text((x1 + 3, y1 - 20), label, fill=(0, 0, 0))

    # 中线
    draw.line([(mid_x, 0), (mid_x, frame_h)], fill=(0, 255, 0), width=1)

    # 顶部信息栏
    ts = fn / fps
    info = f"Frame {fn} | {ts:.1f}s | Speed {speeds[speed_idx]}x"
    if paused:
        info += " | PAUSED"
    draw.rectangle([0, 0, frame_w, 28], fill=(0, 0, 0, 180))
    draw.text((10, 5), info, fill=(255, 255, 255))

    # 帧率
    now = time.time()
    frame_times.append(now - last_frame_time)
    last_frame_time = now
    if frame_times:
        real_fps = len(frame_times) / sum(frame_times)
        draw.text((frame_w - 120, 5), f"FPS: {real_fps:.0f}", fill=(0, 255, 0))

    # ---- 显示 ----
    frame_out = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cv2.imshow('Face Emotion (q=quit, space=pause, up/down=speed)', frame_out)

    # ---- 键盘控制 ----
    delay = int(1000 / (fps * speeds[speed_idx]))
    key = cv2.waitKey(delay if not paused else 0) & 0xFF

    if key == ord('q') or key == 27:  # q or ESC
        break
    elif key == ord(' '):  # space
        paused = not paused
        print("⏸ Paused" if paused else "▶ Playing")
    elif key == ord('w') or key == 2490368:  # w or ↑
        if speed_idx < len(speeds) - 1:
            speed_idx += 1
            print(f"Speed: {speeds[speed_idx]}x")
    elif key == ord('s') or key == 2621440:  # s or ↓
        if speed_idx > 0:
            speed_idx -= 1
            print(f"Speed: {speeds[speed_idx]}x")

    # 正常播放时前进, pause 时不动
    if not paused:
        fn += 1
    if fn >= total_frames:
        fn = 0

cap.release()
cv2.destroyAllWindows()
print("Done.")
