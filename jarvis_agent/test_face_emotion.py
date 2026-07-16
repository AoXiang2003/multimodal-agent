#!/usr/bin/env python3
"""
人脸检测+情绪标注 可视化测试
============================
用法: python test_face_emotion.py [视频路径]
默认: Ses01F_impro01.avi
输出: output/face_test/ 目录下带标注的帧图片
"""
import os, sys, cv2, numpy as np
import torch
torch.backends.cudnn.enabled = False
from PIL import Image, ImageDraw, ImageFont

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output", "face_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMOTION_COLORS = {
    'angry': (0, 0, 255), 'disgust': (0, 128, 0), 'fear': (128, 0, 128),
    'happy': (0, 255, 255), 'neutral': (128, 128, 128), 'sad': (255, 0, 0),
    'surprised': (255, 255, 0),
}

# ---- 加载模型 ----
print("Loading models...")
from ultralytics import YOLO
model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
face_detector = YOLO(os.path.join(model_dir, 'yolov8n-face.pt'), verbose=False)
if torch.cuda.is_available():
    face_detector.to('cuda')
print("✓ YOLOv8n-face loaded")

from transformers import pipeline
emotion_pipe = pipeline('image-classification',
    model='trpakov/vit-face-expression',
    device=0 if torch.cuda.is_available() else -1)
print("✓ ViT emotion model loaded")

# ---- 视频 ----
video_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(PROJECT_DIR), "Session1", "Session1", "dialog", "avi", "DivX",
    "Ses01F_impro03.avi")

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
mid_x = frame_w / 2

print(f"Video: {frame_w}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, {fps:.0f}fps, {total_frames} frames")
print(f"Sampling every 1.5s (~45 frames), output to {OUTPUT_DIR}/")

frame_count = 0

for fn in range(0, total_frames, int(fps * 1.5)):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
    ret, frame = cap.read()
    if not ret: continue
    frame_count += 1

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector(frame_rgb, verbose=False)

    # 收集检测结果
    dets = []
    if results[0].boxes:
        for box in results[0].boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            conf = float(box.conf[0]) if box.conf is not None else 0.5
            if conf > 0.3 and x2 > x1 and y2 > y1:
                dets.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

    # 左右各取最大
    left = [d for d in dets if (d['x1']+d['x2'])/2 < mid_x]
    right = [d for d in dets if (d['x1']+d['x2'])/2 >= mid_x]
    best = []
    if left: best.append(max(left, key=lambda d: (d['x2']-d['x1'])*(d['y2']-d['y1'])))
    if right: best.append(max(right, key=lambda d: (d['x2']-d['x1'])*(d['y2']-d['y1'])))

    # 画框 + 标注情绪
    img_pil = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img_pil)

    for d in best:
        x1, y1, x2, y2 = d['x1'], d['y1'], d['x2'], d['y2']
        face = frame_rgb[y1:y2, x1:x2]
        if face.size == 0: continue

        # ViT 情绪预测
        try:
            face_pil = Image.fromarray(cv2.resize(face, (224, 224)))
            result = emotion_pipe(face_pil)
            emo = result[0]['label']
            score = result[0]['score']
        except:
            emo, score = 'neutral', 0.0

        side = "L" if (x1+x2)/2 < mid_x else "R"
        color = EMOTION_COLORS.get(emo, (255, 255, 255))
        label = f"{side} {emo} ({score:.2f})"

        # 画框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        # 画标签背景
        tw, th = len(label) * 8, 20
        draw.rectangle([x1, y1-th, x1+tw, y1], fill=color)
        # 写标签 (PIL 不支持中文, 用英文)
        draw.text((x1+2, y1-th+2), label, fill=(0, 0, 0))

    # 画中线
    draw.line([(mid_x, 0), (mid_x, frame_w)], fill=(0, 255, 0), width=1)

    # 保存
    ts = fn / fps
    out_path = os.path.join(OUTPUT_DIR, f"frame_{fn:04d}_{ts:.1f}s.jpg")
    img_pil.save(out_path)
    print(f"  [{frame_count}] {out_path}: {len(best)} faces")

cap.release()
print(f"\nDone! {frame_count} frames saved to {OUTPUT_DIR}/")
