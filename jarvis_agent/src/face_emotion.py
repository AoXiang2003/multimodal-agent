"""
人脸表情识别模块 — YOLO检测 + ViT表情分类（含概率输出）
============================================================
职责: 对视频帧进行人脸检测和表情识别，输出硬标签 + 7维概率向量
"""

import os
import cv2
import numpy as np
import torch
torch.backends.cudnn.enabled = False
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import Counter
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class FaceEmotion:
    timestamp: float
    bbox: Tuple[int, int, int, int]
    emotion: str
    confidence: float
    face_side: str = "unknown"


class FaceEmotionAnalyzer:

    def __init__(self, dialog_name: str = "Ses01F_impro01"):
        self.dialog_name = dialog_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.main_gender = "F"
        if "Ses" in dialog_name and len(dialog_name) >= 6:
            self.main_gender = dialog_name[5]

        # ---- FaceDetector: YOLO优先 ----
        self._yolo_mode = False
        self.face_detector = None
        model_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "models"
        )

        # 尝试 YOLO
        try:
            from ultralytics import YOLO
            self.face_detector = YOLO(
                os.path.join(model_dir, 'yolov8n-face.pt'),
                verbose=False
            )
            if self.device == 'cuda':
                self.face_detector.to('cuda')
            self._yolo_mode = True
            logger.info(f"✓ YOLOv8n-face loaded ({self.device})")
        except Exception as e:
            logger.warning(f"YOLO failed: {e}")

        # YOLO fallback → MediaPipe
        if not self._yolo_mode:
            try:
                import mediapipe as mp
                from mediapipe.tasks.python import vision
                self.face_detector = vision.FaceDetector.create_from_options(
                    vision.FaceDetectorOptions(
                        base_options=mp.tasks.BaseOptions(
                            model_asset_path=os.path.join(
                                model_dir, 'blaze_face_short_range.tflite'
                            )
                        ),
                        min_detection_confidence=0.3
                    )
                )
                logger.info("✓ MediaPipe face detector loaded")
            except Exception as e:
                logger.warning(f"Face detector failed: {e}")
                self.face_detector = None

        # ---- ViT 面部表情识别 ----
        self.emotion_pipe = None
        try:
            from transformers import pipeline
            self.emotion_pipe = pipeline(
                'image-classification',
                model=os.path.join(model_dir, 'vit-face-expression'),
                device=0 if torch.cuda.is_available() else -1,
                top_k=7 
            )
            logger.info(f"✓ ViT face emotion loaded ({self.device})")
             # ===== 在这里添加 =====
            logger.info(f"ViT config: num_labels={self.emotion_pipe.model.config.num_labels}")
            logger.info(f"ViT config: id2label={self.emotion_pipe.model.config.id2label}")
    # ======================
        except Exception as e:
            logger.warning(f"ViT emotion failed: {e}")
            self.emotion_pipe = None

    def _predict_emotion_with_probs(self, face_img: np.ndarray):
        """
        返回 (情绪标签, 置信度, 7维概率向量)
        """
        if self.emotion_pipe is None:
            return "neutral", 0.5, [0.0] * 7
        try:
            face_pil = Image.fromarray(cv2.resize(face_img, (224, 224)))
            result = self.emotion_pipe(face_pil)
            # ===== 在这里加日志 =====
            logger.debug(f"ViT result labels: {[item['label'] for item in result]}")
            probs = [item['score'] for item in result]
            label = result[0]['label']
            score = result[0]['score']
            label_map = {
                'anger': 'angry', 'disgust': 'disgust', 'fear': 'fear',
                'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad',
                'surprised': 'surprised'
            }
            return label_map.get(label, label), float(score), probs
        except Exception:
            return "neutral", 0.5, [0.0] * 7

    def _predict_emotion(self, face_img: np.ndarray) -> Tuple[str, float]:
        """兼容旧接口：仅返回标签和置信度"""
        if self.emotion_pipe is None:
            return "neutral", 0.5
        try:
            face_pil = Image.fromarray(cv2.resize(face_img, (224, 224)))
            result = self.emotion_pipe(face_pil)
            label = result[0]['label']
            score = result[0]['score']
            label_map = {
                'anger': 'angry', 'disgust': 'disgust', 'fear': 'fear',
                'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad',
                'surprised': 'surprised'
            }
            return label_map.get(label, label), float(score)
        except Exception:
            return "neutral", 0.5
        
  
    def analyze_frames(self, video_path: str, timestamps: List[float]) -> List[dict]:
        """
       返回每个时间点的检测结果，包括概率向量和YOLO置信度
        """
        if not os.path.exists(video_path):
            return []

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        mid_x = frame_w / 2
        results = []

        for ts in timestamps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts * fps))
            ret, frame = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_bright = cv2.convertScaleAbs(frame_rgb, alpha=1.4, beta=30)

            if not self.face_detector:
                continue

            dets = []
            if self._yolo_mode:
                yolo_r = self.face_detector(frame_rgb, verbose=False)
                if yolo_r[0].boxes:
                    for box in yolo_r[0].boxes:
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                        conf = float(box.conf[0]) if box.conf is not None else 0.5
                        if conf > 0.3:
                            dets.append({
                                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'conf': conf
                            })
            else:
                import mediapipe as mp
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_bright)
                mp_r = self.face_detector.detect(mp_img)
                if mp_r.detections:
                    for det in mp_r.detections:
                        bbox = det.bounding_box
                        dets.append({
                            'x1': int(bbox.origin_x),
                            'y1': int(bbox.origin_y),
                            'x2': int(bbox.origin_x + bbox.width),
                            'y2': int(bbox.origin_y + bbox.height),
                            'conf': 0.5
                        })

            # 左右各取最大的一张脸
            left_dets = [d for d in dets if (d['x1'] + d['x2']) / 2 < mid_x]
            right_dets = [d for d in dets if (d['x1'] + d['x2']) / 2 >= mid_x]
            best_dets = []
            if left_dets:
                best_dets.append(max(left_dets, key=lambda d: (d['x2'] - d['x1']) * (d['y2'] - d['y1'])))
            if right_dets:
                best_dets.append(max(right_dets, key=lambda d: (d['x2'] - d['x1']) * (d['y2'] - d['y1'])))

            for d in best_dets:
                x1, y1, x2, y2 = d['x1'], d['y1'], d['x2'], d['y2']
                if x2 <= x1 or y2 <= y1:
                    continue
                face = frame_rgb[y1:y2, x1:x2]
                side = "left" if ((x1 + x2) / 2 < mid_x) else "right"

                emotion, conf, probs = self._predict_emotion_with_probs(face)

                results.append({
                    "timestamp": ts,
                    "bbox": (x1, y1, x2, y2),
                    "emotion": emotion,
                    "confidence": conf,
                    "yolo_conf": d['conf'],
                    "face_side": side,
                    "probs": probs,
                })
            logger.info(f"Frame {ts:.1f}s: detected {len(best_dets)} faces")

        cap.release()
        return results
    def analyze_utterance(self, video_path, start_time, end_time, num_samples=4):
        dur = end_time - start_time
        if dur <= 0.1:
            return {
                "left_emotion": "neutral",
                "right_emotion": "neutral",
                "left_probs": [0.0]*7,
                "right_probs": [0.0]*7,
                "left_top1": ("neutral", 0.0),
                "left_top2": ("neutral", 0.0),
                "right_top1": ("neutral", 0.0),
                "right_top2": ("neutral", 0.0),
                "valid_ratio_left": 0.0,
                "valid_ratio_right": 0.0,
            }

        # 生成采样时间点
        sample_ts = np.linspace(
            start_time + dur * 0.1,
            end_time - dur * 0.1,
            min(num_samples, max(1, int(dur * 2)))
        )
        frame_results = self.analyze_frames(video_path, sample_ts.tolist())

        # 按左右分组
        left_frames = [r for r in frame_results if r["face_side"] == "left"]
        right_frames = [r for r in frame_results if r["face_side"] == "right"]

        def aggregate_valid(frames):
            """
            对一组帧进行质量过滤，返回平均概率向量和有效占比
            """
            if not frames:
                return None, 0.0

            valid_probs = []
            for f in frames:
                # 获取各项质量指标
                yolo_conf = f.get("yolo_conf", 0.0)
                vit_maxprob = f.get("confidence", 0.0)  # 即ViT输出的top-1概率
                x1, y1, x2, y2 = f["bbox"]
                aspect = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 0

                # 三条件门控
                if (yolo_conf > 0.5) and (vit_maxprob > 0.35) and (0.4 < aspect < 2.5):
                    valid_probs.append(f["probs"])

            if not valid_probs:
                return None, 0.0

            avg_probs = np.mean(valid_probs, axis=0).tolist()
            valid_ratio = len(valid_probs) / len(frames)
            return avg_probs, valid_ratio

        left_avg, left_valid = aggregate_valid(left_frames)
        right_avg, right_valid = aggregate_valid(right_frames)

        # 辅助函数：从概率向量提取 Top-1, Top-2
        def top2_from_probs(probs):
            if not probs:
                return ("neutral", 0.0), ("neutral", 0.0)
            labels = ['angry','disgust','fear','happy','neutral','sad','surprised']
            sorted_idx = np.argsort(probs)[::-1]
            top1 = (labels[sorted_idx[0]], probs[sorted_idx[0]])
            top2 = (labels[sorted_idx[1]], probs[sorted_idx[1]]) if len(sorted_idx) > 1 else (labels[sorted_idx[0]], probs[sorted_idx[0]])
            return top1, top2

        left_top1, left_top2 = top2_from_probs(left_avg)
        right_top1, right_top2 = top2_from_probs(right_avg)

        return {
            "left_emotion": left_top1[0] if left_avg else "neutral",
            "right_emotion": right_top1[0] if right_avg else "neutral",
            "left_probs": left_avg if left_avg else [0.0]*7,
            "right_probs": right_avg if right_avg else [0.0]*7,
            "left_top1": left_top1,
            "left_top2": left_top2,
            "right_top1": right_top1,
            "right_top2": right_top2,
            "valid_ratio_left": left_valid,
            "valid_ratio_right": right_valid,
        }

    def analyze_dialog(self, video_path: str, utterance_list: List[Dict]) -> List[Dict]:
        """
        批量分析对话中的所有句子
        """
        results = []
        total = len(utterance_list)
        for idx, utt in enumerate(utterance_list):
            res = self.analyze_utterance(
                video_path,
                utt["start"],
                utt["end"],
                num_samples=3
            )
            results.append(res)
            if (idx + 1) % 10 == 0 or (idx + 1) == total:
                logger.info(f"  Face: {idx+1}/{total}")
        return results