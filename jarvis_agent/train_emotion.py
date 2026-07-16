#!/usr/bin/env python3
"""
wav2vec2-base 在 IEMOCAP 上微调情感识别
========================================
训练数据: Session1 + Session2
测试数据: Session1 第一个 avi 对应的对话 (Ses01F_impro01)

情绪合并策略:
  6 主类: frustrated, neutral, angry, sad, excited, happy
  surprised → excited
  disgust, fearful → sad

VAD 处理:
  - 归一化: (VAD - 3.0) → 均值为 0
  - 作为辅助回归目标 (多任务学习)

多评估者标签:
  - 类别: 加权投票 (每个评估者等权重)
  - VAD: 取所有评估者均值

模型: facebook/wav2vec2-base (95M 参数)
设备: GPU (CUDA) 优先
"""

import os, re, glob, json, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2Model,
    Wav2Vec2FeatureExtractor,
    TrainingArguments,
    Trainer,
    AutoConfig,
)
import librosa
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================================================
# 配置
# ================================================================
# 自动定位: train_emotion.py 在 jarvis_agent/ 下, 其父目录即项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSIONS = ["Session1", "Session2", "Session3", "Session4"]     # 训练数据
# TEST_SESSION = "Session1"
TEST_DIALOG  = "Ses01F_impro01"         # 测试用

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "finetuned_emotion")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 情绪映射
# 只用 IEMOCAP 第一行标签 — 保持原标签, 仅 sur→excited, xxx→unclear
EMOTION_MAP = {
    "fru": "frustrated", "neu": "neutral", "ang": "angry",
    "sad": "sad",        "exc": "excited", "hap": "happy",
    "sur": "excited",    "dis": "disgust", "fea": "fear",
    "xxx": "unclear",
}

LABEL_LIST = ["frustrated", "neutral", "angry", "sad", "excited", "happy", "disgust", "fear", "unclear"]
NUM_LABELS = len(LABEL_LIST)
label2id = {l: i for i, l in enumerate(LABEL_LIST)}
id2label = {i: l for i, l in enumerate(LABEL_LIST)}

# VAD 归一化
VAD_MEAN = 3.0


# ================================================================
# 数据解析
# ================================================================

def parse_emo_evaluation(filepath):
    """
    解析 IEMOCAP EmoEvaluation 文件.

    返回: [(utterance_id, emotion_votes_dict, vad_average), ...]

    格式:
      [6.2901 - 8.2357]  Ses01F_impro01_F000  neu  [2.5000, 2.5000, 2.5000]
      C-E2: Neutral; ()
      C-E3: Neutral; ()
      A-E3: val 3; act 2; dom 2; ()
    """
    utterances = []
    current = None

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue

            # 主行: [start - end] utterance_id emotion [v, a, d]
            m = re.match(
                r'\[(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\]\s+(\S+)\s+(\S+)\s+\[(.*?)\]',
                line
            )
            if m:
                if current:
                    utterances.append(current)
                current = {
                    "utt_id": m.group(3),
                    "emo": m.group(4),       # 直接取第一行标签 (fru/neu/ang/...)
                    "vad": [float(x.strip()) for x in m.group(5).split(',')],
                    "start": float(m.group(1)),
                    "end": float(m.group(2)),
                }
                continue

    if current:
        utterances.append(current)

    return utterances


def load_iemocap_data(sessions, base_dir):
    """
    加载指定 Session 的所有 IEMOCAP 数据.

    返回: [(wav_path, label_id, vad_normalized), ...]
    """
    samples = []
    skipped_xxx = 0
    skipped_no_wav = 0

    for sess in sessions:
        sess_dir = os.path.join(base_dir, sess, sess)
        emo_dir = os.path.join(sess_dir, "dialog", "EmoEvaluation")
        wav_base = os.path.join(sess_dir, "sentences", "wav")

        if not os.path.exists(emo_dir):
            continue

        for fname in os.listdir(emo_dir):
            if not fname.endswith('.txt'):
                continue

            dialog_name = fname.replace('.txt', '')
            utterances = parse_emo_evaluation(os.path.join(emo_dir, fname))

            for utt in utterances:
                utt_id = utt["utt_id"]
                raw_emo = utt["emo"]

                # 映射标签 (xxx → unclear, 不跳过)
                mapped = EMOTION_MAP.get(raw_emo, "unclear")
                label_id = label2id[mapped]

                # 评估: 该标签自身(单一标签所以平票不存在)
                majority_labels = [mapped]

                # VAD
                vad_mean = np.array(utt["vad"]) - VAD_MEAN

                # --- 找 WAV 文件 ---
                # utt_id 格式: Ses01F_impro01_F000
                # WAV 路径: Session1/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav
                parts = utt_id.rsplit('_', 1)
                if len(parts) == 2:
                    wav_dir_name = parts[0]
                    wav_path = os.path.join(wav_base, wav_dir_name, f"{utt_id}.wav")
                else:
                    skipped_no_wav += 1
                    continue

                if not os.path.exists(wav_path):
                    skipped_no_wav += 1
                    continue

                samples.append({
                    "wav_path": wav_path,
                    "label": label_id,
                    "emotion": mapped,
                    "majority_labels": majority_labels,  # 平票时多个
                    "vad": vad_mean.tolist(),
                    "utt_id": utt_id,
                    "dialog": dialog_name,
                    "start": utt["start"],
                    "end": utt["end"],
                })

    logger.info(f"Loaded {len(samples)} samples")
    logger.info(f"  Skipped (xxx/no consensus): {skipped_xxx}")
    logger.info(f"  Skipped (no wav): {skipped_no_wav}")

    return samples


# ================================================================
# PyTorch Dataset
# ================================================================

class IEMOCAPDataset(Dataset):
    def __init__(self, samples, processor, max_duration=10.0, sample_rate=16000):
        self.samples = samples
        self.processor = processor
        self.max_duration = max_duration
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        audio, sr = librosa.load(s["wav_path"], sr=self.sample_rate)
        max_len = int(self.max_duration * self.sample_rate)
        if len(audio) > max_len:
            audio = audio[:max_len]

        inputs = self.processor(
            audio, sampling_rate=self.sample_rate, return_tensors="pt",
            padding=False, truncation=True, max_length=max_len,
        )
        return {
            "input_values": inputs.input_values.squeeze(0),
            "labels": torch.tensor(s["label"], dtype=torch.long),
            "vad_labels": torch.tensor(s["vad"], dtype=torch.float),
        }


# ================================================================
# 模型定义 (wav2vec2 + 分类头 + VAD回归头)
# ================================================================

class Wav2Vec2ForIEMOCAP(nn.Module):
    """
    wav2vec2-base 微调 IEMOCAP:
    - 主任务: 6 类情绪分类
    - 辅助任务: VAD 三维回归

    损失 = CrossEntropyLoss(emo) + 0.3 * MSELoss(VAD)
    """
    def __init__(self, num_labels=6, vad_dim=3):
        super().__init__()
        # 优先加载本地模型
        _models_dir = os.path.join(BASE_DIR, "models", "wav2vec2-base")
        if os.path.exists(_models_dir):
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(_models_dir)
        else:
            self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

        hidden_size = self.wav2vec2.config.hidden_size  # 768

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_labels),
        )

        self.vad_regressor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, vad_dim),
        )

        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, input_values, attention_mask=None, labels=None, vad_labels=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(hidden)
        vad_pred = self.vad_regressor(hidden)

        loss = None
        if labels is not None:
            ce = self.ce_loss(logits, labels)
            mse = self.mse_loss(vad_pred, vad_labels) if vad_labels is not None else 0
            loss = ce + 0.3 * mse

        return {"loss": loss, "logits": logits, "vad_pred": vad_pred}


# ================================================================
# 训练入口
# ================================================================

def collate_fn(batch, processor):
    """Padding via processor"""
    max_len = max(b["input_values"].shape[0] for b in batch)
    padded_inputs = []
    for b in batch:
        iv = b["input_values"]
        pad_len = max_len - iv.shape[0]
        if pad_len > 0:
            iv = torch.cat([iv, torch.zeros(pad_len)])
        padded_inputs.append(iv)
    input_values = torch.stack(padded_inputs)
    # attention_mask: 1 for real, 0 for padding
    attention_mask = torch.stack([
        torch.cat([torch.ones(b["input_values"].shape[0]), torch.zeros(max_len - b["input_values"].shape[0])])
        for b in batch
    ])
    return {
        "input_values": input_values,
        "attention_mask": attention_mask,
        "labels": torch.stack([b["labels"] for b in batch]),
        "vad_labels": torch.stack([b["vad_labels"] for b in batch]),
    }


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- 1. 加载数据 ---
    logger.info("Loading IEMOCAP data (Session 1+2)...")
    all_samples = load_iemocap_data(SESSIONS, BASE_DIR)

    # --- 2. 划分训练/验证 ---
    train_samples, val_samples = train_test_split(
        all_samples, test_size=0.15, random_state=42,
        stratify=[s["label"] for s in all_samples]
    )
    logger.info(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # 打印分布
    train_dist = Counter(s["emotion"] for s in train_samples)
    logger.info(f"Train distribution: {dict(train_dist)}")

    # --- 3. 构建 Dataset ---
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        "facebook/wav2vec2-base", )
    train_dataset = IEMOCAPDataset(train_samples, processor)
    val_dataset = IEMOCAPDataset(val_samples, processor)

    # --- 4. 构建 DataLoader ---
    class Collator:
        def __init__(self, proc):
            self.proc = proc
        def __call__(self, batch):
            return collate_fn(batch, self.proc)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                              collate_fn=Collator(processor), num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,
                            collate_fn=Collator(processor), num_workers=0)

    # --- 5. 初始化模型 ---
    logger.info("Initializing wav2vec2-base model...")
    model = Wav2Vec2ForIEMOCAP(num_labels=NUM_LABELS).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Total parameters: {total_params:.1f}M")

    # --- 6. 训练 ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    num_epochs = 24

    logger.info(f"Starting training ({num_epochs} epochs)...")
    model.train()
    best_val_acc = 0.0
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            vad_labels = batch["vad_labels"].to(device)

            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(input_values, attention_mask, labels, vad_labels)
                    loss = outputs["loss"]
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_values, attention_mask, labels, vad_labels)
                loss = outputs["loss"]
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            preds = outputs["logits"].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 50 == 0:
                logger.info(f"  Epoch {epoch+1}, Batch {batch_idx+1}: "
                           f"loss={total_loss/(batch_idx+1):.4f}, "
                           f"acc={correct/total*100:.1f}%")

        train_acc = correct / total * 100
        logger.info(f"Epoch {epoch+1} complete: loss={total_loss/len(train_loader):.4f}, "
                   f"acc={train_acc:.1f}%")

        # --- 验证 (多标签匹配) ---
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_values = batch["input_values"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_values, attention_mask)
                preds = outputs["logits"].argmax(dim=-1)
                # 检查每个预测是否命中多数票标签之一
                for i, pred_idx in enumerate(preds.tolist()):
                    pred_emo = id2label[pred_idx]
                    # 找对应样本的多数票标签
                    sample_idx = val_total + i
                    if sample_idx < len(val_samples):
                        majority = val_samples[sample_idx].get("majority_labels",
                            [val_samples[sample_idx]["emotion"]])
                        val_correct += (pred_emo in majority)
                val_total += labels.size(0)
        val_acc = val_correct / val_total * 100
        logger.info(f"Validation accuracy (multi-label): {val_acc:.1f}%")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(OUTPUT_DIR, "wav2vec2_iemocap_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_list": LABEL_LIST,
                "id2label": id2label,
                "label2id": label2id,
                "epoch": epoch + 1,
                "val_acc": val_acc,
                "config": {
                    "num_labels": NUM_LABELS,
                    "model_name": "facebook/wav2vec2-base",
                    "train_sessions": SESSIONS,
                },
            }, best_path)
            logger.info(f"  Saved best model (epoch {epoch+1}, val_acc={val_acc:.1f}%)")
        model.train()

    # --- 7. 保存模型 ---
    save_path = os.path.join(OUTPUT_DIR, "wav2vec2_iemocap_finetuned.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "label_list": LABEL_LIST,
        "id2label": id2label,
        "label2id": label2id,
        "config": {
            "num_labels": NUM_LABELS,
            "model_name": "facebook/wav2vec2-base",
            "train_sessions": SESSIONS,
        },
    }, save_path)
    logger.info(f"Model saved to: {save_path}")

    # 保存 processor config (离线)
    try:
        from transformers import Wav2Vec2Config
        config = Wav2Vec2Config.from_pretrained(
            "facebook/wav2vec2-base", )
        config.num_labels = NUM_LABELS
        config.id2label = id2label
        config.label2id = label2id
        config.save_pretrained(OUTPUT_DIR)
        logger.info(f"Config saved to: {OUTPUT_DIR}")
    except Exception as e:
        logger.warning(f"Config save skipped: {e}")

    # --- 8. 测试 (Ses01F_impro01) ---
    logger.info(f"\nTesting on {TEST_DIALOG}...")
    test_samples = [s for s in all_samples if TEST_DIALOG in s["dialog"]]
    logger.info(f"Test samples: {len(test_samples)}")

    model.eval()
    test_correct = 0
    with torch.no_grad():
        for s in test_samples:
            audio, sr = librosa.load(s["wav_path"], sr=16000)
            if len(audio) > 160000:
                audio = audio[:160000]
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            input_values = inputs.input_values.to(device)
            outputs = model(input_values)
            pred = outputs["logits"].argmax(dim=-1).item()
            majority = s.get("majority_labels", [s["emotion"]])
            pred_emo = id2label[pred]
            if pred_emo in majority:
                test_correct += 1
    test_acc = test_correct / len(test_samples) * 100
    logger.info(f"Test accuracy on {TEST_DIALOG} (multi-label): {test_acc:.1f}%")

    return model


if __name__ == "__main__":
    train()
