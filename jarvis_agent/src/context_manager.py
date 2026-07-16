"""
上下文管理器 — 对话历史存储、用户画像维护、总结生成
========================================================
职责: 管理对话历史, 维护用户画像, 对话结束后生成总结
存储: SQLite 数据库 (轻量、无外部依赖)

数据结构:
  ┌─────────────────────────────────────────────┐
  │ conversations 表                             │
  │ ├── session_id (TEXT)  会话标识               │
  │ ├── timestamp (REAL)   时间戳                 │
  │ ├── window_start (REAL)                      │
  │ ├── user_emotion (TEXT)                      │
  │ ├── partner_emotion (TEXT)                   │
  │ ├── user_speech (TEXT)                       │
  │ ├── partner_speech (TEXT)                    │
  │ ├── jarvis_response (TEXT) Jarvis 的建议     │
  │ ├── tension_level (REAL)                     │
  │ └── conversation_phase (TEXT)                │
  ├─────────────────────────────────────────────┤
  │ summaries 表                                 │
  │ ├── session_id (TEXT)                        │
  │ ├── summary (TEXT)       对话总结             │
  │ ├── key_insights (TEXT)  关键洞察             │
  │ ├── partner_profile (TEXT) 对方画像更新       │
  │ └── created_at (REAL)                        │
  ├─────────────────────────────────────────────┤
  │ user_profiles 表                             │
  │ ├── user_id (TEXT)                           │
  │ ├── profile_json (TEXT) 用户画像 JSON         │
  │ └── updated_at (REAL)                        │
  └─────────────────────────────────────────────┘
"""

import os
import json
import sqlite3
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import time

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """单轮对话记录"""
    session_id: str
    timestamp: float
    window_start: float
    window_end: float
    user_emotion: str
    partner_emotion: str
    user_speech: str
    partner_speech: str
    jarvis_response: str
    tension_level: float
    conversation_phase: str
    user_speech_top2: str = ""
    partner_speech_top2: str = ""
    gate_prefix: str = ""


@dataclass
class SessionSummary:
    """会话总结"""
    session_id: str
    summary: str
    key_insights: str
    partner_profile: str
    user_feedback: str
    next_session_prep: str
    created_at: float = 0.0


class ContextManager:
    """
    上下文管理器

    功能:
    1. 存储对话历史 (每 20s 一个 turn)
    2. 检索最近 N 轮对话作为 Prompt 上下文
    3. 对话结束生成总结
    4. 维护用户画像 (从历史中学习偏好)
    """

    def __init__(self, config: Dict):
        # 固定写到 jarvis_agent/output/, 不读 config (避免相对路径随 cwd 变化)
        _project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.db_path = os.path.join(_project_dir, "output", "conversation_history.db")
        self.summaries_dir = os.path.join(_project_dir, "output", "summaries")
        self.max_history_turns = config.get("agent", {}).get("max_history_turns", 50)

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)

        self._init_db()
        logger.info(f"ContextManager initialized: db={self.db_path}")

    def _init_db(self):
        """初始化 SQLite 数据库表"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    window_start REAL,
                    window_end REAL,
                    user_emotion TEXT DEFAULT 'neutral',
                    partner_emotion TEXT DEFAULT 'neutral',
                    user_speech TEXT DEFAULT '',
                    partner_speech TEXT DEFAULT '',
                    user_speech_top2 TEXT DEFAULT '',
                    partner_speech_top2 TEXT DEFAULT '',
                    jarvis_response TEXT DEFAULT '',
                    gate_prefix TEXT DEFAULT '',
                    tension_level REAL DEFAULT 0.0,
                    conversation_phase TEXT DEFAULT 'development',
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                )
            """)
            # 迁移旧数据库：添加 Top2 列（如果不存在则忽略错误）
            try:
                conn.execute("ALTER TABLE conversations ADD COLUMN user_speech_top2 TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE conversations ADD COLUMN partner_speech_top2 TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE conversations ADD COLUMN gate_prefix TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            conn.execute("""
                CREATE TABLE IF NOT EXISTS summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    summary TEXT DEFAULT '',
                    key_insights TEXT DEFAULT '',
                    partner_profile TEXT DEFAULT '',
                    user_feedback TEXT DEFAULT '',
                    next_session_prep TEXT DEFAULT '',
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    profile_json TEXT DEFAULT '{}',
                    updated_at REAL DEFAULT (strftime('%s', 'now'))
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_time
                ON conversations(session_id, timestamp)
            """)
            conn.commit()

    # ============================================================
    # 对话历史 CRUD
    # ============================================================

    def save_turn(self, turn: ConversationTurn):
        """保存一轮对话"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO conversations
                   (session_id, timestamp, window_start, window_end,
                    user_emotion, partner_emotion, user_speech,
                    partner_speech, user_speech_top2, partner_speech_top2,
                    jarvis_response, gate_prefix, tension_level,
                    conversation_phase)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    turn.session_id, turn.timestamp,
                    turn.window_start, turn.window_end,
                    turn.user_emotion, turn.partner_emotion,
                    turn.user_speech, turn.partner_speech,
                    turn.user_speech_top2, turn.partner_speech_top2,
                    turn.jarvis_response, turn.gate_prefix,
                    turn.tension_level, turn.conversation_phase,
                ),
            )
            conn.commit()

    def get_recent_history(
        self, session_id: str, n_turns: int = 10
    ) -> List[ConversationTurn]:
        """获取最近 N 轮对话"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT * FROM conversations
                   WHERE session_id = ?
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (session_id, n_turns),
            ).fetchall()

        turns = []
        for row in reversed(rows):  # 按时间正序返回
            turns.append(ConversationTurn(
                session_id=row["session_id"],
                timestamp=row["timestamp"],
                window_start=row["window_start"],
                window_end=row["window_end"],
                user_emotion=row["user_emotion"],
                partner_emotion=row["partner_emotion"],
                user_speech=row["user_speech"],
                partner_speech=row["partner_speech"],
                user_speech_top2=row["user_speech_top2"] if "user_speech_top2" in row.keys() else "",
                partner_speech_top2=row["partner_speech_top2"] if "partner_speech_top2" in row.keys() else "",
                jarvis_response=row["jarvis_response"],
                gate_prefix=row["gate_prefix"] if "gate_prefix" in row.keys() else "",
                tension_level=row["tension_level"],
                conversation_phase=row["conversation_phase"],
            ))
        return turns

    def get_full_conversation(
        self, session_id: str
    ) -> List[ConversationTurn]:
        """获取完整对话"""
        return self.get_recent_history(session_id, n_turns=99999)

    def format_history_for_prompt(
        self, session_id: str, n_turns: int = 10
    ) -> str:
        """
        将历史对话格式化为 Prompt 可用的文本
        """
        turns = self.get_recent_history(session_id, n_turns)
        if not turns:
            return "(No previous history — this is the beginning of the conversation)"

        lines = []
        for i, turn in enumerate(turns):
            # 优先使用 Top2 格式，回退到纯文本
            user_display = turn.user_speech_top2 or turn.user_speech
            partner_display = turn.partner_speech_top2 or turn.partner_speech
            # 截取 gate_prefix 的简短标签（去掉具体句子文本）
            import re
            short_tag = re.sub(r" (?:in|:) '.+'", "", turn.gate_prefix) if turn.gate_prefix else ""
            lines.append(
                f"[Turn {i+1}]\n"
                f"  User: {user_display}\n"
                f"  Partner: {partner_display}"
                f"{'  ' + short_tag if short_tag else ''}"
            )
        return "\n\n".join(lines)

    # ============================================================
    # 对话总结
    # ============================================================

    def save_summary(self, summary: SessionSummary):
        """保存对话总结"""
        summary.created_at = time.time()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO summaries
                   (session_id, summary, key_insights, partner_profile,
                    user_feedback, next_session_prep, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    summary.session_id, summary.summary,
                    summary.key_insights, summary.partner_profile,
                    summary.user_feedback, summary.next_session_prep,
                    summary.created_at,
                ),
            )
            conn.commit()

        # 同时保存为文本文件
        summary_path = os.path.join(
            self.summaries_dir, f"{summary.session_id}_summary.txt"
        )
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"=== Session Summary: {summary.session_id} ===\n\n")
            f.write(f"## 总结\n{summary.summary}\n\n")
            f.write(f"## 关键洞察\n{summary.key_insights}\n\n")
            f.write(f"## 对方画像\n{summary.partner_profile}\n\n")
            f.write(f"## 用户反馈\n{summary.user_feedback}\n\n")
            f.write(f"## 下次会话准备\n{summary.next_session_prep}\n")

        logger.info(f"Summary saved: {summary_path}")

    def get_summary(self, session_id: str) -> Optional[SessionSummary]:
        """获取历史总结"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM summaries WHERE session_id = ?",
                (session_id,),
            ).fetchone()

        if not row:
            return None

        return SessionSummary(
            session_id=row["session_id"],
            summary=row["summary"],
            key_insights=row["key_insights"],
            partner_profile=row["partner_profile"],
            user_feedback=row["user_feedback"],
            next_session_prep=row["next_session_prep"],
            created_at=row["created_at"],
        )

    def get_all_summaries(self) -> List[SessionSummary]:
        """获取所有历史总结"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM summaries ORDER BY created_at DESC"
            ).fetchall()

        summaries = []
        for row in rows:
            summaries.append(SessionSummary(
                session_id=row["session_id"],
                summary=row["summary"],
                key_insights=row["key_insights"],
                partner_profile=row["partner_profile"],
                user_feedback=row["user_feedback"],
                next_session_prep=row["next_session_prep"],
                created_at=row["created_at"],
            ))
        return summaries

    # ============================================================
    # 用户画像
    # ============================================================

    def update_user_profile(self, user_id: str, profile_data: Dict):
        """更新用户画像"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO user_profiles
                   (user_id, profile_json, updated_at)
                   VALUES (?, ?, ?)""",
                (user_id, json.dumps(profile_data), time.time()),
            )
            conn.commit()

    def get_user_profile(self, user_id: str) -> Dict:
        """获取用户画像"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT profile_json FROM user_profiles WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if row:
            return json.loads(row[0])
        return {}

    # ============================================================
    # 数据聚合与统计
    # ============================================================

    def get_emotion_timeline(
        self, session_id: str
    ) -> List[Dict[str, Any]]:
        """获取情感时间线 (用于总结和可视化)"""
        turns = self.get_full_conversation(session_id)
        return [
            {
                "time": t.timestamp,
                "user_emotion": t.user_emotion,
                "partner_emotion": t.partner_emotion,
                "phase": t.conversation_phase,
            }
            for t in turns
        ]

    def get_conversation_stats(self, session_id: str) -> Dict:
        """获取对话统计信息"""
        turns = self.get_full_conversation(session_id)
        if not turns:
            return {}

        emotions = [t.user_emotion for t in turns]
        from collections import Counter
        emotion_dist = Counter(emotions)

        speech_words = sum(len(t.user_speech.split()) + len(t.partner_speech.split())
                          for t in turns)

        return {
            "total_turns": len(turns),
            "duration_seconds":  turns[-1].window_end - turns[0].window_start if turns else 0,
            "dominant_emotion": emotion_dist.most_common(1)[0][0] if emotion_dist else "neutral",
            "emotion_distribution": dict(emotion_dist),
            "total_words": speech_words,
            "phases": list(set(t.conversation_phase for t in turns)),
        }
