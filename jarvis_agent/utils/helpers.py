"""
工具函数: 配置加载、时间处理、IEMOCAP 数据集路径解析
"""
import os
import re
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """加载 YAML 配置文件并解析环境变量"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 递归解析 ${ENV_VAR} 形式的环境变量
    def resolve_env(obj):
        if isinstance(obj, str):
            pattern = re.compile(r"\$\{(\w+)\}")
            matches = pattern.findall(obj)
            for m in matches:
                obj = obj.replace(f"${{{m}}}", os.environ.get(m, ""))
            return obj
        elif isinstance(obj, dict):
            return {k: resolve_env(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve_env(item) for item in obj]
        return obj

    return resolve_env(config)


def parse_iemocap_path(
    session_id: int,
    dialog_name: str,
    data_type: str,
    base_path: str = "../Session1/Session1",
) -> Optional[str]:
    """
    解析 IEMOCAP 数据集文件路径

    Args:
        session_id: Session 编号 (1-5)
        dialog_name: 对话名称, 如 Ses01F_impro01
        data_type: 数据类型 (avi, wav, transcriptions, MOCAP_head 等)
        base_path: 数据根目录

    Returns:
        完整文件路径, 不存在则返回 None
    """
    # 替换模板中的 Session 编号
    path = base_path.replace("Session1", f"Session{session_id}")

    if data_type == "avi":
        file_path = os.path.join(path, "dialog", "avi", "DivX", f"{dialog_name}.avi")
    elif data_type == "wav":
        file_path = os.path.join(path, "dialog", "wav", f"{dialog_name}.wav")
    elif data_type == "transcriptions":
        file_path = os.path.join(path, "dialog", "transcriptions", f"{dialog_name}.txt")
    elif data_type == "EmoEvaluation":
        file_path = os.path.join(path, "dialog", "EmoEvaluation", f"{dialog_name}.txt")
    elif data_type == "MOCAP_head":
        file_path = os.path.join(path, "dialog", "MOCAP_head", f"{dialog_name}.txt")
    elif data_type == "MOCAP_hand":
        file_path = os.path.join(path, "dialog", "MOCAP_hand", f"{dialog_name}.txt")
    elif data_type == "MOCAP_rotated":
        file_path = os.path.join(path, "dialog", "MOCAP_rotated", f"{dialog_name}.txt")
    else:
        file_path = os.path.join(path, "dialog", data_type, f"{dialog_name}.txt")

    if os.path.exists(file_path):
        return file_path
    return None


def parse_dialog_name_from_video(video_path: str) -> Tuple[str, int, str]:
    """
    从视频文件路径解析对话元信息

    Example:
        Ses01F_impro01.avi → ("Ses01F_impro01", 1, "F", "impro01")
    """
    basename = os.path.splitext(os.path.basename(video_path))[0]
    # Ses01F_impro01 or Ses01M_script02_1
    match = re.match(r"Ses(\d+)([FM])_(.+)$", basename)
    if not match:
        raise ValueError(f"Cannot parse dialog name: {basename}")

    session_id = int(match.group(1))
    gender = match.group(2)
    dialog_type = match.group(3)

    return basename, session_id, gender, dialog_type


def setup_logging(config: Dict[str, Any]):
    """配置日志系统"""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO"))
    fmt = log_config.get("format", "[%(asctime)s] [%(levelname)s] %(message)s")

    # 固定写到 jarvis_agent/output/logs/
    _project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _logs_dir = os.path.join(_project_dir, "output", "logs")
    os.makedirs(_logs_dir, exist_ok=True)

    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(_logs_dir, "jarvis.log"),
                encoding="utf-8",
            ),
        ],
    )


def mocap_time_to_seconds(frame_num: int) -> float:
    """
    IEMOCAP MOCAP 时间转换: 帧号 → 秒
    参考 timeinfo.txt: t2 = (t + 2) / 100
    """
    return (frame_num + 2) / 100.0
