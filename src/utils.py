"""
工具函数
"""

import os
import json
import yaml
import hashlib
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

from .models import RunConfig, Scenario


def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)


def load_config(config_path: str) -> RunConfig:
    """加载运行配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return RunConfig(**data)


def load_scenarios(csv_path: str) -> List[Scenario]:
    """加载情景库"""
    df = pd.read_csv(csv_path, encoding='utf-8')
    scenarios = []
    for _, row in df.iterrows():
        scenarios.append(Scenario(
            scenario_id=row['scenario_id'],
            risk_tier=row['risk_tier'],
            domain=row['domain'],
            goal=row['goal'],
            user_text=row['user_text']
        ))
    return scenarios


def ensure_dir(path: str):
    """确保目录存在"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def jsonl_append(path: str, data: Dict[str, Any]):
    """追加 JSONL 行"""
    ensure_dir(path)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')


def hash_prompt(messages: List[Dict[str, str]]) -> str:
    """计算 prompt 的短哈希"""
    content = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """读取 JSONL 文件"""
    results = []
    if not os.path.exists(path):
        return results
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def get_env_or_raise(key: str) -> str:
    """获取环境变量，不存在则抛出异常"""
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Environment variable {key} is not set. Please check your .env file.")
    return value

