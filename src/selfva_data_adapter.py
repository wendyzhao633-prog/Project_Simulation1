from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


# Centralized defaults for fields not present in text_VA_emo.csv.
DEFAULT_RISK_TIER = "M"
DEFAULT_DOMAIN = "wellbeing"
DEFAULT_GOAL = "T2_debrief_diagnosis"


# Required mapping requested by user:
# 0=neutral, 1=happy, 2=sad, 3=angry, 4=fearful, 5=disgust, 6=surprise, 7=calm
EMOTION_LABEL_MAP: Dict[int, str] = {
    0: "neutral",
    1: "happy",
    2: "sad",
    3: "angry",
    4: "fearful",
    5: "disgust",
    6: "surprise",
    7: "calm",
}


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if not np.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _coerce_int_like(x: Any, default: int) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def _clip_pm1(x: float) -> float:
    return float(min(max(float(x), -1.0), 1.0))


def emotion_label_to_text(raw: Any) -> str:
    try:
        k = int(float(raw))
        return EMOTION_LABEL_MAP.get(k, str(k))
    except Exception:
        s = str(raw).strip().lower()
        return s if s else "unknown"


def load_text_va_emo_standard(csv_path: str) -> pd.DataFrame:
    """
    Load data/text_VA_emo.csv as the single source of truth.

    Output schema:
      - scenario_id: deterministic per row, format p{participants_id}_n{text_no}
      - user_text
      - valence_pm1
      - arousal_pm1
      - emotion_label
      - participants_id
      - text_no
      - source_row_index (1-based)
      - risk_tier/domain/goal (centralized defaults)

    IMPORTANT:
      Valence/Arousal are already in [-1,1] and are used directly.
      No [0,1] -> [-1,1] remapping is applied.
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    required_cols = ["participants_id", "No.", "text", "Valence", "Arousal", "emotion_label"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"text_VA_emo.csv missing required columns: {missing}")

    rows = []
    for i, r in df.iterrows():
        pid = _coerce_int_like(r.get("participants_id"), default=0)
        text_no = _coerce_int_like(r.get("No."), default=i + 1)
        scenario_id = f"p{pid}_n{text_no}"

        user_text = str(r.get("text", "")).strip()
        if not user_text:
            user_text = f"[AUTO] Empty text at row {i + 1}"

        valence_pm1 = _clip_pm1(_safe_float(r.get("Valence"), 0.0))
        arousal_pm1 = _clip_pm1(_safe_float(r.get("Arousal"), 0.0))
        emo_text = emotion_label_to_text(r.get("emotion_label"))

        rows.append(
            {
                "scenario_id": scenario_id,
                "user_text": user_text,
                "valence_pm1": valence_pm1,
                "arousal_pm1": arousal_pm1,
                "emotion_label": emo_text,
                "participants_id": pid,
                "text_no": text_no,
                "source_row_index": i + 1,
                "risk_tier": DEFAULT_RISK_TIER,
                "domain": DEFAULT_DOMAIN,
                "goal": DEFAULT_GOAL,
            }
        )

    out_df = pd.DataFrame(rows)
    if out_df["scenario_id"].duplicated().any():
        dups = out_df[out_df["scenario_id"].duplicated(keep=False)]["scenario_id"].tolist()
        raise ValueError(
            "Generated scenario_id is not unique. "
            "Expected unique participant_id + No. pairs. "
            f"Examples: {dups[:10]}"
        )
    return out_df
