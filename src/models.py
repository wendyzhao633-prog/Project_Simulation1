"""
Data structures (Pydantic models)
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field


class Quality(BaseModel):
    snr: Optional[Literal["high", "medium", "low"]] = "high"
    motion: Optional[Literal["none", "low", "high"]] = "none"
    blur: Optional[Literal["none", "low", "high"]] = "none"
    other: Optional[Dict[str, Any]] = None


class AffectVA(BaseModel):
    # normalized VA in [-1, 1]
    valence: float = Field(..., ge=-1.0, le=1.0)
    arousal: float = Field(..., ge=-1.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)


class ModalitySample(BaseModel):
    modality: str
    valence: float = Field(..., ge=-1.0, le=1.0)
    arousal: float = Field(..., ge=-1.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    quality: Quality = Field(default_factory=Quality)
    metadata: Optional[Dict[str, Any]] = None


class FusedAffect(BaseModel):
    """
    Unified fused affect container.

    For VA condition:
      - valence/arousal/confidence are used.

    For PHYSIO condition:
      - hr_mean_bpm / hrv_* are used (valence/arousal can be dummy 0.0).
    """
    valence: float = 0.0
    arousal: float = 0.0
    confidence: float = Field(1.0, ge=0.0, le=1.0)

    # IMPORTANT: give a default so it won't trigger "Field required"
    used_modalities: List[str] = Field(default_factory=list)

    fusion_method: str = "weighted_average"

    # ---- Added for your PHYSIO branch ----
    hr_mean_bpm: Optional[float] = None
    hrv_sdnn_ms: Optional[float] = None
    hrv_rmssd_ms: Optional[float] = None
    hrv_pnn50_pct: Optional[float] = None


class DiscreteEmotion(BaseModel):
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    va_source: Optional[AffectVA] = None


class Scenario(BaseModel):
    scenario_id: str
    risk_tier: Literal["L", "M", "H"]
    domain: str
    goal: str
    user_text: str


class RunConfig(BaseModel):
    """
    Keep this for compatibility with existing utils/config loader.
    """
    llms: List[str]

    modalities_all: List[str]
    generate_all_combinations: bool = True
    modality_sets: Optional[List[List[str]]] = None

    # legacy
    modalities: Optional[List[List[str]]] = None

    affect_repr: List[Literal["VA", "DISCRETE", "PHYSIO"]]
    prompt: List[Literal["baseline", "soft", "explicit"]]

    va_levels: List[Dict[str, float]]

    seed: int = 42
    repeats_per_cell: int = 1

    emulator: Optional[Dict[str, Any]] = None
    discrete_mapping: Optional[Dict[str, Any]] = None

    thresholds: Dict[str, float]
    decoding: Dict[str, Any]

    provider_timeouts: Dict[str, int]
    reruns_on_rate_limit: int = 3
    sleep_on_rate_limit: float = 2.0

    budget: Optional[Dict[str, Any]] = None
    judge: Optional[Dict[str, Any]] = None
    filters: Optional[Dict[str, Any]] = None


class GenerationResult(BaseModel):
    text: str
    model: str
    provider: str
    duration_seconds: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    decoding_params: Dict[str, Any] = Field(default_factory=dict)


class RunLog(BaseModel):
    run_id: str
    timestamp: str
    scenario: Scenario

    modalities: List[str] = Field(default_factory=list)
    modality_samples: List[ModalitySample] = Field(default_factory=list)

    fused_affect: FusedAffect
    affect_repr: Literal["VA", "DISCRETE", "PHYSIO"]

    discrete_emotion: Optional[DiscreteEmotion] = None

    prompt_type: Literal["baseline", "soft", "explicit"] = "explicit"
    prompt_hash: str
    messages: List[Dict[str, str]]

    generation_result: GenerationResult

    modality_label: Optional[str] = None
    repeat: Optional[int] = None
    va_level: Optional[Dict[str, float]] = None


class Score(BaseModel):
    run_id: str
    emotional_competence: Optional[float] = Field(None, ge=1.0, le=5.0)
    empathy: float = Field(..., ge=1.0, le=5.0)
    usefulness: float = Field(..., ge=1.0, le=5.0)
    appropriateness: float = Field(..., ge=1.0, le=5.0)
    safety_stability: Optional[float] = Field(None, ge=1.0, le=5.0)

    # legacy
    stability: Optional[float] = Field(None, ge=1.0, le=5.0)
    safety: Optional[float] = Field(None, ge=1.0, le=5.0)

    comment: Optional[str] = None
    judge_type: Literal["human", "llm"] = "llm"
    weighted_overall: Optional[float] = None
