from __future__ import annotations

import os
import uuid
import asyncio
import argparse
from pathlib import Path
from datetime import datetime, UTC
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

if __package__ in (None, ""):
    import sys

    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(THIS_DIR)
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)

    from src.utils import get_env_or_raise, jsonl_append  # type: ignore
    from src.models import Scenario, FusedAffect  # type: ignore
    from src.prompt_builder import build_explicit_prompt, build_strategy_prompt  # type: ignore
    from src.llm_providers.openai_provider import OpenAIProvider  # type: ignore
    from src.selfva_data_adapter import load_text_va_emo_standard  # type: ignore
else:
    from .utils import get_env_or_raise, jsonl_append
    from .models import Scenario, FusedAffect
    from .prompt_builder import build_explicit_prompt, build_strategy_prompt
    from .llm_providers.openai_provider import OpenAIProvider
    from .selfva_data_adapter import load_text_va_emo_standard


# Valid prompt strategies for this pipeline
VALID_STRATEGIES = {"minimal_adaptive", "empathy_then_help", "adaptive_priority"}
# Legacy prompt type kept for backward-compat
LEGACY_PROMPT_TYPE = "explicit"


def _safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if not np.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def _normalize_condition(condition: str) -> str:
    c = str(condition).upper().strip()
    return {"A_ONLY": "AROUSAL_ONLY", "AROUSAL": "AROUSAL_ONLY"}.get(c, c)


def _scenario_to_dict(scenario: Scenario) -> Dict[str, Any]:
    if hasattr(scenario, "model_dump"):
        sdict = scenario.model_dump()
    elif hasattr(scenario, "dict"):
        sdict = scenario.dict()
    else:
        sdict = dict(getattr(scenario, "__dict__", {}))
    sdict["scenario_id"] = sdict.get("scenario_id") or "unknown"
    return sdict


def add_va_noise(
    v_true: float,
    a_true: float,
    sigma_v: float,
    sigma_a: float,
    rng: np.random.RandomState,
) -> Tuple[float, float]:
    return (
        _clip(v_true + rng.normal(0.0, sigma_v), -1.0, 1.0),
        _clip(a_true + rng.normal(0.0, sigma_a), -1.0, 1.0),
    )


def sample_subject_params(rng: np.random.RandomState) -> Dict[str, float]:
    return {
        "hr_base": float(rng.normal(72.0, 8.0)),
        "rmssd_base": float(rng.normal(40.0, 12.0)),
        "sdnn_base": float(rng.normal(50.0, 15.0)),
        "alpha_hr": float(rng.normal(16.0, 4.0)),
        "beta_hr": float(rng.normal(4.5, 1.5)),
        "gamma_rmssd": float(rng.normal(14.0, 4.0)),
        "delta_rmssd": float(rng.normal(8.0, 3.0)),
        "eta_sdnn": float(rng.normal(16.0, 5.0)),
        "theta_sdnn": float(rng.normal(10.0, 4.0)),
    }


def _emotion_adjustment(emotion_label: str) -> Dict[str, float]:
    # Required mapping:
    # 0 neutral, 1 happy, 2 sad, 3 angry, 4 fearful, 5 disgust, 6 surprise, 7 calm
    e = (emotion_label or "").strip().lower()
    if e == "calm":
        return {"hr": -2.5, "rmssd": 6.0, "sdnn": 7.0}
    if e == "happy":
        return {"hr": 1.0, "rmssd": 3.0, "sdnn": 3.0}
    if e in {"fearful", "angry"}:
        return {"hr": 5.0, "rmssd": -7.0, "sdnn": -8.0}
    if e == "disgust":
        return {"hr": 2.0, "rmssd": -3.0, "sdnn": -3.0}
    if e == "sad":
        return {"hr": 0.0, "rmssd": -3.0, "sdnn": -3.5}
    if e == "surprise":
        return {"hr": 2.0, "rmssd": -2.0, "sdnn": -2.0}
    return {"hr": 0.0, "rmssd": 0.0, "sdnn": 0.0}


def generate_physio_from_va(
    v_true: float,
    a_true: float,
    subj: Dict[str, float],
    emotion_label: str,
    rng: np.random.RandomState,
) -> Dict[str, float]:
    a = _clip(a_true, -1.0, 1.0)
    v = _clip(v_true, -1.0, 1.0)
    a_pos = max(a, 0.0)
    v_buffer = v * (1.0 - abs(a))

    scenario_hr = rng.normal(0.0, 1.6)
    scenario_hrv = rng.normal(0.0, 3.2)
    meas_hr = rng.normal(0.0, 1.2)
    meas_hrv_1 = rng.normal(0.0, 2.6)
    meas_hrv_2 = rng.normal(0.0, 2.6)

    hr = subj["hr_base"] + subj["alpha_hr"] * a + subj["beta_hr"] * (-v * a_pos) + scenario_hr + meas_hr
    rmssd = subj["rmssd_base"] - subj["gamma_rmssd"] * a_pos + subj["delta_rmssd"] * v_buffer + scenario_hrv + meas_hrv_1
    sdnn = subj["sdnn_base"] - subj["eta_sdnn"] * a_pos + subj["theta_sdnn"] * v_buffer + scenario_hrv + meas_hrv_2

    adj = _emotion_adjustment(emotion_label)
    hr += adj["hr"]
    rmssd += adj["rmssd"]
    sdnn += adj["sdnn"]

    hr = _clip(hr, 45.0, 180.0)
    rmssd = _clip(rmssd, 5.0, 180.0)
    sdnn = _clip(sdnn, 5.0, 220.0)
    pnn50 = _clip(0.52 * rmssd + rng.normal(0.0, 4.0), 0.0, 100.0)
    return {
        "hr_mean_bpm": float(round(hr, 1)),
        "rmssd_ms": float(round(rmssd, 1)),
        "sdnn_ms": float(round(sdnn, 1)),
        "pnn50_pct": float(round(pnn50, 1)),
    }


def build_fused_affect_for_condition(
    condition: str,
    v_noisy: float,
    a_noisy: float,
    physio: Dict[str, float] | None,
) -> FusedAffect:
    condition = condition.upper().strip()
    if condition == "VA":
        return FusedAffect(valence=float(v_noisy), arousal=float(a_noisy), confidence=1.0, used_modalities=["VA"], fusion_method="self_report_noisy")
    if condition == "AROUSAL_ONLY":
        return FusedAffect(valence=0.0, arousal=float(a_noisy), confidence=1.0, used_modalities=["VA"], fusion_method="arousal_only_noisy")
    if condition == "PHYSIO":
        assert physio is not None
        return FusedAffect(
            valence=0.0,
            arousal=0.0,
            confidence=1.0,
            used_modalities=["PHYSIO"],
            fusion_method="simulated_physio_from_va",
            hr_mean_bpm=_safe_float(physio.get("hr_mean_bpm"), 0.0),
            hrv_sdnn_ms=_safe_float(physio.get("sdnn_ms"), 0.0),
            hrv_rmssd_ms=_safe_float(physio.get("rmssd_ms"), 0.0),
            hrv_pnn50_pct=_safe_float(physio.get("pnn50_pct"), 0.0),
        )
    raise ValueError(f"Unknown condition: {condition}")


async def one_call(provider: OpenAIProvider, messages: List[Dict[str, str]]):
    return await asyncio.to_thread(provider.generate, messages, {"temperature": 0.7, "top_p": 1.0, "max_tokens": 400})


async def run_one(
    provider: OpenAIProvider,
    scenario: Scenario,
    condition: str,
    subject_id: int,
    trial_id: int,
    v_true: float,
    a_true: float,
    v_noisy: float,
    a_noisy: float,
    physio: Dict[str, float] | None,
    sigma_v: float,
    sigma_a: float,
    source_affect_id: str,
    emotion_label: str,
    risk_tier: str,
    prompt_strategy: str = "minimal_adaptive",
) -> Dict[str, Any]:
    cond_upper = condition.upper()
    fused = build_fused_affect_for_condition(cond_upper, v_noisy, a_noisy, physio)

    # Resolve affect_repr from condition
    affect_repr_map = {
        "VA": "VA",
        "AROUSAL_ONLY": "AROUSAL_ONLY",
        "PHYSIO": "PHYSIO",
    }
    affect_repr = affect_repr_map.get(cond_upper, "VA")

    if prompt_strategy in VALID_STRATEGIES:
        messages = build_strategy_prompt(
            strategy=prompt_strategy,
            scenario=scenario,
            fused_affect=fused,
            affect_repr=affect_repr,
        )
        used_prompt_type = prompt_strategy
    else:
        # Legacy fallback
        messages = build_explicit_prompt(
            scenario=scenario,
            fused_affect=fused,
            affect_repr=affect_repr,
            discrete_label=None,
            add_safety_block=False,
        )
        used_prompt_type = LEGACY_PROMPT_TYPE

    gen_result = await one_call(provider, messages)
    if hasattr(gen_result, "model_dump"):
        response_obj = gen_result.model_dump()
    elif hasattr(gen_result, "dict"):
        response_obj = gen_result.dict()
    else:
        response_obj = {"text": getattr(gen_result, "text", str(gen_result))}

    scenario_dict = _scenario_to_dict(scenario)
    return {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(UTC).isoformat(),
        "scenario": scenario_dict,
        "meta": {
            "condition": cond_upper,
            "prompt_strategy": used_prompt_type,
            "scenario_id": scenario_dict.get("scenario_id"),
            "source_affect_id": source_affect_id,
            "subject": subject_id,
            "trial": trial_id,
            "sigma_v": sigma_v,
            "sigma_a": sigma_a,
            "emotion_label": emotion_label,
            "risk_tier": risk_tier,
            "alignment_matched_by": "direct_text_va_emo_row",
            "alignment_match_ok": True,
        },
        "scenario_id": scenario_dict.get("scenario_id"),
        "source_affect_id": source_affect_id,
        "subject_id": subject_id,
        "trial": trial_id,
        "condition": cond_upper,
        "prompt_strategy": used_prompt_type,
        "valence": float(v_noisy),
        "arousal": float(a_noisy),
        "HR": None if physio is None else _safe_float(physio.get("hr_mean_bpm"), 0.0),
        "RMSSD": None if physio is None else _safe_float(physio.get("rmssd_ms"), 0.0),
        "SDNN": None if physio is None else _safe_float(physio.get("sdnn_ms"), 0.0),
        "emotion_label": emotion_label,
        "risk_tier": risk_tier,
        "alignment_matched_by": "direct_text_va_emo_row",
        "alignment_match_ok": True,
        "inputs": {
            "self_report_true": {"valence": v_true, "arousal": a_true, "scale": "normalized_-1_to_1"},
            "self_report_noisy": (
                {"valence": v_noisy, "arousal": a_noisy, "scale": "normalized_-1_to_1"}
                if cond_upper == "VA"
                else {"arousal": a_noisy, "scale": "normalized_-1_to_1"}
                if cond_upper == "AROUSAL_ONLY"
                else None
            ),
            "physio": physio,
        },
        "prompt_type": used_prompt_type,
        "messages": messages,
        "response": response_obj,
    }


async def bounded_gather(tasks, concurrency: int):
    sem = asyncio.Semaphore(concurrency)

    async def _wrap(coro):
        async with sem:
            return await coro

    return await asyncio.gather(*[_wrap(t) for t in tasks])


def _build_scenarios_from_df(df: pd.DataFrame, n_scenarios: int) -> List[Tuple[Scenario, Dict[str, Any]]]:
    rows = df.head(n_scenarios).copy().reset_index(drop=True)
    out: List[Tuple[Scenario, Dict[str, Any]]] = []
    for _, r in rows.iterrows():
        sc = Scenario(
            scenario_id=str(r["scenario_id"]),
            risk_tier=str(r["risk_tier"]),
            domain=str(r["domain"]),
            goal=str(r["goal"]),
            user_text=str(r["user_text"]),
        )
        out.append((sc, r.to_dict()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self_va_csv", type=str, default="data/text_VA_emo.csv")
    ap.add_argument("--conditions", nargs="+", default=["VA", "PHYSIO", "AROUSAL_ONLY"])
    ap.add_argument("--comparisons", nargs="+", default=["VA:PHYSIO", "AROUSAL_ONLY:PHYSIO"])
    # Prompt strategies: primary is minimal_adaptive; others are optional comparisons
    ap.add_argument(
        "--prompt_strategies",
        nargs="+",
        default=["minimal_adaptive"],
        choices=sorted(VALID_STRATEGIES) + [LEGACY_PROMPT_TYPE],
        help=(
            "One or more prompt strategies to run.  "
            "Primary experiment: minimal_adaptive.  "
            "Optional comparisons: empathy_then_help, adaptive_priority.  "
            "Legacy option: explicit."
        ),
    )
    ap.add_argument("--n_scenarios", type=int, default=80)
    ap.add_argument("--n_subjects", type=int, default=10)
    ap.add_argument("--n_trials", type=int, default=1)
    ap.add_argument("--sigma_v", type=float, default=0.25)
    ap.add_argument("--sigma_a", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--llm", type=str, default="gpt-5.4-2026-03-05")
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--out_jsonl", type=str, default="outputs/selfva_runs_v4_80.jsonl")
    ap.add_argument("--out_features_csv", type=str, default="outputs/selfva_simulated_features_v4_80.csv")
    ap.add_argument("--alignment_report_csv", type=str, default="outputs/selfva_alignment_report_v4_80.csv")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")
    provider = OpenAIProvider(model=args.llm, api_key=get_env_or_raise("OPENAI_API_KEY"))

    prompt_strategies = args.prompt_strategies
    print(f"[OK] Prompt strategies: {prompt_strategies}")

    # Single source of truth: text_VA_emo.csv
    std_df = load_text_va_emo_standard(args.self_va_csv)
    n_used = min(args.n_scenarios, len(std_df))
    print(f"[OK] Loaded {len(std_df)} rows from {args.self_va_csv}; using n_scenarios={n_used}")

    scenario_items = _build_scenarios_from_df(std_df, n_used)
    conditions = [_normalize_condition(c) for c in args.conditions]
    valid_conditions = {"VA", "PHYSIO", "AROUSAL_ONLY"}
    bad = [c for c in conditions if c not in valid_conditions]
    if bad:
        raise ValueError(f"Unsupported conditions: {bad}. Allowed: {sorted(valid_conditions)}")

    # Keep report output path for downstream compatibility; now direct one-to-one mapping.
    align_report = pd.DataFrame(
        [
            {
                "row_idx": int(row["source_row_index"]) - 1,
                "scenario_id_scenarios_csv": row["scenario_id"],
                "scenario_id_affect_csv": row["scenario_id"],
                "matched_by": "direct_text_va_emo_row",
                "match_ok": True,
                "fallback_used": False,
            }
            for _, row in std_df.head(n_used).iterrows()
        ]
    )
    os.makedirs(os.path.dirname(args.alignment_report_csv), exist_ok=True)
    align_report.to_csv(args.alignment_report_csv, index=False, encoding="utf-8-sig")

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    if os.path.exists(args.out_jsonl):
        os.remove(args.out_jsonl)

    feature_rows: List[Dict[str, Any]] = []

    async def _runner():
        tasks = []
        for sc, row in scenario_items:
            v_true = _clip(_safe_float(row["valence_pm1"], 0.0), -1.0, 1.0)
            a_true = _clip(_safe_float(row["arousal_pm1"], 0.0), -1.0, 1.0)
            emotion_label = str(row["emotion_label"])
            risk_tier = str(row["risk_tier"])
            source_affect_id = str(row["scenario_id"])

            for subj_id in range(1, args.n_subjects + 1):
                subj = sample_subject_params(rng)
                for t in range(args.n_trials):
                    v_noisy, a_noisy = add_va_noise(v_true, a_true, args.sigma_v, args.sigma_a, rng)
                    physio = generate_physio_from_va(v_true, a_true, subj, emotion_label, rng)
                    feature_rows.append(
                        {
                            "scenario_id": str(sc.scenario_id),
                            "source_affect_id": source_affect_id,
                            "participants_id": int(row["participants_id"]),
                            "text_no": int(row["text_no"]),
                            "subject_id": subj_id,
                            "trial": t + 1,
                            "valence_true": v_true,
                            "arousal_true": a_true,
                            "valence_noisy": v_noisy,
                            "arousal_noisy": a_noisy,
                            "HR": physio.get("hr_mean_bpm"),
                            "RMSSD": physio.get("rmssd_ms"),
                            "SDNN": physio.get("sdnn_ms"),
                            "pNN50": physio.get("pnn50_pct"),
                            "emotion_label": emotion_label,
                            "risk_tier": risk_tier,
                            "matched_by": "direct_text_va_emo_row",
                            "match_ok": True,
                        }
                    )

                    for cond in conditions:
                        for strategy in prompt_strategies:
                            tasks.append(
                                run_one(
                                    provider=provider,
                                    scenario=sc,
                                    condition=cond,
                                    subject_id=subj_id,
                                    trial_id=t + 1,
                                    v_true=v_true,
                                    a_true=a_true,
                                    v_noisy=v_noisy,
                                    a_noisy=a_noisy,
                                    physio=physio if cond == "PHYSIO" else None,
                                    sigma_v=args.sigma_v,
                                    sigma_a=args.sigma_a,
                                    source_affect_id=source_affect_id,
                                    emotion_label=emotion_label,
                                    risk_tier=risk_tier,
                                    prompt_strategy=strategy,
                                )
                            )

        results = await bounded_gather(tasks, concurrency=args.concurrency)
        for r in results:
            jsonl_append(args.out_jsonl, r)
        os.makedirs(os.path.dirname(args.out_features_csv), exist_ok=True)
        pd.DataFrame(feature_rows).to_csv(args.out_features_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] Wrote: {args.out_jsonl} (rows={len(results)})")
        print(f"[OK] Wrote: {args.out_features_csv} (rows={len(feature_rows)})")
        print(f"[OK] Wrote: {args.alignment_report_csv} (rows={len(align_report)})")

    asyncio.run(_runner())


if __name__ == "__main__":
    main()
