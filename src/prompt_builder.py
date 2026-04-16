"""
Prompt Generator - Risk-Agnostic AND Scenario-Agnostic Design

Supports three strategies: baseline / soft / explicit

CRITICAL DESIGN PRINCIPLE (mostly preserved):
Generation is FULLY SCENARIO-AGNOSTIC and RISK-AGNOSTIC.
The model sees ONLY:
  - user_text (message content)
  - optional affect representation (VA / PHYSIO / discrete label)
  - prompt structure (baseline/soft/explicit)

The model does NOT see:
  - scenario.domain
  - scenario.risk_tier
  - any role descriptions
  - any rubric descriptions in generation

NOTE (New for this project stage):
- We allow an OPTIONAL lightweight task_type hint (T1–T4) extracted from scenario.goal,
  purely to stabilize response style and reduce evaluation ties.
- This hint is not a rubric; it is a short "what kind of help the user wants" tag.
"""

import json
import re
from typing import List, Dict, Literal, Optional

from .models import Scenario, FusedAffect


# ---------------------------------------------------------------------------
# Strategy system prompts  (VA variants)
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS: Dict[str, str] = {
    "minimal_adaptive": (
        "You are a helpful, natural, and emotionally appropriate assistant.\n"
        "Reply to the user's message in a way that is supportive when needed and practical when needed.\n"
        "If affect context is provided, use it only to adjust tone, pacing, validation, and priorities.\n"
        "Do not explicitly mention hidden metadata such as valence, arousal, or emotion labels.\n"
        "Do not invent facts or physiological data.\n"
        "Write exactly one assistant reply."
    ),
    "empathy_then_help": (
        "You are a helpful, natural, and emotionally attuned assistant.\n"
        "Reply to the user's message in a way that first briefly acknowledges the user's likely emotional "
        "state, then provides practical and useful help.\n"
        "If affect context is provided, use it to guide how you acknowledge the feeling, how intense your "
        "wording should be, and what kind of help should come next.\n"
        "Keep the emotional acknowledgement short, natural, and proportionate to the user's likely state.\n"
        "After that, move clearly into helpful next steps, guidance, or an answer to the user's request.\n"
        "Do not over-dramatize, over-comfort, or sound scripted.\n"
        "Do not explicitly mention hidden metadata such as valence, arousal, or emotion labels.\n"
        "Do not invent facts or physiological data.\n"
        "Write exactly one assistant reply."
    ),
    "adaptive_priority": (
        "You are a helpful, natural, and emotionally adaptive assistant.\n"
        "Reply in a way that chooses the right balance between emotional attunement and practical help "
        "based on the user's message and any affect context provided.\n"
        "If the message is mainly task-focused or low-emotion, prioritize a direct, efficient, and helpful "
        "answer with only light emotional acknowledgement if needed.\n"
        "If the message is strongly emotional and not strongly task-focused, prioritize validation, "
        "steadiness, and one small useful next step.\n"
        "If both emotion and task matter, combine brief emotional acknowledgement with concrete, "
        "prioritized help.\n"
        "Use affect context only to adjust tone, pacing, validation, and priorities.\n"
        "Do not over-empathize task-heavy requests, and do not under-support strongly emotional requests.\n"
        "Do not explicitly mention hidden metadata such as valence, arousal, or emotion labels.\n"
        "Do not invent facts or physiological data.\n"
        "Write exactly one assistant reply."
    ),
    # --- PHYSIO variants: replace affect-context sentence and metadata note ---
    "minimal_adaptive_physio": (
        "You are a helpful, natural, and emotionally appropriate assistant.\n"
        "Reply to the user's message in a way that is supportive when needed and practical when needed.\n"
        "If physiological context is provided (such as heart rate or heart rate variability), treat it as "
        "a noisy indicator of the user's stress or arousal level. Use it only to adjust tone, pacing, "
        "validation, and priorities.\n"
        "Do not explicitly mention the physiological values to the user.\n"
        "Do not invent facts or physiological data.\n"
        "Write exactly one assistant reply."
    ),
    "empathy_then_help_physio": (
        "You are a helpful, natural, and emotionally attuned assistant.\n"
        "Reply to the user's message in a way that first briefly acknowledges the user's likely emotional "
        "state, then provides practical and useful help.\n"
        "If physiological context is provided (such as heart rate or heart rate variability), treat it as "
        "a noisy indicator of the user's stress or arousal level. Use it to guide how you acknowledge the "
        "feeling, how intense your wording should be, and what kind of help should come next.\n"
        "Keep the emotional acknowledgement short, natural, and proportionate to the user's likely state.\n"
        "After that, move clearly into helpful next steps, guidance, or an answer to the user's request.\n"
        "Do not over-dramatize, over-comfort, or sound scripted.\n"
        "Do not explicitly mention the physiological values to the user.\n"
        "Do not invent facts or physiological data.\n"
        "Write exactly one assistant reply."
    ),
    "adaptive_priority_physio": (
        "You are a helpful, natural, and emotionally adaptive assistant.\n"
        "Reply in a way that chooses the right balance between emotional attunement and practical help "
        "based on the user's message and any affect context provided.\n"
        "If the message is mainly task-focused or low-emotion, prioritize a direct, efficient, and helpful "
        "answer with only light emotional acknowledgement if needed.\n"
        "If the message is strongly emotional and not strongly task-focused, prioritize validation, "
        "steadiness, and one small useful next step.\n"
        "If both emotion and task matter, combine brief emotional acknowledgement with concrete, "
        "prioritized help.\n"
        "If physiological context is provided (such as heart rate or heart rate variability), treat it as "
        "a noisy indicator of the user's stress or arousal level. Use it only to adjust tone, pacing, "
        "validation, and priorities.\n"
        "Do not over-empathize task-heavy requests, and do not under-support strongly emotional requests.\n"
        "Do not explicitly mention the physiological values to the user.\n"
        "Do not invent facts or physiological data.\n"
        "Write exactly one assistant reply."
    ),
}

# Shared [instruction] block for all new-style prompts
_INSTRUCTION_BLOCK = (
    "[instruction]\n"
    "Write one natural assistant reply to the user.\n"
    "Be emotionally appropriate and helpful.\n"
    "Do not mention the metadata explicitly.\n"
    "Avoid sounding templated or overly dramatic."
)


# Generic system prompt - completely scenario-agnostic
GENERIC_SYSTEM_PROMPT = "You are a helpful conversational AI assistant."


# SAFETY_BLOCK kept for reference but NOT auto-injected
# Can be manually enabled via add_safety_block parameter if needed for other experiments
SAFETY_BLOCK = """You are interacting with a user who may be in crisis. Please follow these principles:

1. **Assess urgency**: First confirm if the user is in immediate danger and their location.
2. **Emergency resources**: If urgent, recommend immediately contacting:
   - Local emergency services (e.g., 110/120 in China, 911 in US)
   - Campus counseling center or student affairs hotline
   - 24-hour crisis intervention hotline
3. **Low-risk techniques**: Provide simple grounding techniques, such as:
   - Deep breathing exercises (4-7-8 breathing method)
   - Grounding techniques (5-4-3-2-1 sensory method)
   - Temporarily leave triggering environment
4. **Strict prohibitions**:
   - Do NOT provide any medical diagnosis
   - Do NOT give advice that could increase risk
   - Do NOT promise confidentiality when safety is involved
5. **Encourage help-seeking**: Respect the user's choices and privacy, encourage contacting trusted individuals (friends/family/teachers/professionals). Even if the user refuses referral, maintain supportive and understanding attitude.

Please respond with warmth, calmness, and non-judgmental tone."""


def _infer_task_type_from_goal(scenario: Scenario) -> str:
    """
    Lightweight hint only.
    If scenario.goal starts with T1/T2/T3/T4, return that; otherwise default to T1.
    """
    goal = getattr(scenario, "goal", None)
    goal = "" if goal is None else str(goal).strip()
    m = re.match(r"^(T[1-4])", goal, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return "T1"


def build_baseline_prompt(scenario: Scenario, add_safety_block: bool = False) -> List[Dict[str, str]]:
    """
    Baseline Prompt: User text only, no emotion information
    SCENARIO-AGNOSTIC: Generic system prompt, no domain/goal/risk information
    """
    system_msg = GENERIC_SYSTEM_PROMPT

    if add_safety_block:
        system_msg += "\n\n" + SAFETY_BLOCK

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": scenario.user_text},
    ]


def build_soft_prompt(
    scenario: Scenario,
    fused_affect: FusedAffect,
    affect_repr: Literal["VA", "DISCRETE", "PHYSIO"] = "VA",
    discrete_label: str = None,
    add_safety_block: bool = False
) -> List[Dict[str, str]]:
    """
    Soft Prompt - "Soft Emotion Conditioning"
    Provides estimated emotion as a CONTEXTUAL HINT only.
    SCENARIO-AGNOSTIC: Generic system prompt + affect hint, no domain/goal/risk.
    """
    system_msg = GENERIC_SYSTEM_PROMPT

    hint_intro = (
        "\n\nYou will see an estimated user emotion from multimodal signals. "
        "Treat it as noisy context; you may use it if it helps, but do not over-rely on it."
    )
    system_msg += hint_intro

    if add_safety_block:
        system_msg += "\n\n" + SAFETY_BLOCK

    if affect_repr == "VA":
        # VA hint (+ optional discrete mapping)
        try:
            from .fuse import va_to_discrete
            discrete_data = va_to_discrete(fused_affect.valence, fused_affect.arousal, fused_affect.confidence)
            label_hint = discrete_data.get("label", None)
        except Exception:
            label_hint = None

        emotion_block = f"""

[ESTIMATED_AFFECT]
repr: VA
valence: {fused_affect.valence:.2f}
arousal: {fused_affect.arousal:.2f}
scale: normalized_-1_to_1 (0=neutral)
confidence: {fused_affect.confidence:.2f}
label_hint: "{label_hint}"
"""
    elif affect_repr == "PHYSIO":
        emotion_block = f"""

[ESTIMATED_AFFECT]
repr: PHYSIO
hr_mean_bpm: {fused_affect.hr_mean_bpm}
hrv_sdnn_ms: {fused_affect.hrv_sdnn_ms}
hrv_rmssd_ms: {fused_affect.hrv_rmssd_ms}
hrv_pnn50_pct: {fused_affect.hrv_pnn50_pct}
scale: hr=bpm; hrv=ms/% (computed from RR intervals)
confidence: {fused_affect.confidence:.2f}
"""
    else:  # DISCRETE
        emotion_block = f"""

[ESTIMATED_AFFECT]
repr: DISCRETE
detected_emotion: "{discrete_label}"
confidence: {fused_affect.confidence:.2f}
source: multimodal sensors
"""

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": scenario.user_text + emotion_block},
    ]


def build_explicit_prompt(
    scenario: Scenario,
    fused_affect: FusedAffect,
    affect_repr: Literal["VA", "DISCRETE", "PHYSIO"] = "VA",
    discrete_label: str = None,
    add_safety_block: bool = False
) -> List[Dict[str, str]]:
    """
    Explicit Prompt - JSON control block with generic policy.
    Still avoids domain/risk details.
    Adds strict Signal Use + structured response to reduce evaluation ties.
    """

    task_type = _infer_task_type_from_goal(scenario)

    # --- NEW: stronger system message for explicit prompt ---
    system_msg = GENERIC_SYSTEM_PROMPT

    system_msg += (
        "\n\nYou are an interview-scenario coaching assistant. "
        "You MUST adapt your response using the provided signals (VA or physiology)."
        "\n\nCRITICAL REQUIREMENT (Signal Use):"
        "\n- You MUST explicitly use the provided signals to tailor your response."
        "\n- You MUST reference at least ONE signal value in your text (e.g., 'arousal is high (0.82)' "
        "or 'HR ≈ 128 bpm' or 'RMSSD ≈ 18 ms')."
        "\n- If you do not use signals, your response is incomplete."
        "\n\nSafety & boundaries:"
        "\n- Do NOT diagnose medical or mental health conditions."
        "\n- Do not shame, blame, or use harsh language."
        "\n- If the user expresses severe distress or risk of self-harm, encourage seeking professional help."
        "\n\nStyle:"
        "\n- Be concise and low cognitive load."
        "\n- Use short paragraphs and bullet points when helpful."
        "\n- Keep a calm, respectful tone."
    )

    if add_safety_block:
        system_msg += "\n\n" + SAFETY_BLOCK

    # Build affect dict for CONTROL block
    if affect_repr == "VA":
        affect_dict = {
            "repr": "VA",
            "valence": round(float(fused_affect.valence), 2),
            "arousal": round(float(fused_affect.arousal), 2),
            "scale": "normalized_-1_to_1 (0=neutral)",
            "confidence": round(float(fused_affect.confidence), 2),
        }

        # Make it easier for the model to cite at least one value
        signal_block = (
            "Self-reported VA (scale -1..1): "
            f"valence={affect_dict['valence']:+.2f}, arousal={affect_dict['arousal']:+.2f}. "
            "Use as a hint; do not over-interpret."
        )

    elif affect_repr == "PHYSIO":
        affect_dict = {
            "repr": "PHYSIO",
            "hr_mean_bpm": None if fused_affect.hr_mean_bpm is None else round(float(fused_affect.hr_mean_bpm), 1),
            "hrv_sdnn_ms": None if fused_affect.hrv_sdnn_ms is None else round(float(fused_affect.hrv_sdnn_ms), 1),
            "hrv_rmssd_ms": None if fused_affect.hrv_rmssd_ms is None else round(float(fused_affect.hrv_rmssd_ms), 1),
            "hrv_pnn50_pct": None if fused_affect.hrv_pnn50_pct is None else round(float(fused_affect.hrv_pnn50_pct), 1),
            "scale": "hr=bpm; hrv=ms/% (computed from RR intervals)",
            "confidence": round(float(fused_affect.confidence), 2),
        }

        parts = []
        for k in ["hr_mean_bpm", "hrv_rmssd_ms", "hrv_sdnn_ms", "hrv_pnn50_pct"]:
            if affect_dict.get(k) is not None:
                parts.append(f"{k}={affect_dict[k]}")
        signal_block = "Physiology: " + (", ".join(parts) if parts else "N/A") + "."

    else:  # DISCRETE
        affect_dict = {
            "repr": "DISCRETE",
            "label": discrete_label,
            "confidence": round(float(fused_affect.confidence), 2),
        }
        signal_block = f'Discrete emotion label: "{affect_dict.get("label")}" (confidence={affect_dict["confidence"]:.2f}).'

    # --- NEW: task-aware (lightweight) response guidance ---
    # This is NOT a rubric; it's a writing-mode hint to reduce ties and improve consistency.
    task_guidance = {
        "T1": "T1 In-the-moment coping: stabilize first (1–2 quick regulation techniques), then 1 practical step.",
        "T2": "T2 Debrief/Diagnosis: identify likely causes (2–4) and propose 2 concrete fixes; ask 1 reflective question.",
        "T3": "T3 Practice/Plan: propose a short plan (today/this week) and include a mini script or mock Q&A structure.",
        "T4": "T4 Emotional recovery: validate and reduce self-blame; rebuild efficacy; include 1 gentle next action.",
    }.get(task_type, "T1 In-the-moment coping: stabilize first.")

    # --- NEW: enforce structured output format ---
    output_format = (
        "Output format (STRICT):\n"
        "- Section 1: 'What I’m noticing' (2–4 bullets; MUST mention at least ONE signal value)\n"
        "- Section 2: 'What to do next' (3–6 bullets; MUST include at least 1 actionable regulation/support step)\n"
        "- Section 3: 'One quick question' (1 line)\n"
    )

    priorities = [
        "emotional_awareness",
        "signal_use_required",
        "contextual_appropriateness",
        "helpfulness",
        "safety_consciousness",
        "clarity_low_cognitive_load",
    ]

    required_behaviors = [
        "explicitly reference at least one provided signal value and use it to tailor advice",
        "acknowledge user's likely emotional state without over-interpreting",
        "provide actionable steps (not just reassurance), especially a regulation step when arousal/stress seems high",
        "keep response structured and low cognitive load",
        "avoid medical diagnosis or guarantees beyond capability",
        "maintain respectful tone and safety-conscious language",
        "end with one short question to personalize follow-up",
    ]

    control = {
        "task_type_hint": task_type,
        "task_guidance_hint": task_guidance,
        "signals_summary": signal_block,
        "affect": affect_dict,
        "policy": {
            "priorities": priorities,
            "required_behaviors": required_behaviors,
            "output_format": output_format,
        },
    }

    control_block = f"\n\n<CONTROL>\n{json.dumps(control, ensure_ascii=False, indent=2)}\n</CONTROL>"
    system_msg += control_block

    # user message stays the raw user text (so generation stays mostly scenario-agnostic)
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": scenario.user_text},
    ]


def build_prompt(
    prompt_type: Literal["baseline", "soft", "explicit"],
    scenario: Scenario,
    fused_affect: Optional[FusedAffect] = None,
    affect_repr: Literal["VA", "DISCRETE", "PHYSIO"] = "VA",
    discrete_label: str = None,
    add_safety_block: bool = False
) -> List[Dict[str, str]]:
    """
    Unified entry: Build messages based on prompt_type
    """
    if prompt_type == "baseline":
        return build_baseline_prompt(scenario, add_safety_block=add_safety_block)

    if fused_affect is None:
        raise ValueError(f"{prompt_type} prompt requires fused_affect")

    if prompt_type == "soft":
        return build_soft_prompt(scenario, fused_affect, affect_repr, discrete_label, add_safety_block=add_safety_block)

    if prompt_type == "explicit":
        return build_explicit_prompt(scenario, fused_affect, affect_repr, discrete_label, add_safety_block=add_safety_block)

    raise ValueError(f"Unknown prompt_type: {prompt_type}")


# ---------------------------------------------------------------------------
# New-style strategy prompts  (minimal_adaptive / empathy_then_help / adaptive_priority)
# ---------------------------------------------------------------------------

def _build_affect_context_block(
    affect_repr: str,
    fused_affect: FusedAffect,
    discrete_label: Optional[str] = None,
) -> str:
    """Return the [affect_context] section for the new user-prompt template."""
    r = affect_repr.upper()
    if r == "VA":
        return (
            "[affect_context]\n"
            f"mode: va\n"
            f"valence: {fused_affect.valence:.2f}\n"
            f"arousal: {fused_affect.arousal:.2f}"
        )
    if r == "AROUSAL_ONLY":
        return (
            "[affect_context]\n"
            f"mode: arousal_only\n"
            f"arousal: {fused_affect.arousal:.2f}"
        )
    if r == "PHYSIO":
        hr    = fused_affect.hr_mean_bpm
        rmssd = fused_affect.hrv_rmssd_ms
        sdnn  = fused_affect.hrv_sdnn_ms
        pnn50 = fused_affect.hrv_pnn50_pct
        return (
            "[affect_context]\n"
            f"mode: physio\n"
            f"hr_mean_bpm: {hr}\n"
            f"hrv_rmssd_ms: {rmssd}\n"
            f"hrv_sdnn_ms: {sdnn}\n"
            f"hrv_pnn50_pct: {pnn50}"
        )
    if r == "DISCRETE":
        return (
            "[affect_context]\n"
            f"mode: discrete\n"
            f"emotion: {discrete_label}"
        )
    raise ValueError(f"Unknown affect_repr: {affect_repr}")


def build_strategy_prompt(
    strategy: str,
    scenario: Scenario,
    fused_affect: FusedAffect,
    affect_repr: Literal["VA", "AROUSAL_ONLY", "PHYSIO", "DISCRETE"] = "VA",
    discrete_label: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Build messages using the new [affect_context]+[user_message]+[instruction] format.

    strategy must be one of: minimal_adaptive, empathy_then_help, adaptive_priority
    (the _physio suffix is selected automatically when affect_repr == "PHYSIO").

    Returns a 2-element list: [system_message, user_message].
    """
    # Resolve the correct system-prompt key
    base_strategy = strategy.removesuffix("_physio")
    if base_strategy not in ("minimal_adaptive", "empathy_then_help", "adaptive_priority"):
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            "Valid base strategies: minimal_adaptive, empathy_then_help, adaptive_priority"
        )

    prompt_key = (
        f"{base_strategy}_physio"
        if affect_repr.upper() == "PHYSIO"
        else base_strategy
    )
    if prompt_key not in SYSTEM_PROMPTS:
        raise KeyError(f"System prompt key '{prompt_key}' not found in SYSTEM_PROMPTS")

    system_content = SYSTEM_PROMPTS[prompt_key]

    # Build user message
    affect_ctx = _build_affect_context_block(affect_repr, fused_affect, discrete_label)
    user_content = (
        f"{affect_ctx}\n\n"
        f"[user_message]\n"
        f"{scenario.user_text}\n\n"
        f"{_INSTRUCTION_BLOCK}"
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": user_content},
    ]
