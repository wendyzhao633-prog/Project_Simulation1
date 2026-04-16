from __future__ import annotations

import os
from typing import Dict, List


DEFAULT_RUBRICS_PATH = "RUBRICS.md"


def load_rubrics_markdown(path: str = DEFAULT_RUBRICS_PATH) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Rubric file not found: {path}. "
            "Please place your uploaded RUBRICS.md in the project root."
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_pairwise_judge_messages(
    *,
    rubrics_md: str,
    user_text: str,
    response_a: str,
    response_b: str,
    signal_a: str,
    signal_b: str,
) -> List[Dict[str, str]]:
    """
    Prompt style:
    - blind pairwise evaluator
    - strict JSON only
    - D1-D5 quality, D6 mechanism
    - winner_overall should be based on D1-D5 only (computed in Python side)
    """
    system = (
        "You are a blind pairwise evaluator for assistant responses.\n"
        "Return STRICT JSON only, no markdown, no extra keys.\n"
        "Do not reward verbosity by itself.\n"
        "Do not reward explicit emotion words alone.\n"
        "Do not punish concise but well-calibrated responses."
    )

    schema = (
        "{\n"
        '  "A": {"D1":1-5,"D2":1-5,"D3":1-5,"D4":1-5,"D5":1-5,"D6":1-5,\n'
        '        "rationale":{"D1":"...","D2":"...","D3":"...","D4":"...","D5":"...","D6":"..."}},\n'
        '  "B": {"D1":1-5,"D2":1-5,"D3":1-5,"D4":1-5,"D5":1-5,"D6":1-5,\n'
        '        "rationale":{"D1":"...","D2":"...","D3":"...","D4":"...","D5":"...","D6":"..."}}\n'
        "}"
    )

    user = (
        "RUBRIC SOURCE OF TRUTH (markdown):\n"
        f"{rubrics_md}\n\n"
        "Evaluation instructions:\n"
        "- Score A and B independently on D1..D6 using the rubric.\n"
        "- Keep D6 as mechanism signal-use dimension.\n"
        "- Do NOT compute winner_overall in JSON; numeric scores only.\n"
        "- Do NOT compute any weighted total in JSON.\n"
        "- Be strict and calibration-aware.\n\n"
        f"USER MESSAGE:\n{user_text}\n\n"
        f"Signals for A:\n{signal_a}\n\n"
        f"Signals for B:\n{signal_b}\n\n"
        f"RESPONSE A:\n{response_a}\n\n"
        f"RESPONSE B:\n{response_b}\n\n"
        f"Output JSON schema:\n{schema}\n"
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
