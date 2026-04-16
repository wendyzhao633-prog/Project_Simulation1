from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, UTC
from typing import Any, Dict, List, Tuple

import pandas as pd
import typer

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

from src.judge_prompt_builder import build_pairwise_judge_messages, load_rubrics_markdown
from src.llm_providers.openai_provider import OpenAIProvider
from src.utils import get_env_or_raise


def load_jsonl(path: str) -> List[Dict]:
    logs: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                logs.append(json.loads(line))
    return logs


def save_csv(rows: List[Dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved to: {out_path}")


def get_scenario(log: Dict) -> Dict:
    scen = log.get("scenario", {})
    return scen if isinstance(scen, dict) else {}


def get_scenario_id(log: Dict) -> str:
    scen = get_scenario(log)
    if scen.get("scenario_id"):
        return str(scen["scenario_id"])
    meta = log.get("meta", {})
    if isinstance(meta, dict) and meta.get("scenario_id"):
        return str(meta["scenario_id"])
    return "unknown"


def get_prompt_key(log: Dict) -> str:
    return str(log.get("prompt_type", "unknown"))


def get_modality_key(log: Dict) -> str:
    return log.get("modality_label") or "DEFAULT"


def get_representation_key(log: Dict) -> str:
    if log.get("affect_repr"):
        return str(log["affect_repr"])
    meta = log.get("meta") if isinstance(log.get("meta"), dict) else {}
    if meta.get("condition"):
        c = str(meta["condition"]).upper()
        return {"A_ONLY": "AROUSAL_ONLY", "AROUSAL": "AROUSAL_ONLY"}.get(c, c)
    return "UNKNOWN"


def get_subject_trial_key(log: Dict) -> Tuple[str, str]:
    meta = log.get("meta") if isinstance(log.get("meta"), dict) else {}
    subj = meta.get("subject")
    trial = meta.get("trial")
    return (str(subj) if subj is not None else "NA", str(trial) if trial is not None else "NA")


def normalize_log(log: Dict) -> Dict:
    if not isinstance(log, dict):
        return log
    meta = log.get("meta") if isinstance(log.get("meta"), dict) else {}
    log.setdefault("modality_label", "DEFAULT")
    if not log.get("affect_repr") and meta.get("condition"):
        log["affect_repr"] = meta["condition"]
    if ("scenario" not in log or not isinstance(log.get("scenario"), dict)) and meta.get("scenario_id"):
        log["scenario"] = {"scenario_id": meta.get("scenario_id")}
    return log


def _extract_response_text(log: Dict) -> str:
    resp = log.get("response")
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        if "text" in resp:
            return str(resp["text"])
        if "content" in resp:
            return str(resp["content"])
    if "output" in log:
        return str(log["output"])
    return ""


def _extract_user_text(log: Dict) -> str:
    scen = get_scenario(log)
    if scen.get("user_text"):
        return str(scen["user_text"])
    msgs = log.get("messages", [])
    if not isinstance(msgs, list):
        return ""
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "user":
            return str(m.get("content", ""))
    return ""


def _summarize_signals(log: Dict) -> str:
    inputs = log.get("inputs", {})
    if not isinstance(inputs, dict):
        inputs = {}

    sr = inputs.get("self_report_noisy")
    if isinstance(sr, dict) and ("valence" in sr or "arousal" in sr):
        return f"Self-report signal: {sr}"

    ph = inputs.get("physio")
    if isinstance(ph, dict):
        return f"Physiology signal: {ph}"

    return f"Signal representation: {get_representation_key(log)}"


def _extract_context_fields(log: Dict) -> Dict[str, Any]:
    scen = get_scenario(log)
    meta = log.get("meta") if isinstance(log.get("meta"), dict) else {}
    inputs = log.get("inputs") if isinstance(log.get("inputs"), dict) else {}
    sr_true = inputs.get("self_report_true") if isinstance(inputs.get("self_report_true"), dict) else {}

    def _num(v):
        try:
            return float(v)
        except Exception:
            return None

    return {
        "source_affect_id": meta.get("source_affect_id"),
        "emotion_label": meta.get("emotion_label"),
        "risk_tier": meta.get("risk_tier") or scen.get("risk_tier"),
        "valence": _num(sr_true.get("valence")),
        "arousal": _num(sr_true.get("arousal")),
        "alignment_matched_by": meta.get("alignment_matched_by"),
        "alignment_match_ok": meta.get("alignment_match_ok"),
    }


def _make_judge_prompt_newrubric(a: Dict, b: Dict, rubrics_md: str) -> List[Dict[str, str]]:
    return build_pairwise_judge_messages(
        rubrics_md=rubrics_md,
        user_text=_extract_user_text(a),
        response_a=_extract_response_text(a),
        response_b=_extract_response_text(b),
        signal_a=_summarize_signals(a),
        signal_b=_summarize_signals(b),
    )


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    raise ValueError(f"Could not parse judge JSON. Raw output:\n{text[:2000]}")


def _coerce_scores(obj: Dict[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for d in ["D1", "D2", "D3", "D4", "D5", "D6"]:
        v = obj.get(d, 1)
        try:
            v = int(v)
        except Exception:
            v = 1
        out[d] = max(1, min(5, v))
    return out


def _to_jsonable(x: Any) -> Any:
    if hasattr(x, "model_dump"):
        try:
            return x.model_dump()
        except Exception:
            pass
    if hasattr(x, "dict"):
        try:
            return x.dict()
        except Exception:
            pass
    if isinstance(x, (dict, list, str, int, float, bool)) or x is None:
        return x
    return str(x)


def _s_map(score_1_to_5: int) -> float:
    return ((score_1_to_5 - 1) / 4.0) * 100.0


def overall_from_d1_to_d5(scores: Dict[str, int]) -> float:
    vals = [_s_map(int(scores[d])) for d in ["D1", "D2", "D3", "D4", "D5"]]
    return round(sum(vals) / len(vals), 2)


def winner_overall_from_d1_to_d5(overall_a: float, overall_b: float, tie_margin: float) -> str:
    diff = overall_b - overall_a
    if abs(diff) < tie_margin:
        return "tie"
    return "B" if diff > 0 else "A"


def winner_d6(a_scores: Dict[str, int], b_scores: Dict[str, int], d6_tie_margin: int = 0) -> str:
    diff = int(b_scores["D6"]) - int(a_scores["D6"])
    if abs(diff) <= int(d6_tie_margin):
        return "tie"
    return "B" if diff > 0 else "A"


def run_coro(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_box = {}
    error_box = {}

    def _thread_runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result_box["result"] = loop.run_until_complete(coro)
        except Exception as e:
            error_box["error"] = e
        finally:
            loop.close()

    t = threading.Thread(target=_thread_runner, daemon=True)
    t.start()
    t.join()
    if "error" in error_box:
        raise error_box["error"]
    return result_box.get("result")


app = typer.Typer(add_completion=False)


@dataclass
class Pair:
    a: Dict
    b: Dict


def _group_for_representation_comparison(logs: List[Dict]) -> Dict[Tuple[str, str, str, str, str], List[Dict]]:
    groups = defaultdict(list)
    for log in logs:
        key = (
            get_scenario_id(log),
            get_modality_key(log),
            get_prompt_key(log),
            get_subject_trial_key(log)[0],
            get_subject_trial_key(log)[1],
        )
        groups[key].append(log)
    return groups


def _make_pairs_for_representation(
    groups: Dict[Tuple[str, str, str, str, str], List[Dict]],
    a_repr: str = "VA",
    b_repr: str = "PHYSIO",
) -> List[Pair]:
    pairs: List[Pair] = []
    a_repr_u = a_repr.upper().strip()
    b_repr_u = b_repr.upper().strip()
    skipped_groups = 0

    for _, items in groups.items():
        by_repr = defaultdict(list)
        for log in items:
            by_repr[str(get_representation_key(log)).upper().strip()].append(log)
        if len(by_repr.keys()) < 2:
            skipped_groups += 1
            continue

        if a_repr_u in by_repr and b_repr_u in by_repr:
            a_list = by_repr[a_repr_u]
            b_list = by_repr[b_repr_u]
        else:
            reprs = sorted(by_repr.keys())
            a_list = by_repr[reprs[0]]
            b_list = by_repr[reprs[1]]

        for i in range(min(len(a_list), len(b_list))):
            pairs.append(Pair(a=a_list[i], b=b_list[i]))

    if skipped_groups:
        print(f"Skipped {skipped_groups} groups with <2 representations")
    return pairs


async def _score_pairs_async(
    pairs: List[Pair],
    judge_model: str,
    concurrency: int,
    api_key: str,
    tie_margin: float,
    d6_tie_margin: int,
    comparison_label: str,
    rubrics_md: str,
) -> List[Dict]:
    provider = OpenAIProvider(model=judge_model, api_key=api_key)
    sem = asyncio.Semaphore(concurrency)
    out_rows: List[Dict] = []

    async def run_one(pair: Pair):
        async with sem:
            msgs = _make_judge_prompt_newrubric(pair.a, pair.b, rubrics_md=rubrics_md)
            result = await asyncio.to_thread(
                provider.generate,
                msgs,
                {"temperature": 0.0, "top_p": 1.0, "max_tokens": 900},
            )
            result_j = _to_jsonable(result)
            raw_text = str(result_j.get("text") if isinstance(result_j, dict) else result_j)
            parsed = _extract_json_from_text(raw_text)
            a_scores = _coerce_scores(parsed.get("A", {}))
            b_scores = _coerce_scores(parsed.get("B", {}))

            overall_a = overall_from_d1_to_d5(a_scores)
            overall_b = overall_from_d1_to_d5(b_scores)
            delta_overall = round(overall_b - overall_a, 2)
            winner_overall = winner_overall_from_d1_to_d5(overall_a, overall_b, tie_margin=tie_margin)
            winner_d6_label = winner_d6(a_scores, b_scores, d6_tie_margin=d6_tie_margin)
            d6_delta = int(b_scores["D6"]) - int(a_scores["D6"])

            ctx = _extract_context_fields(pair.a)
            row = {
                "comparison": comparison_label,
                "scenario_id": get_scenario_id(pair.a),
                "subject": get_subject_trial_key(pair.a)[0],
                "trial": get_subject_trial_key(pair.a)[1],
                "prompt_type": get_prompt_key(pair.a),
                "modality_label": get_modality_key(pair.a),
                "repr_a": str(get_representation_key(pair.a)).upper(),
                "repr_b": str(get_representation_key(pair.b)).upper(),
                "run_id_a": pair.a.get("run_id"),
                "run_id_b": pair.b.get("run_id"),
                **{f"A_{d}": a_scores[d] for d in ["D1", "D2", "D3", "D4", "D5", "D6"]},
                **{f"B_{d}": b_scores[d] for d in ["D1", "D2", "D3", "D4", "D5", "D6"]},
                "overall_a": overall_a,
                "overall_b": overall_b,
                "delta_overall": delta_overall,
                "winner_overall": winner_overall,
                "winner_d6": winner_d6_label,
                "d6_delta": d6_delta,
                # backward-compatible winner column
                "winner": winner_overall,
                **ctx,
                "judge_raw": raw_text[:3000],
                "scored_at_utc": datetime.now(UTC).isoformat(),
            }
            out_rows.append(row)

    await asyncio.gather(*[asyncio.create_task(run_one(p)) for p in pairs])
    return out_rows


def _run_multi_comparisons(
    *,
    runs_jsonl: str,
    out_csv: str,
    comparisons: List[str],
    judge_model: str,
    concurrency: int,
    tie_margin: float,
    d6_tie_margin: int,
    rubrics_path: str,
) -> None:
    if load_dotenv:
        load_dotenv()
    rubrics_md = load_rubrics_markdown(rubrics_path)
    api_key = get_env_or_raise("OPENAI_API_KEY")
    logs = [normalize_log(l) for l in load_jsonl(runs_jsonl)]
    groups = _group_for_representation_comparison(logs)
    rows_all: List[Dict[str, Any]] = []

    for comp in comparisons:
        if ":" not in comp:
            raise ValueError(f"Invalid comparison '{comp}', expected A:B")
        a_repr, b_repr = [x.strip().upper() for x in comp.split(":", 1)]
        pairs = _make_pairs_for_representation(groups, a_repr=a_repr, b_repr=b_repr)
        print(f"[INFO] Comparison {a_repr}:{b_repr} -> pairs={len(pairs)}")
        if not pairs:
            continue
        rows = run_coro(
            _score_pairs_async(
                pairs=pairs,
                judge_model=judge_model,
                concurrency=concurrency,
                api_key=api_key,
                tie_margin=tie_margin,
                d6_tie_margin=d6_tie_margin,
                comparison_label=f"{a_repr}:{b_repr}",
                rubrics_md=rubrics_md,
            )
        )
        rows_all.extend(rows)

    save_csv(rows_all, out_csv)
    print(f"[OK] Completed total scores: {len(rows_all)}")


@app.command()
def score(
    input: str = typer.Option(..., help="Input jsonl logs"),
    output: str = typer.Option(..., help="Output CSV path"),
    mode: str = typer.Option("pairwise_representation"),
    judge_model: str = typer.Option("gpt-5.4-2026-03-05"),
    concurrency: int = typer.Option(10),
    a_repr: str = typer.Option("VA"),
    b_repr: str = typer.Option("PHYSIO"),
    tie_margin: float = typer.Option(3.0),
    d6_tie_margin: int = typer.Option(0),
    rubrics_md: str = typer.Option("RUBRICS.md"),
):
    if mode != "pairwise_representation":
        raise ValueError("This version supports only mode=pairwise_representation")
    _run_multi_comparisons(
        runs_jsonl=input,
        out_csv=output,
        comparisons=[f"{a_repr}:{b_repr}"],
        judge_model=judge_model,
        concurrency=concurrency,
        tie_margin=tie_margin,
        d6_tie_margin=d6_tie_margin,
        rubrics_path=rubrics_md,
    )


def _legacy_main_from_args() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_jsonl", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--comparisons", nargs="+", default=["VA:PHYSIO", "AROUSAL_ONLY:PHYSIO"])
    parser.add_argument("--judge_model", default="gpt-5.4-2026-03-05")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--tie_margin", type=float, default=3.0)
    parser.add_argument("--d6_tie_margin", type=int, default=0)
    parser.add_argument("--rubrics_md", default="RUBRICS.md")
    args = parser.parse_args()

    _run_multi_comparisons(
        runs_jsonl=args.runs_jsonl,
        out_csv=args.out_csv,
        comparisons=args.comparisons,
        judge_model=args.judge_model,
        concurrency=args.concurrency,
        tie_margin=args.tie_margin,
        d6_tie_margin=args.d6_tie_margin,
        rubrics_path=args.rubrics_md,
    )


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--runs_jsonl" in argv or "--out_csv" in argv:
        _legacy_main_from_args()
    else:
        app()
