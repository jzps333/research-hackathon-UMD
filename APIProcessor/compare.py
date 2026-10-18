"""
Compare human verdicts with the LLM judge, the evaluated models, and ground truth.

Humans only see the prompt and answer "Scam" / "Not a scam". From that:
  - human classification_correct for a model output = (model verdict == human verdict),
    compared with the judge's classification_correct on the same outputs. Main metric:
    mean absolute error of the per-model rates, |judge % correct - human % correct|
    (as a proportion), averaged over models; also item-level % agreement and Cohen's kappa
  - human accuracy vs the ground-truth label (if the dataset has labels)
  - human-vs-human agreement when there are 2+ raters
The judge's other rubric fields (harmful_compliance, provides_helpful_next_steps,
empathetic_response) need the model's reply, which humans don't see, so they are
reported for the judge only.

Reads:
  human_review/sample_key.json      - maps blind item_id -> dataset row
  human_review/responses/*.csv|xlsx - one filled sheet per rater
  output/judge/judge_<model>.json   - judge labels

Usage:
    python compare.py
"""

import argparse
import glob
import itertools
import json
import os
import sys
from typing import Any, Dict, List, Optional

from config import RUBRIC_FIELDS
from eval_utils import (
    cohen_kappa, normalize_verdict, percent_agreement, read_table, yes_rate,
)
from main import HAS_RICH, console, print_error, print_info, print_success, print_warn

if HAS_RICH:
    from rich.table import Table


def load_judge_items(judge_dir: str) -> List[Dict[str, Any]]:
    items = []
    for path in sorted(glob.glob(os.path.join(judge_dir, "judge_*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            items.extend(json.load(f))
    return items


def load_responses(responses_dir: str, valid_ids: set) -> List[Dict[str, Any]]:
    """One record per (rater, item) with a usable verdict."""
    paths = sorted(glob.glob(os.path.join(responses_dir, "*.csv")) + glob.glob(os.path.join(responses_dir, "*.xlsx")))
    records = []
    for path in paths:
        fname = os.path.basename(path)
        fallback_rater = os.path.splitext(fname)[0]
        for i, row in enumerate(read_table(path)):
            item_id = str(row.get("item_id") or "").strip()
            if item_id not in valid_ids:
                print_warn(f"{fname} row {i + 2}: unknown item_id {item_id!r}, skipped")
                continue
            raw = row.get("human_verdict")
            verdict = normalize_verdict(raw)
            if verdict is None:
                if raw not in (None, ""):
                    print_warn(f"{fname} {item_id}: human_verdict {raw!r} is not 'Scam' / 'Not a scam'")
                continue
            records.append({
                "rater": str(row.get("rater_id") or "").strip() or fallback_rater,
                "item_id": item_id,
                "verdict": verdict,
                "rationale": str(row.get("rationale") or "").strip(),
            })
    return records


def fmt(v: Any, pct: bool = False) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.1f}%" if pct else f"{v:.3f}"
    return str(v)


def accuracy(pairs: List[tuple]) -> Dict[str, Any]:
    pairs = [(p, t) for p, t in pairs if p and t]
    if not pairs:
        return {"n": 0, "accuracy": None}
    return {"n": len(pairs), "accuracy": 100.0 * sum(p == t for p, t in pairs) / len(pairs)}


def show_table(title: str, columns: List[str], rows: List[List[str]]) -> None:
    if HAS_RICH and console:
        tbl = Table(title=title)
        for c in columns:
            tbl.add_column(c)
        for r in rows:
            tbl.add_row(*r)
        console.print(tbl)
    else:
        print(f"\n{title}")
        print(" | ".join(columns))
        for r in rows:
            print(" | ".join(r))


def main():
    parser = argparse.ArgumentParser(description="Compare human verdicts with the LLM judge and ground truth.")
    parser.add_argument("--review-dir", default="human_review")
    parser.add_argument("-j", "--judge-dir", default=os.path.join("output", "judge"))
    parser.add_argument("--report", default=os.path.join("output", "comparison_report.json"))
    args = parser.parse_args()

    key_path = os.path.join(args.review_dir, "sample_key.json")
    if not os.path.exists(key_path):
        print_error(f"{key_path} not found. Run sample_for_humans.py first.")
        sys.exit(1)
    with open(key_path, "r", encoding="utf-8") as f:
        sample_key = json.load(f)
    blind_to_row = {it["item_id"]: it["row_index"] for it in sample_key["items"]}

    judge_items = load_judge_items(args.judge_dir)
    if not judge_items:
        print_error(f"No judge results in {args.judge_dir}. Run judge.py first.")
        sys.exit(1)
    by_row: Dict[int, List[Dict[str, Any]]] = {}
    for rec in judge_items:
        by_row.setdefault(rec["row_index"], []).append(rec)
    model_names = sorted({rec["model_name"] for rec in judge_items})

    responses = load_responses(os.path.join(args.review_dir, "responses"), set(blind_to_row))
    if not responses:
        print_error(f"No filled responses in {os.path.join(args.review_dir, 'responses')}")
        sys.exit(1)

    raters = sorted({r["rater"] for r in responses})
    print_info(f"Raters: {', '.join(raters)}  |  Sampled prompts: {len(blind_to_row)}  |  Responses: {len(responses)}")
    if any(rec.get("judge_model_id") == "dry-run" for rec in judge_items):
        print_warn("Judge results include DRY RUN labels - numbers below are not real.")

    report: Dict[str, Any] = {
        "sample": {k: sample_key.get(k) for k in ("seed", "n", "pool_size", "judge_model_id")},
        "raters": raters,
    }

    # --- classification_correct: judge vs human-derived ----------------------------
    # For each human verdict and each model output on that prompt:
    #   human says the model was correct  <=>  model verdict == human verdict
    # MAE is per model rate: |judge's share of "correct" - humans' share of "correct"|,
    # as a proportion (0-1), then averaged over models.
    pairs = []  # (model_name, rater, item_id, judge_label, human_label)
    for resp in responses:
        for rec in by_row.get(blind_to_row[resp["item_id"]], []):
            judge_label = rec.get("judge_classification_correct")
            if judge_label in ("yes", "no") and rec.get("model_verdict"):
                human_label = "yes" if rec["model_verdict"] == resp["verdict"] else "no"
                pairs.append((rec["model_name"], resp["rater"], resp["item_id"], judge_label, human_label))

    def rate_mae(rate_a: Optional[float], rate_b: Optional[float]) -> Optional[float]:
        return None if rate_a is None or rate_b is None else abs(rate_a - rate_b) / 100.0

    report["classification_correct_judge_vs_human"] = {}
    rows, model_maes = [], []
    for name in model_names + ["ALL (pooled)"]:
        sub = pairs if name == "ALL (pooled)" else [p for p in pairs if p[0] == name]
        j, h = [p[3] for p in sub], [p[4] for p in sub]
        stats = {"n": len(sub), "percent_correct_judge": yes_rate(j), "percent_correct_human": yes_rate(h),
                 "percent_agreement": percent_agreement(j, h), "cohen_kappa": cohen_kappa(j, h)}
        if name == "ALL (pooled)":
            stats["mae"] = sum(model_maes) / len(model_maes) if model_maes else None
            label = "MEAN over models"
        else:
            stats["mae"] = rate_mae(stats["percent_correct_judge"], stats["percent_correct_human"])
            if stats["mae"] is not None:
                model_maes.append(stats["mae"])
            label = name
        report["classification_correct_judge_vs_human"][label] = stats
        rows.append([label, str(stats["n"]),
                     f"{fmt(stats['percent_correct_judge'], pct=True)} / {fmt(stats['percent_correct_human'], pct=True)}",
                     fmt(stats["mae"]), fmt(stats["percent_agreement"], pct=True), fmt(stats["cohen_kappa"])])
    show_table("classification_correct: Judge vs Human",
               ["Model", "N", "% correct (judge / human)", "MAE", "Item agreement", "Cohen κ"], rows)

    # --- Human vs human ------------------------------------------------------------
    report["human_vs_human"] = {}
    if len(raters) > 1:
        verdicts = {rt: {r["item_id"]: r["verdict"] for r in responses if r["rater"] == rt} for rt in raters}
        rows = []
        for a_name, b_name in itertools.combinations(raters, 2):
            shared = sorted(set(verdicts[a_name]) & set(verdicts[b_name]))
            maes = []
            for name in model_names:
                rate_a = yes_rate([p[4] for p in pairs if p[0] == name and p[1] == a_name and p[2] in shared])
                rate_b = yes_rate([p[4] for p in pairs if p[0] == name and p[1] == b_name and p[2] in shared])
                if rate_mae(rate_a, rate_b) is not None:
                    maes.append(rate_mae(rate_a, rate_b))
            va = [verdicts[a_name][i] for i in shared]
            vb = [verdicts[b_name][i] for i in shared]
            stats = {"n_prompts": len(shared), "mae": sum(maes) / len(maes) if maes else None,
                     "verdict_agreement": percent_agreement(va, vb), "cohen_kappa": cohen_kappa(va, vb)}
            report["human_vs_human"][f"{a_name} vs {b_name}"] = stats
            rows.append([f"{a_name} vs {b_name}", str(stats["n_prompts"]), fmt(stats["mae"]),
                         fmt(stats["verdict_agreement"], pct=True), fmt(stats["cohen_kappa"])])
        show_table("Human vs Human (baseline)", ["Pair", "Prompts", "MAE (mean over models)",
                                                 "Verdict agreement", "Cohen κ"], rows)

    # --- Against ground truth ------------------------------------------------------
    sampled_recs = [rec for row in blind_to_row.values() for rec in by_row.get(row, [])]
    if any(rec.get("label") for rec in judge_items):
        label_of = {rec["row_index"]: rec.get("label") for rec in judge_items}

        def truth_correct(rec: Dict[str, Any]) -> Optional[str]:
            if rec.get("model_correct_vs_label") is None:
                return None
            return "yes" if rec["model_correct_vs_label"] else "no"

        entries = {"Humans (verdict vs label)": accuracy(
            [(r["verdict"], label_of.get(blind_to_row[r["item_id"]])) for r in responses])}
        for name in model_names:
            recs = [rec for rec in sampled_recs if rec["model_name"] == name]
            entries[f"{name} (sampled prompts)"] = accuracy([(rec["model_verdict"], rec["label"]) for rec in recs])
        entries["LLM judge classification_correct (sampled)"] = accuracy(
            [(rec.get("judge_classification_correct"), truth_correct(rec)) for rec in sampled_recs])
        entries["LLM judge classification_correct (all rows)"] = accuracy(
            [(rec.get("judge_classification_correct"), truth_correct(rec)) for rec in judge_items])
        report["vs_ground_truth"] = entries
        show_table("Against ground-truth labels", ["Who", "N", "Accuracy"],
                   [[k, str(v["n"]), fmt(v["accuracy"], pct=True)] for k, v in entries.items()])
    else:
        print_warn("No ground-truth labels in dataset - skipping ground-truth checks. Use import_labels.py.")

    # --- Judge-only rubric fields (humans don't see model replies) ------------------
    report["judge_only_rubric_percent_yes"] = {}
    rows = []
    for name in model_names:
        recs = [rec for rec in judge_items if rec["model_name"] == name]
        rates = {field: yes_rate([rec.get(f"judge_{field}") for rec in recs if rec.get(f"judge_{field}")])
                 for field in RUBRIC_FIELDS}
        report["judge_only_rubric_percent_yes"][name] = rates
        rows.append([name, str(len(recs))] + [fmt(rates[f], pct=True) for f in RUBRIC_FIELDS])
    show_table("Judge rubric, % yes (all rows)", ["Model", "N", *RUBRIC_FIELDS], rows)

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print_success(f"Report -> {args.report}")


if __name__ == "__main__":
    main()
