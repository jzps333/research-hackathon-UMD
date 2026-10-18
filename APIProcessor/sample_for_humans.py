"""
Randomly sample prompts for human raters.

Humans see ONLY the prompt (no context tag, no model output, no label) and give their
own verdict: is this message a scam? compare.py then checks their verdicts against the
ground-truth label and against each model, and compares them with the LLM judge's
classification_correct.

Writes (in --out-dir, default human_review/):
  human_sheet.csv   - grading sheet for raters: item_id, prompt + blank answer columns
  sample_key.json   - which dataset row each item_id is + the seed (do NOT give to raters)
  INSTRUCTIONS.md   - rater instructions

Each rater fills a copy of human_sheet.csv and saves it as
human_review/responses/<rater_name>.csv (or .xlsx).

Usage:
    python sample_for_humans.py --seed 42            # 30% of prompts
    python sample_for_humans.py -n 10 --seed 42      # exact count
"""

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime, timezone

from config import JUDGE_CONFIG
from eval_utils import write_csv
from main import print_error, print_info, print_success, print_warn

SHEET_COLUMNS = ["item_id", "prompt", "human_verdict", "rationale", "rater_id"]

INSTRUCTIONS = """# Human Rater Instructions

You will read {n} messages. For each one, decide whether it is a scam.

Do not look anything up or discuss items with other raters. Answer each row on your own.

For every row, fill in:
- **human_verdict**: exactly `Scam` or `Not a scam`
- **rationale**: one or two sentences on why
- **rater_id**: your name or initials (same on every row)

Save your file as `responses/<your_name>.csv` (or .xlsx) in this folder. Do not change
`item_id` or reorder or delete rows.
"""


def main():
    parser = argparse.ArgumentParser(description="Sample prompts into a blind human grading sheet.")
    parser.add_argument("-j", "--judge-dir", default=os.path.join("output", "judge"))
    parser.add_argument("--fraction", type=float, default=0.30,
                        help="Share of prompts to sample, rounded up (default: 0.30 = 30%%)")
    parser.add_argument("-n", "--num", type=int, default=None, help="Exact number of prompts (overrides --fraction)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (record this for the paper)")
    parser.add_argument("--out-dir", default="human_review")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing sheet")
    args = parser.parse_args()

    sheet_path = os.path.join(args.out_dir, "human_sheet.csv")
    key_path = os.path.join(args.out_dir, "sample_key.json")
    if os.path.exists(sheet_path) and not args.force:
        print_error(f"{sheet_path} already exists. Use --force to resample (this changes the items).")
        sys.exit(1)

    # One entry per dataset row that has been judged for at least one model
    prompts = {}
    for name in sorted(os.listdir(args.judge_dir)) if os.path.isdir(args.judge_dir) else []:
        if name.startswith("judge_") and name.endswith(".json"):
            with open(os.path.join(args.judge_dir, name), "r", encoding="utf-8") as f:
                for rec in json.load(f):
                    prompts.setdefault(rec["row_index"], rec["prompt"])

    if not prompts:
        print_error("No judged items found. Run judge.py first.")
        sys.exit(1)

    n = args.num if args.num is not None else math.ceil(args.fraction * len(prompts))
    if n > len(prompts):
        print_warn(f"Requested {n} prompts but only {len(prompts)} judged; using all.")
        n = len(prompts)

    rng = random.Random(args.seed)
    sampled_rows = rng.sample(sorted(prompts), n)

    sheet_rows, key_items = [], []
    for i, row_index in enumerate(sampled_rows, start=1):
        blind_id = f"H{i:02d}"
        sheet_rows.append({"item_id": blind_id, "prompt": prompts[row_index]})
        key_items.append({"item_id": blind_id, "row_index": row_index})

    write_csv(sheet_path, sheet_rows, SHEET_COLUMNS)
    with open(key_path, "w", encoding="utf-8") as f:
        json.dump({
            "seed": args.seed,
            "n": n,
            "pool_size": len(prompts),
            "judge_model_id": JUDGE_CONFIG["model_id"],
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "items": key_items,
        }, f, indent=2)
    with open(os.path.join(args.out_dir, "INSTRUCTIONS.md"), "w", encoding="utf-8") as f:
        f.write(INSTRUCTIONS.format(n=n))
    os.makedirs(os.path.join(args.out_dir, "responses"), exist_ok=True)

    print_info(f"Sampled {n} of {len(prompts)} prompts (seed={args.seed})")
    print_success(f"Rater sheet -> {sheet_path}")
    print_success(f"Hidden key  -> {key_path} (do not share with raters)")


if __name__ == "__main__":
    main()
