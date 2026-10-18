"""
Convert the team's Excel/CSV sheet (incident_id, evaluation_prompt, label, ...) into a dataset JSON
that main.py and judge.py both read. Row order is preserved, so row N in the
sheet becomes row_index N everywhere downstream.

The 'label' field is ground truth. main.py never sends it to the evaluated models.

Usage:
    python import_labels.py -i scams.xlsx -o older_adults_scams_dataset.json
    python import_labels.py -i scams.csv -o older_adults_scams_dataset.json --label-col "is scam"
"""

import argparse
import json
import sys

from eval_utils import normalize_verdict, read_table
from main import print_error, print_info, print_success, print_warn


def main():
    parser = argparse.ArgumentParser(description="Import a labeled Excel/CSV sheet into dataset JSON.")
    parser.add_argument("-i", "--input", required=True, help="Path to .xlsx or .csv file")
    parser.add_argument("-o", "--output", default="older_adults_scams_dataset.json", help="Output JSON path")
    parser.add_argument("--sheet", default=None, help="Excel sheet name (default: first/active sheet)")
    parser.add_argument("--prompt-col", default=None,
                        help="Header of the prompt column (default: evaluation_prompt or prompt)")
    parser.add_argument("--label-col", default="label", help="Header of the ground-truth column (Scam / Not a scam)")
    args = parser.parse_args()

    try:
        table = read_table(args.input, sheet=args.sheet)
    except Exception as e:
        print_error(str(e))
        sys.exit(1)

    if not table:
        print_error(f"No rows found in {args.input}")
        sys.exit(1)

    label_col = args.label_col.strip().lower()
    headers = set(table[0].keys())
    if args.prompt_col:
        prompt_col = args.prompt_col.strip().lower()
    else:
        prompt_col = next((c for c in ("evaluation_prompt", "prompt") if c in headers), "evaluation_prompt")
    if prompt_col not in headers:
        print_error(f"Missing prompt column {prompt_col!r}. Found headers: {sorted(headers)}")
        sys.exit(1)
    if label_col not in headers:
        print_warn(f"No '{label_col}' column found - dataset will have no ground-truth labels.")

    rows, bad = [], []
    for i, rec in enumerate(table):
        prompt = str(rec.get(prompt_col) or "").strip()
        if not prompt:
            print_warn(f"Sheet row {i + 2}: empty prompt, skipped")
            continue
        # Keep every column from the sheet (incident_id, context, scam_type, ...); label is normalized below
        out = {k: (v.strip() if isinstance(v, str) else v) for k, v in rec.items() if k and k != label_col}
        out[prompt_col] = prompt
        if label_col in headers:
            raw = rec.get(label_col)
            label = normalize_verdict(raw)
            if label is None:
                bad.append((i + 2, raw))
            out["label"] = label
        rows.append(out)

    if bad:
        print_error("Unrecognized label values (use 'Scam' or 'Not a scam'):")
        for sheet_row, raw in bad:
            print_error(f"  sheet row {sheet_row}: {raw!r}")
        sys.exit(1)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    n_scam = sum(r.get("label") == "Scam" for r in rows)
    print_info(f"Rows: {len(rows)}  |  Scam: {n_scam}  |  Not a scam: {len(rows) - n_scam if label_col in headers else 'n/a'}")
    print_success(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
