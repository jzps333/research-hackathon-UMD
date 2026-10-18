"""
APIProcessor - Multi-Model OpenRouter Evaluation Processor
Evaluates user prompts and incident IDs row-by-row across:
1. OpenAI: GPT-6 Luna Decisions
2. Meta: Muse Spark 1.3
3. Google: Gemini 3.7 Flash

Enforces strictly binary judgment ("Scam" or "Not a scam").
"""

import os
import sys
import json
import time
import re
import argparse
from typing import Any, Dict, List, Optional, Union
import requests
from dotenv import load_dotenv

from config import MODELS_CONFIG

load_dotenv()

# Rich styling with fallback
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    console = Console()
    HAS_RICH = True
except ImportError:
    console = None
    HAS_RICH = False


def print_info(text: str) -> None:
    if HAS_RICH and console:
        console.print(f"[bold cyan][INFO][/bold cyan] {text}")
    else:
        print(f"[INFO] {text}")


def print_warn(text: str) -> None:
    if HAS_RICH and console:
        console.print(f"[bold yellow][WARN][/bold yellow] {text}")
    else:
        print(f"[WARN] {text}")


def print_error(text: str) -> None:
    if HAS_RICH and console:
        console.print(f"[bold red][ERROR][/bold red] {text}")
    else:
        print(f"[ERROR] {text}", file=sys.stderr)


def print_success(text: str) -> None:
    if HAS_RICH and console:
        console.print(f"[bold green][SUCCESS][/bold green] {text}")
    else:
        print(f"[SUCCESS] {text}")


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Load and validate JSON dataset from disk."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON file not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "rows" in data:
        return data["rows"]
    elif isinstance(data, dict):
        return [data]
    else:
        raise ValueError(f"Unexpected JSON structure in {file_path}. Expected a list of rows.")


def extract_clean_json_object(raw_text: str, default_row: int = 0, default_incident_id: str = "") -> Dict[str, Any]:
    """
    Extract a valid JSON object from model response text.
    Handles responses wrapped in markdown fences or commentary.
    """
    cleaned = raw_text.strip()

    # Direct parse
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, list) and len(obj) > 0:
            return obj[0]
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Markdown block
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if match:
        block = match.group(1).strip()
        try:
            obj = json.loads(block)
            if isinstance(obj, list) and len(obj) > 0:
                return obj[0]
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # Search for { ... }
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        snippet = cleaned[first_brace : last_brace + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass

    # Fallback to safe structure
    is_scam = "scam" in cleaned.lower() and "not a scam" not in cleaned.lower()
    return {
        "row_index": default_row,
        "incident_id": default_incident_id,
        "judgment": "Scam" if is_scam else "Not a scam",
        "is_scam": is_scam,
        "reasoning": f"Extracted from unstructured response: {raw_text[:200]}",
    }


def row_prompt(row: Dict[str, Any]) -> str:
    """Prompt text of a dataset row (current datasets use 'evaluation_prompt')."""
    return row.get("evaluation_prompt") or row.get("prompt") or row.get("Prompt", "")


def row_incident_id(row: Dict[str, Any], idx: int) -> str:
    return row.get("incident_id") or f"Incident_{idx}"


class NonRetryableError(RuntimeError):
    """Request can never succeed as sent (bad model ID, bad key, wrong endpoint) - don't retry."""


def raise_for_openrouter_error(response: requests.Response) -> None:
    if response.ok:
        return
    try:
        err = response.json()
    except Exception:
        err = response.text
    msg = f"OpenRouter error ({response.status_code}): {err}"
    # 4xx other than rate limiting / timeout means the request itself is wrong
    if 400 <= response.status_code < 500 and response.status_code not in (408, 429):
        raise NonRetryableError(msg)
    raise RuntimeError(msg)


def call_openrouter_for_row(
    model_cfg: Dict[str, Any],
    row_idx: int,
    row_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Send an individual row with its context and prompt to the model via OpenRouter API.
    """
    api_key = model_cfg.get("api_key")
    if not api_key or "your_" in api_key:
        raise ValueError(
            f"Missing API key for {model_cfg['name']}. Set in .env or provide OPENROUTER_API_KEY."
        )

    model_id = model_cfg["model_id"]
    limits = model_cfg["limits"]
    system_prompt = model_cfg["system_prompt"]

    incident_id = row_incident_id(row_data, row_idx)
    prompt_text = row_prompt(row_data)

    instruction = (
        f"Analyze Row Index {row_idx}.\n"
        f"incident Id: {incident_id}\n"
        f"User Prompt: {prompt_text}\n\n"
        f"Evaluate the prompt within this context. Your judgment must be strictly binary: 'Scam' or 'Not a scam'.\n"
        f"Respond ONLY with a single JSON object matching this schema:\n"
        f'{{"row_index": {row_idx}, "incident_id": "{incident_id}", "judgment": "Scam" | "Not a scam", '
        f'"is_scam": true | false, "reasoning": "<concise explanation>"}}'
    )

    is_decisions = model_cfg.get("endpoint_type") == "decisions"

    if is_decisions:
        url = "https://openrouter.ai/api/alpha/decisions"
        payload = {
            "model": model_id,
            "state": (
                f"{system_prompt}\n\n"
                f"Row Index: {row_idx}\n"
                f"Incident ID: {incident_id}\n"
                f"User Prompt: {prompt_text}\n\n"
                "Classify whether this user prompt should be judged "
                "as a scam, following the evaluation instructions."
            ),
            "questions": {
                "judgment": {
                    "type": "choice",
                    "instructions": (
                        "Choose whether the user prompt is a scam. "
                        "Evaluate the prompt itself, not merely whether "
                        "it mentions a scam."
                    ),
                    "criteria": {
                        "Scam": (
                            "The prompt itself is part of, requests, "
                            "or exhibits a fraudulent or deceptive scam."
                        ),
                        "Not a scam": (
                            "The prompt does not itself constitute "
                            "a scam or deceptive solicitation."
                        ),
                    },
                }
            },
        }
    else:
        url = "https://openrouter.ai/api/v1/chat/completions"
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction},
            ],
            "max_tokens": limits.get("max_tokens", 4096),
        }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    site_url = os.getenv("OPENROUTER_SITE_URL")
    site_name = os.getenv("OPENROUTER_SITE_NAME")
    if site_url:
        headers["HTTP-Referer"] = site_url
    if site_name:
        headers["X-Title"] = site_name

    timeout = limits.get("timeout_seconds", 60)
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)

    raise_for_openrouter_error(response)

    res_json = response.json()

    if is_decisions:
        answer = res_json.get("answers", {}).get("judgment", {})

        if answer.get("type") != "choice":
            raise ValueError(
                f"Missing or invalid judgment answer: {res_json}"
            )

        judgment = answer.get("choice")
        if judgment not in ("Scam", "Not a scam"):
            raise ValueError(
                f"Unexpected judgment value: {judgment!r}"
            )

        parsed_row = {
            "row_index": row_idx,
            "incident_id": incident_id,
            "judgment": judgment,
            "is_scam": judgment == "Scam",
            "reasoning": (
                "Classified by the Decisions API. "
                f"Choice probabilities: {answer.get('probabilities', {})}"
            ),
        }
    else:
        choices = res_json.get("choices", [])
        if not choices:
            raise ValueError(f"No choices returned: {res_json}")

        raw_text = choices[0]["message"]["content"]
        parsed_row = extract_clean_json_object(
            raw_text,
            default_row=row_idx,
            default_incident_id=incident_id,
        )

    # Standardize binary field values
    j_val = str(parsed_row.get("judgment", "")).strip().lower()
    is_scam = parsed_row.get("is_scam")
    if is_scam is None:
        is_scam = (j_val == "scam")

    parsed_row["row_index"] = row_idx
    parsed_row["incident_id"] = incident_id
    parsed_row["judgment"] = "Scam" if is_scam else "Not a scam"
    parsed_row["is_scam"] = bool(is_scam)
    parsed_row.pop("confidence_percentage", None)

    return parsed_row


# Alias for backward compatibility
call_openrouter_vision_for_row = call_openrouter_for_row


def generate_theoretical_mock_row(
    model_key: str,
    row_idx: int,
    row_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Simulates evaluation of the context and user prompt to determine
    whether there is evidence of a scam or not (strictly binary judgment).
    """
    incident_id = row_incident_id(row_data, row_idx)
    prompt = row_prompt(row_data)

    prompt_lower = prompt.lower()

    # Row 0: USPS Smishing
    if "usps" in prompt_lower or "smishing" in prompt_lower or "redelivery" in prompt_lower:
        judgment = "Scam"
        is_scam = True
        if model_key == "openai":
            reasoning = (
                "Textual analysis confirms fraudulent domain 'usps-redelivery-portal.top' "
                "and artificial 12-hour expiration urgency. Definitive smishing credential harvesting attempt."
            )
        elif model_key == "meta":
            reasoning = (
                "Message analysis shows an unsolicited text message from an unknown sender paired with an "
                "untrusted .top top-level domain pretending to be postal authorities. High-confidence deceptive scam."
            )
        else:  # google
            reasoning = (
                "Direct message reading confirms phishing URL 'usps-redelivery-portal.top' and spoofed package alert "
                "signature. Textual evidence confirms smishing fraud."
            )

    # Row 1: Recipe Card (Prompt unrelated to scam)
    elif "recipe" in prompt_lower or "soup" in prompt_lower or "leek" in prompt_lower:
        judgment = "Not a scam"
        is_scam = False
        if model_key == "openai":
            reasoning = (
                "Prompt text presents ingredients and cooking directions for Potato Leek Soup. "
                "No financial requests, links, or deceptive mechanisms present. Clean and legitimate."
            )
        elif model_key == "meta":
            reasoning = (
                "Context depicts authentic culinary instructions with standard kitchen measurements. "
                "Prompt asks for ingredient clarification. Zero indicators of deception or fraud."
            )
        else:  # google
            reasoning = (
                "Semantic analysis confirms standard culinary recipe (leeks, potatoes, broth). "
                "No scam indicators detected."
            )

    # Row 2: Telegram Job Check Scam
    elif "telegram" in prompt_lower or "job" in prompt_lower or "check" in prompt_lower or "zelle" in prompt_lower:
        judgment = "Scam"
        is_scam = True
        if model_key == "openai":
            reasoning = (
                "Prompt details a Telegram recruitment communication instructing the user to deposit a $3,200 company check "
                "and immediately send $2,400 via Zelle to a third-party vendor. Classic fake check overpayment fraud."
            )
        elif model_key == "meta":
            reasoning = (
                "Contextual review of job offer demonstrates textbook advance-fee employment fraud: "
                "unrealistic pay ($55/hr), remote hiring on Telegram, and an irreversible Zelle transfer demand."
            )
        else:  # google
            reasoning = (
                "Textual evidence in communication highlights high-risk fraud markers: fake check issuance coupled "
                "with an urgent Zelle remittance for home office equipment."
            )

    # Row 3: Washing Machine Valve (Prompt unrelated to scam)
    elif "appliance" in prompt_lower or "valve" in prompt_lower or "machine" in prompt_lower or "hose" in prompt_lower:
        judgment = "Not a scam"
        is_scam = False
        if model_key == "openai":
            reasoning = (
                "User prompt describes washing machine maintenance and standard 3/4\" hose thread specifications. "
                "Valid technical plumbing inquiry; no scam elements detected."
            )
        elif model_key == "meta":
            reasoning = (
                "Inquiry concerns mechanical equipment maintenance for water inlet valves and filter cleaning. "
                "Prompt concerns plumbing repairs. Absolutely no evidence of fraudulent manipulation."
            )
        else:  # google
            reasoning = (
                "Prompt analysis shows authentic appliance maintenance specifications. "
                "Zero fraudulent or malicious characteristics present."
            )

    # Row 4: Fake Invoice
    elif "invoice" in prompt_lower or "subscription" in prompt_lower or "geek" in prompt_lower or "1-888" in prompt_lower:
        judgment = "Scam"
        is_scam = True
        if model_key == "openai":
            reasoning = (
                "Analysis of the invoice notice shows an unauthorized $799.00 auto-debit claim paired with "
                "a prominent telephone dispute number (1-888-555-0194). Classic callback tech support phishing scam."
            )
        elif model_key == "meta":
            reasoning = (
                "Prompt describes a fabricated billing notification engineered to induce urgency and panic, "
                "directing the recipient to call a toll-free number for a refund. Clear refund fraud vector."
            )
        else:  # google
            reasoning = (
                "Notice details reveal callback scam characteristics: bogus 'Geek Cloud Protection' debit and "
                "fraudulent phone solicitation vector. Strong evidence of a scam."
            )

    else:
        # Default fallback
        judgment = "Not a scam"
        is_scam = False
        reasoning = "Textual review confirms no presence of fraudulent schemes, phishing, or deceptive solicitations."

    return {
        "row_index": row_idx,
        "incident_id": incident_id,
        "judgment": judgment,
        "is_scam": is_scam,
        "reasoning": reasoning,
    }


def process_models(
    input_file: str,
    target_model: str,
    output_dir: str,
    dry_run: bool = False,
    force: bool = False,
    max_retries: int = 3,
) -> None:
    """
    Execute processing across designated models row-by-row and save individual JSON outputs.
    Results are saved after every row; a rerun skips rows already done by the same model
    (dry-run rows are never reused for a live run). Failed rows are retried, then skipped.
    """
    os.makedirs(output_dir, exist_ok=True)
    input_rows = load_json_file(input_file)

    selected_models = (
        list(MODELS_CONFIG.keys()) if target_model == "all" else [target_model]
    )

    print_info(f"Target Input File: {input_file} ({len(input_rows)} rows)")
    print_info(f"Output Directory: {output_dir}")
    print_info(f"Executing models: {', '.join(selected_models).upper()}")
    if dry_run:
        print_warn("Mode: DRY RUN (Simulating text evaluation of prompt & binary decisions)")

    results_summary = []

    for key in selected_models:
        cfg = MODELS_CONFIG[key]
        model_name = cfg["name"]
        model_id = cfg["model_id"]
        out_path = os.path.join(output_dir, cfg["output_filename"])

        print("\n" + "=" * 70)
        print_info(f"Processing with [bold]{model_name}[/bold] ({model_id})")
        print_info(f"OpenRouter Profile: {cfg['openrouter_link']}")

        generated_by = "dry-run" if dry_run else model_id
        start_time = time.time()

        api_key = cfg.get("api_key")
        if not dry_run and (not api_key or "your_" in api_key):
            print_error(f"Missing API key for {model_name}. Set OPENROUTER_API_KEY in .env. Skipping model.")
            results_summary.append({"Model": model_name, "Model ID": model_id, "Status": "FAILED",
                                    "Duration": "0s", "Output File": "None"})
            continue

        # Resume: reuse rows from a previous run of this same model on this same dataset
        done: Dict[int, Dict[str, Any]] = {}
        if os.path.exists(out_path) and not force:
            with open(out_path, "r", encoding="utf-8") as f:
                previous = json.load(f)
            for item in previous:
                idx = item.get("row_index")
                if (
                    isinstance(idx, int) and 0 <= idx < len(input_rows)
                    and item.get("generated_by") == generated_by
                    and item.get("incident_id") == row_incident_id(input_rows[idx], idx)
                ):
                    done[idx] = item
            if done:
                print_info(f"Resuming: {len(done)}/{len(input_rows)} row(s) already done (use --force to redo)")

        failed_rows = []
        todo = [i for i in range(len(input_rows)) if i not in done]
        for n, idx in enumerate(todo):
            row = input_rows[idx]
            incident_id = row_incident_id(row, idx)

            eval_obj = None
            for attempt in range(1, max_retries + 1):
                try:
                    if dry_run:
                        time.sleep(0.15)  # Simulate brief processing
                        eval_obj = generate_theoretical_mock_row(key, idx, row)
                    else:
                        eval_obj = call_openrouter_for_row(cfg, idx, row)
                    break
                except Exception as e:
                    if isinstance(e, NonRetryableError):
                        print_error(f"Row {idx} ({incident_id}) failed: {e}")
                        break
                    if attempt < max_retries:
                        wait = 2 ** attempt
                        print_warn(f"Row {idx} ({incident_id}) attempt {attempt} failed: {e}. Retrying in {wait}s")
                        time.sleep(wait)
                    else:
                        print_error(f"Row {idx} ({incident_id}) failed after {max_retries} attempts: {e}")

            if eval_obj is None:
                failed_rows.append(idx)
            else:
                eval_obj["generated_by"] = generated_by
                done[idx] = eval_obj
                # Save after every row so a crash or rate limit loses nothing
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump([done[i] for i in sorted(done)], f, indent=2, ensure_ascii=False)

            # Rate limit delay between rows if needed
            delay = cfg["limits"].get("rate_limit_delay", 0.5)
            if delay > 0 and not dry_run and n < len(todo) - 1:
                time.sleep(delay)

        model_results = [done[i] for i in sorted(done)]
        success = not failed_rows
        elapsed = round(time.time() - start_time, 2)
        if failed_rows:
            print_warn(f"{model_name}: {len(failed_rows)} row(s) failed {failed_rows} - rerun the same command to retry them")

        if model_results:
            print_success(
                f"Saved binary evaluation JSON -> [bold]{out_path}[/bold] "
                f"({len(model_results)}/{len(input_rows)} rows, {elapsed}s)"
            )

            # Table breakdown of the binary judgments
            if HAS_RICH and console:
                row_table = Table(title=f"Binary Judgments: {model_name}")
                row_table.add_column("Row", justify="center", style="dim")
                row_table.add_column("Incident ID", style="cyan")
                row_table.add_column("Base Judgment", justify="center")
                row_table.add_column("Reasoning", style="green", max_width=50)

                for item in model_results:
                    j_str = str(item.get("judgment", ""))
                    is_scam = item.get("is_scam", False)
                    j_color = "bold red" if is_scam else "bold green"
                    row_table.add_row(
                        str(item.get("row_index", "-")),
                        str(item.get("incident_id", "")),
                        f"[{j_color}]{j_str}[/{j_color}]",
                        str(item.get("reasoning", "")),
                    )
                console.print(row_table)

            results_summary.append({
                "Model": model_name,
                "Model ID": model_id,
                "Status": "SUCCESS" if success else f"PARTIAL ({len(model_results)}/{len(input_rows)})",
                "Duration": f"{elapsed}s",
                "Output File": cfg["output_filename"],
            })
        else:
            results_summary.append({
                "Model": model_name,
                "Model ID": model_id,
                "Status": "FAILED",
                "Duration": f"{elapsed}s",
                "Output File": "None",
            })

    # Summary table
    print("\n" + "=" * 70)
    print_info("Execution Summary:")
    if HAS_RICH and console:
        summary_tbl = Table(title="Overall Run Summary")
        summary_tbl.add_column("Model Name", style="cyan")
        summary_tbl.add_column("Model ID", style="magenta")
        summary_tbl.add_column("Status")
        summary_tbl.add_column("Duration")
        summary_tbl.add_column("Output JSON File", style="yellow")
        for row in results_summary:
            status_style = {"SUCCESS": "green", "FAILED": "red"}.get(row["Status"], "yellow")
            summary_tbl.add_row(
                row["Model"],
                row["Model ID"],
                f"[{status_style}]{row['Status']}[/{status_style}]",
                row["Duration"],
                row["Output File"],
            )
        console.print(summary_tbl)
    else:
        for row in results_summary:
            print(f"- {row['Model']} ({row['Model ID']}): {row['Status']} -> {row['Output File']}")


def main():
    parser = argparse.ArgumentParser(
        description="Binary Scam Evaluation across OpenAI GPT-6 Luna, Meta Muse Spark, and Google Gemini 3.7 Flash via OpenRouter."
    )
    parser.add_argument(
        "-f", "--file",
        default="sample_data.json",
        help="Path to input JSON file with context and prompt fields (default: sample_data.json)",
    )
    parser.add_argument(
        "-m", "--model",
        choices=["all", "openai", "meta", "google"],
        default="all",
        help="Which model to run: 'all', 'openai', 'meta', or 'google' (default: all)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="output",
        help="Directory to save the resulting individual JSON files (default: output)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore previous results in the output directory and rerun every row.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate execution and generate theoretical binary JSON outputs without calling live APIs.",
    )

    args = parser.parse_args()

    try:
        process_models(
            input_file=args.file,
            target_model=args.model,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            force=args.force,
        )
    except Exception as e:
        print_error(f"Execution error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
