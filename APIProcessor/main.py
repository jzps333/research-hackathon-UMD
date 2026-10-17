"""
APIProcessor - Multi-Model OpenRouter Evaluation Processor
Evaluates user prompts and personas row-by-row across:
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


def extract_clean_json_object(raw_text: str, default_row: int = 0, default_persona: str = "") -> Dict[str, Any]:
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
        "persona": default_persona,
        "judgment": "Scam" if is_scam else "Not a scam",
        "is_scam": is_scam,
        "confidence_percentage": 50.0,
        "reasoning": f"Extracted from unstructured response: {raw_text[:200]}",
    }


def call_openrouter_for_row(
    model_cfg: Dict[str, Any],
    row_idx: int,
    row_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Send an individual row with its persona and prompt to the model via OpenRouter API.
    """
    api_key = model_cfg.get("api_key")
    if not api_key or "your_" in api_key:
        raise ValueError(
            f"Missing API key for {model_cfg['name']}. Set in .env or provide OPENROUTER_API_KEY."
        )

    model_id = model_cfg["model_id"]
    limits = model_cfg["limits"]
    system_prompt = model_cfg["system_prompt"]

    persona = row_data.get("persona") or row_data.get("Persona", f"Persona_{row_idx}")
    prompt_text = row_data.get("prompt") or row_data.get("Prompt", "")

    instruction = (
        f"Analyze Row Index {row_idx}.\n"
        f"Persona: {persona}\n"
        f"User Prompt: {prompt_text}\n\n"
        f"Evaluate the prompt in the context of the persona. Your judgment must be strictly binary: 'Scam' or 'Not a scam'.\n"
        f"Respond ONLY with a single JSON object matching this schema:\n"
        f'{{"row_index": {row_idx}, "persona": "{persona}", "judgment": "Scam" | "Not a scam", '
        f'"is_scam": true | false, "confidence_percentage": <number 0.0-100.0>, "reasoning": "<concise explanation>"}}'
    )

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
        ],
        "max_tokens": limits.get("max_tokens", 4096),
        "temperature": limits.get("temperature", 0.1),
    }

    url = "https://openrouter.ai/api/v1/chat/completions"
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

    if not response.ok:
        try:
            err = response.json()
        except Exception:
            err = response.text
        raise RuntimeError(f"OpenRouter error ({response.status_code}): {err}")

    res_json = response.json()
    choices = res_json.get("choices", [])
    if not choices:
        raise ValueError(f"No choices returned: {res_json}")

    raw_text = choices[0]["message"]["content"]
    parsed_row = extract_clean_json_object(raw_text, default_row=row_idx, default_persona=persona)

    # Standardize binary field values
    j_val = str(parsed_row.get("judgment", "")).strip().lower()
    is_scam = parsed_row.get("is_scam")
    if is_scam is None:
        is_scam = (j_val == "scam")

    parsed_row["row_index"] = row_idx
    parsed_row["persona"] = persona
    parsed_row["judgment"] = "Scam" if is_scam else "Not a scam"
    parsed_row["is_scam"] = bool(is_scam)
    if "confidence_percentage" not in parsed_row:
        parsed_row["confidence_percentage"] = 95.0

    return parsed_row


# Alias for backward compatibility
call_openrouter_vision_for_row = call_openrouter_for_row


def generate_theoretical_mock_row(
    model_key: str,
    row_idx: int,
    row_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Simulates evaluation of the persona and user prompt to determine
    whether there is evidence of a scam or not (strictly binary judgment).
    """
    persona = row_data.get("persona") or row_data.get("Persona", f"Persona_{row_idx}")
    prompt = row_data.get("prompt") or row_data.get("Prompt", "")

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
            conf = 99.4
        elif model_key == "meta":
            reasoning = (
                "Message analysis shows an unsolicited text message from an unknown sender paired with an "
                "untrusted .top top-level domain pretending to be postal authorities. High-confidence deceptive scam."
            )
            conf = 98.9
        else:  # google
            reasoning = (
                "Direct message reading confirms phishing URL 'usps-redelivery-portal.top' and spoofed package alert "
                "signature. Textual evidence confirms smishing fraud."
            )
            conf = 99.6

    # Row 1: Recipe Card (Prompt unrelated to scam)
    elif "recipe" in prompt_lower or "soup" in prompt_lower or "leek" in prompt_lower:
        judgment = "Not a scam"
        is_scam = False
        if model_key == "openai":
            reasoning = (
                "Prompt text presents ingredients and cooking directions for Potato Leek Soup. "
                "No financial requests, links, or deceptive mechanisms present. Clean and legitimate."
            )
            conf = 99.8
        elif model_key == "meta":
            reasoning = (
                "Context depicts authentic culinary instructions with standard kitchen measurements. "
                "Prompt asks for ingredient clarification. Zero indicators of deception or fraud."
            )
            conf = 99.7
        else:  # google
            reasoning = (
                "Semantic analysis confirms standard culinary recipe (leeks, potatoes, broth). "
                "No scam indicators detected."
            )
            conf = 99.9

    # Row 2: Telegram Job Check Scam
    elif "telegram" in prompt_lower or "job" in prompt_lower or "check" in prompt_lower or "zelle" in prompt_lower:
        judgment = "Scam"
        is_scam = True
        if model_key == "openai":
            reasoning = (
                "Prompt details a Telegram recruitment communication instructing the user to deposit a $3,200 company check "
                "and immediately send $2,400 via Zelle to a third-party vendor. Classic fake check overpayment fraud."
            )
            conf = 99.2
        elif model_key == "meta":
            reasoning = (
                "Contextual review of job offer demonstrates textbook advance-fee employment fraud: "
                "unrealistic pay ($55/hr), remote hiring on Telegram, and an irreversible Zelle transfer demand."
            )
            conf = 98.8
        else:  # google
            reasoning = (
                "Textual evidence in communication highlights high-risk fraud markers: fake check issuance coupled "
                "with an urgent Zelle remittance for home office equipment."
            )
            conf = 99.3

    # Row 3: Washing Machine Valve (Prompt unrelated to scam)
    elif "appliance" in prompt_lower or "valve" in prompt_lower or "machine" in prompt_lower or "hose" in prompt_lower:
        judgment = "Not a scam"
        is_scam = False
        if model_key == "openai":
            reasoning = (
                "User prompt describes washing machine maintenance and standard 3/4\" hose thread specifications. "
                "Valid technical plumbing inquiry; no scam elements detected."
            )
            conf = 99.8
        elif model_key == "meta":
            reasoning = (
                "Inquiry concerns mechanical equipment maintenance for water inlet valves and filter cleaning. "
                "Prompt concerns plumbing repairs. Absolutely no evidence of fraudulent manipulation."
            )
            conf = 99.6
        else:  # google
            reasoning = (
                "Prompt analysis shows authentic appliance maintenance specifications. "
                "Zero fraudulent or malicious characteristics present."
            )
            conf = 99.9

    # Row 4: Fake Invoice
    elif "invoice" in prompt_lower or "subscription" in prompt_lower or "geek" in prompt_lower or "1-888" in prompt_lower:
        judgment = "Scam"
        is_scam = True
        if model_key == "openai":
            reasoning = (
                "Analysis of the invoice notice shows an unauthorized $799.00 auto-debit claim paired with "
                "a prominent telephone dispute number (1-888-555-0194). Classic callback tech support phishing scam."
            )
            conf = 97.2
        elif model_key == "meta":
            reasoning = (
                "Prompt describes a fabricated billing notification engineered to induce urgency and panic, "
                "directing the recipient to call a toll-free number for a refund. Clear refund fraud vector."
            )
            conf = 96.5
        else:  # google
            reasoning = (
                "Notice details reveal callback scam characteristics: bogus 'Geek Cloud Protection' debit and "
                "fraudulent phone solicitation vector. Strong evidence of a scam."
            )
            conf = 97.8

    else:
        # Default fallback
        judgment = "Not a scam"
        is_scam = False
        reasoning = "Textual review confirms no presence of fraudulent schemes, phishing, or deceptive solicitations."
        conf = 95.0

    return {
        "row_index": row_idx,
        "persona": persona,
        "judgment": judgment,
        "is_scam": is_scam,
        "confidence_percentage": conf,
        "reasoning": reasoning,
    }


def process_models(
    input_file: str,
    target_model: str,
    output_dir: str,
    dry_run: bool = False,
) -> None:
    """Execute processing across designated models row-by-row and save individual JSON outputs."""
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

        model_results = []
        start_time = time.time()
        success = True

        for idx, row in enumerate(input_rows):
            persona_name = row.get("persona") or row.get("Persona", f"Persona_{idx}")

            try:
                if dry_run:
                    time.sleep(0.15)  # Simulate brief processing
                    eval_obj = generate_theoretical_mock_row(key, idx, row)
                else:
                    eval_obj = call_openrouter_for_row(cfg, idx, row)

                model_results.append(eval_obj)

            except Exception as e:
                print_error(f"Error on Row {idx} ({persona_name}): {e}")
                success = False
                break

            # Rate limit delay between rows if needed
            delay = cfg["limits"].get("rate_limit_delay", 0.5)
            if delay > 0 and not dry_run and idx < len(input_rows) - 1:
                time.sleep(delay)

        elapsed = round(time.time() - start_time, 2)

        if success and model_results:
            # Save distinct JSON file
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(model_results, f, indent=2)

            print_success(f"Saved binary evaluation JSON -> [bold]{out_path}[/bold] ({elapsed}s)")

            # Table breakdown of the binary judgments
            if HAS_RICH and console:
                row_table = Table(title=f"Binary Judgments: {model_name}")
                row_table.add_column("Row", justify="center", style="dim")
                row_table.add_column("Persona", style="cyan")
                row_table.add_column("Base Judgment", justify="center")
                row_table.add_column("Confidence", justify="right", style="magenta")
                row_table.add_column("Reasoning", style="green", max_width=50)

                for item in model_results:
                    j_str = str(item.get("judgment", ""))
                    is_scam = item.get("is_scam", False)
                    j_color = "bold red" if is_scam else "bold green"
                    conf = f"{item.get('confidence_percentage', 0.0):.1f}%"
                    row_table.add_row(
                        str(item.get("row_index", "-")),
                        str(item.get("persona", "")),
                        f"[{j_color}]{j_str}[/{j_color}]",
                        conf,
                        str(item.get("reasoning", "")),
                    )
                console.print(row_table)

            results_summary.append({
                "Model": model_name,
                "Model ID": model_id,
                "Status": "SUCCESS",
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
            status_style = "green" if row["Status"] == "SUCCESS" else "red"
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
        help="Path to input JSON file with persona and prompt fields (default: sample_data.json)",
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
        )
    except Exception as e:
        print_error(f"Execution error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
