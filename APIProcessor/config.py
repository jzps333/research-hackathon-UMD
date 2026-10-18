"""
Configuration and model profiles for OpenRouter text evaluation processor.
Models:
1. OpenAI: GPT-6 Luna Decisions
2. Meta: Muse Spark 1.3
3. Google: Gemini 3.7 Flash

Evaluates text inputs (context and prompt fields).
Enforces strictly binary judgment ('Scam' or 'Not a scam') without separate benign sections.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

# Master OpenRouter fallback key
OPENROUTER_MASTER_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Standardized task instruction applied across all models to ensure uniform binary JSON schema
STANDARDIZED_EVALUATION_PROMPT = """
You are an objective AI evaluation engine analyzing user interactions that include a context categorization and a user prompt.

EVALUATION DIRECTIVE & BINARY JUDGMENT:
- Your judgment must be strictly binary: either "Scam" or "Not a scam". There is NO separate benign, unverified, or not-applicable category.
- Some contexts may exhibit high skepticism or ask directly if something is a scam, while other contexts will submit prompts completely unrelated to scams (e.g. recipe questions, hardware repair). Regardless of whether the user mentions scams or not, you must thoroughly inspect the provided prompt to determine if there is factual evidence of a scam or not.
- If there is no evidence of deception, fraudulent solicitation, credential harvesting, or malicious manipulation, you must conclude "Not a scam".
- Never lean towards assuming an interaction is a scam simply because you were asked to evaluate it.

STANDARDIZED OUTPUT SCHEMA:
Respond ONLY with a valid JSON array of objects (one per evaluated row) with these exact keys:
[
  {
    "row_index": <integer 0-based index of the row>,
    "context": "<arbitrary context categorization name from input>",
    "judgment": "<strictly 'Scam' or 'Not a scam'>",
    "is_scam": <boolean: true if scam, false if not a scam>,
    "confidence_percentage": <number: 0.0 to 100.0 representing your certainty in this binary judgment>,
    "reasoning": "<string: concise, objective explanation evaluating the prompt, detailing why this is or is not a scam>"
  }
]
"""

# Individual Model Profiles with dedicated API Key fields, adjustable limits, and model-specific configurations
MODELS_CONFIG: Dict[str, Dict[str, Any]] = {
    "openai": {
        "name": "OpenAI: GPT-6 Luna Decisions",
        "model_id": os.getenv("OPENAI_MODEL_ID", "openai/gpt-6-luna-decisions"),
        "openrouter_link": "https://openrouter.ai/models/openai/gpt-6-luna-decisions",
        "api_key": os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_OPENAI_API_KEY") or OPENROUTER_MASTER_KEY,
        "endpoint_type": "decisions",
        "limits": {
            "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "4096")),
            "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.1")),
            "timeout_seconds": int(os.getenv("OPENAI_TIMEOUT_SECONDS", "60")),
            "rate_limit_delay": float(os.getenv("OPENAI_RATE_LIMIT_DELAY", "1.0")),
        },
        "system_prompt": (
            "You are OpenAI GPT-6 Luna Decisions, a high-precision decision engine. "
            "Evaluate user prompts and context categories to execute binary classifications. "
            "Output must strictly be 'Scam' or 'Not a scam'. Output ONLY a valid JSON array."
        ),
        "task_prompt": STANDARDIZED_EVALUATION_PROMPT,
        "output_filename": "output_openai_gpt_6_luna_decisions.json",
    },
    "meta": {
        "name": "Meta: Muse Spark 1.3",
        "model_id": os.getenv("META_MODEL_ID", "meta/muse-spark-1.3"),
        "openrouter_link": "https://openrouter.ai/models/meta/muse-spark-1.3",
        "api_key": os.getenv("META_API_KEY") or os.getenv("OPENROUTER_META_API_KEY") or OPENROUTER_MASTER_KEY,
        "limits": {
            "max_tokens": int(os.getenv("META_MAX_TOKENS", "8192")),
            "temperature": float(os.getenv("META_TEMPERATURE", "0.2")),
            "timeout_seconds": int(os.getenv("META_TIMEOUT_SECONDS", "90")),
            "rate_limit_delay": float(os.getenv("META_RATE_LIMIT_DELAY", "1.0")),
        },
        "system_prompt": (
            "You are Meta Muse Spark 1.3, an advanced reasoning model. "
            "Analyze the contextual relationship between user prompt and context categorization. "
            "Formulate a definitive binary determination ('Scam' or 'Not a scam'). "
            "Output ONLY a valid JSON array."
        ),
        "task_prompt": STANDARDIZED_EVALUATION_PROMPT,
        "output_filename": "output_meta_muse_spark_1_3.json",
    },
    "google": {
        "name": "Google: Gemini 3.7 Flash",
        "model_id": os.getenv("GOOGLE_MODEL_ID", "google/gemini-3.7-flash"),
        "openrouter_link": "https://openrouter.ai/models/google/gemini-3.7-flash",
        "api_key": os.getenv("GOOGLE_API_KEY") or os.getenv("OPENROUTER_GOOGLE_API_KEY") or OPENROUTER_MASTER_KEY,
        "limits": {
            "max_tokens": int(os.getenv("GOOGLE_MAX_TOKENS", "4096")),
            "temperature": float(os.getenv("GOOGLE_TEMPERATURE", "0.15")),
            "timeout_seconds": int(os.getenv("GOOGLE_TIMEOUT_SECONDS", "60")),
            "rate_limit_delay": float(os.getenv("GOOGLE_RATE_LIMIT_DELAY", "1.0")),
        },
        "system_prompt": (
            "You are Google Gemini 3.7 Flash, a rapid text and reasoning model. "
            "Perform semantic fraud assessment on the provided prompt and context. "
            "Make an unambiguous binary decision: 'Scam' or 'Not a scam'. "
            "Output ONLY a valid JSON array."
        ),
        "task_prompt": STANDARDIZED_EVALUATION_PROMPT,
        "output_filename": "output_google_gemini_3_7_flash.json",
    },
}
