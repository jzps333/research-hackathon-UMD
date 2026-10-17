# APIProcessor: Multi-Model Binary Scam Evaluator

A Python pipeline that ingests a JSON dataset containing user interactions (`context` and `prompt` fields) and routes each row through three models on OpenRouter:

1. **OpenAI: GPT-6 Luna Decisions** (`openai/gpt-6-luna-decisions`)
2. **Meta: Muse Spark 1.3** (`meta/muse-spark-1.3`)
3. **Google: Gemini 3.7 Flash** (`google/gemini-3.7-flash`)

The models inspect the user prompts and context categories, passing a **strictly binary judgment** on whether there is evidence of a scam or not.

---

## 🔗 Model Links on OpenRouter

| Provider & Model | OpenRouter Model Identifier | OpenRouter Model Profile |
| :--- | :--- | :--- |
| **OpenAI: GPT-6 Luna Decisions** | `openai/gpt-6-luna-decisions` | [OpenRouter Model Page](https://openrouter.ai/models/openai/gpt-6-luna-decisions) |
| **Meta: Muse Spark 1.3** | `meta/muse-spark-1.3` | [OpenRouter Model Page](https://openrouter.ai/models/meta/muse-spark-1.3) |
| **Google: Gemini 3.7 Flash** | `google/gemini-3.7-flash` | [OpenRouter Model Page](https://openrouter.ai/models/google/gemini-3.7-flash) |

---

## 📥 Input Data Format

The pipeline expects a JSON file where each row contains:
* **`context`**: Arbitrary category or user context (e.g. `Category_SeniorConsumer`, `Category_HomeCook`).
* **`prompt`**: The user query or message text to be evaluated.

```json
[
  {
    "context": "Category_SeniorConsumer",
    "prompt": "I received this text message: 'USPS Alert: Package #US9514901192 is on hold due to missing street address. Please confirm your delivery address at: http://usps-redelivery-portal.top/track. Action required within 12 hours or item returns.' Is this text message real or is it a scam?"
  },
  {
    "context": "Category_HomeCook",
    "prompt": "Here is a recipe for Rustic Potato Leek Soup: 4 large leeks, washed & sliced; 3 russet potatoes, peeled & cubed; 4 cups vegetable broth; 1 cup heavy cream; 2 tbsp butter, salt & fresh pepper. Instructions: Simmer leeks in butter for 10 min. Add potatoes and broth. Simmer for 25 min until tender. Can you read the measurements on this recipe and list the ingredients for me?"
  }
]
```

---

## 📤 Standardized Output JSON Schema

Each model writes its own separate JSON file in the `output/` directory with strictly binary judgments:

```json
[
  {
    "row_index": 0,
    "context": "Category_SeniorConsumer",
    "judgment": "Scam",
    "is_scam": true,
    "confidence_percentage": 99.6,
    "reasoning": "Direct message reading confirms phishing URL 'usps-redelivery-portal.top' and spoofed package alert signature. Textual evidence confirms smishing fraud."
  },
  {
    "row_index": 1,
    "context": "Category_HomeCook",
    "judgment": "Not a scam",
    "is_scam": false,
    "confidence_percentage": 99.9,
    "reasoning": "Semantic analysis confirms standard culinary recipe (leeks, potatoes, broth). No scam indicators detected."
  }
]
```

---

## ⚙️ Configuration & Adjustable Limits

Create your `.env` file from `.env.example`:
```powershell
copy .env.example .env
```

Set either a unified `OPENROUTER_API_KEY` or individual provider keys with custom limits:
```env
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxx

# OpenAI GPT-6 Luna Decisions
OPENAI_API_KEY=sk-or-v1-openai-key
OPENAI_MODEL_ID=openai/gpt-6-luna-decisions
OPENAI_MAX_TOKENS=4096
OPENAI_TEMPERATURE=0.1
OPENAI_TIMEOUT_SECONDS=60
OPENAI_RATE_LIMIT_DELAY=0.5

# Meta Muse Spark 1.3
META_API_KEY=sk-or-v1-meta-key
META_MODEL_ID=meta/muse-spark-1.3
META_MAX_TOKENS=8192
META_TEMPERATURE=0.2
META_TIMEOUT_SECONDS=90
META_RATE_LIMIT_DELAY=0.5

# Google Gemini 3.7 Flash
GOOGLE_API_KEY=sk-or-v1-google-key
GOOGLE_MODEL_ID=google/gemini-3.7-flash
GOOGLE_MAX_TOKENS=4096
GOOGLE_TEMPERATURE=0.15
GOOGLE_TIMEOUT_SECONDS=60
GOOGLE_RATE_LIMIT_DELAY=0.5
```

---

## 🚀 Running the Engine

```powershell
# Run all 3 models with live API keys
python main.py -f sample_data.json

# Run a specific model
python main.py -m openai
python main.py -m meta
python main.py -m google

# Run dry run preview (simulates text evaluation and binary decisions)
python main.py --dry-run
```

---

## 📂 Output Files Generated

* `output/output_openai_gpt_6_luna_decisions.json`
* `output/output_meta_muse_spark_1_3.json`
* `output/output_google_gemini_3_7_flash.json`
