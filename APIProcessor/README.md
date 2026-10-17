# APIProcessor: Multi-Model Multimodal Binary Scam Evaluator

A Python pipeline that ingests a JSON dataset containing multimodal user interactions (`Persona`, `Prompt`, and local `Image` paths) and routes each row through three models on OpenRouter:

1. **OpenAI: GPT-6 Luna Decisions** (`openai/gpt-6-luna-decisions`)
2. **Meta: Muse Spark 1.3** (`meta/muse-spark-1.3`)
3. **Google: Gemini 3.7 Flash** (`google/gemini-3.7-flash`)

The models inspect the actual local image pixels (via base64 data URIs) and user prompts, passing a **strictly binary judgment** on whether there is evidence of a scam or not.

---

## 🔗 Model Links on OpenRouter

| Provider & Model | OpenRouter Model Identifier | OpenRouter Model Profile | Multimodal Vision Support |
| :--- | :--- | :--- | :--- |
| **OpenAI: GPT-6 Luna Decisions** | `openai/gpt-6-luna-decisions` | [OpenRouter Model Page](https://openrouter.ai/models/openai/gpt-6-luna-decisions) | Yes (text & up to 128 inline images, rapid rubric checks) |
| **Meta: Muse Spark 1.3** | `meta/muse-spark-1.3` | [OpenRouter Model Page](https://openrouter.ai/models/meta/muse-spark-1.3) | Yes (native visual document & UI reasoning) |
| **Google: Gemini 3.7 Flash** | `google/gemini-3.7-flash` | [OpenRouter Model Page](https://openrouter.ai/models/google/gemini-3.7-flash) | Yes (high-throughput native OCR and visual telemetry) |

---

## 📥 Input Data Format

The pipeline expects a JSON file where each row contains:
* **`Persona`**: Arbitrary category name (e.g. `Category_SeniorConsumer`, `Category_HomeCook`).
* **`Prompt`**: The user's query or instruction.
* **`Image`**: Relative or absolute path to a local image stored on disk (e.g. `images/usps_smishing.png`).

```json
[
  {
    "Persona": "Category_SeniorConsumer",
    "Prompt": "Is this text message real or is it a scam?",
    "Image": "images/usps_smishing.png"
  },
  {
    "Persona": "Category_HomeCook",
    "Prompt": "Can you read the measurements on this recipe card and list the ingredients for me?",
    "Image": "images/recipe_card.png"
  }
]
```

### 🖼️ Local Image Processing
* Local images from disk (`.png`, `.jpg`, `.webp`) are encoded as base64 data URIs (`data:image/png;base64,...`) and passed as multimodal image payloads to OpenRouter.
* Sample test images can be generated anytime via `python generate_sample_images.py`.

---

## 📤 Standardized Output JSON Schema

Each model writes its own separate JSON file in the `output/` directory with strictly binary judgments:

```json
[
  {
    "row_index": 0,
    "persona": "Category_SeniorConsumer",
    "judgment": "Scam",
    "is_scam": true,
    "confidence_percentage": 99.6,
    "reasoning": "Direct image reading confirms phishing URL 'usps-redelivery-portal.top' and spoofed package alert signature. Visual and textual evidence confirm smishing fraud."
  },
  {
    "row_index": 1,
    "persona": "Category_HomeCook",
    "judgment": "Not a scam",
    "is_scam": false,
    "confidence_percentage": 99.9,
    "reasoning": "OCR and image features confirm standard culinary recipe card (leeks, potatoes, broth). No scam indicators detected."
  }
]
```

---

## ⚙️ Configuration & Adjustable Limits

Create your `.env` file from [`.env.example`](file:///C:/Users/mobil/alan-code/APIProcessor/.env.example):
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
# Generate sample images locally (already generated in images/)
python generate_sample_images.py

# Run all 3 models with live API keys
python main.py -f sample_data.json

# Run a specific model
python main.py -m openai
python main.py -m meta
python main.py -m google

# Run dry run preview (simulates image reading and binary evaluation)
python main.py --dry-run
```

---

## 📂 Output Files Generated

* **[`output/output_openai_gpt_6_luna_decisions.json`](file:///C:/Users/mobil/alan-code/APIProcessor/output/output_openai_gpt_6_luna_decisions.json)**
* **[`output/output_meta_muse_spark_1_3.json`](file:///C:/Users/mobil/alan-code/APIProcessor/output/output_meta_muse_spark_1_3.json)**
* **[`output/output_google_gemini_3_7_flash.json`](file:///C:/Users/mobil/alan-code/APIProcessor/output/output_google_gemini_3_7_flash.json)**
