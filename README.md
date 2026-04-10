# Oxford Clinical AI Hackathon – Clinical AI Assistant

An AI-powered assistant designed to help NHS doctors with **conversations**, **clinical frameworks**, and **admin** tasks.  
Originally developed as a Google Colab project for the Oxford Clinical AI Hackathon.

---

## Features

| Module | Capabilities |
|---|---|
| 🗣 **Conversations** | Multi-turn clinical chatbot · Differential diagnosis · Patient-friendly explanations · Treatment options |
| 📋 **Frameworks** | SOAP notes · SBAR handover · NEWS2 calculator · Discharge summaries · Referral letters |
| 🗂 **Admin** | Clinic / GP letters · Medication list formatter · MDT summaries · ICD-10 coding · Task reminders · Notes summariser |

---

## Quick Start

### Option A – Google Colab (recommended for exploration)

1. Open [`clinical_ai_assistant.ipynb`](clinical_ai_assistant.ipynb) in Google Colab.
2. Add your `OPENAI_API_KEY` to Colab **Secrets** (🔑 icon in the left panel).
3. Run all cells top-to-bottom.

> **No API key?** The app runs in **demo mode** with pre-written example outputs so you can explore all features without any cost.

### Option B – Local Python

```bash
# Clone the repo
git clone https://github.com/mckenna-dev/Oxford_Clinical_AI_Hackathon.git
cd Oxford_Clinical_AI_Hackathon

# Install dependencies
pip install -r requirements.txt

# (Optional) set your OpenAI API key
export OPENAI_API_KEY="sk-..."

# Launch the interactive CLI
python app.py
```

---

## Project Structure

```
Oxford_Clinical_AI_Hackathon/
├── app.py                        # Interactive CLI entry point
├── clinical_ai_assistant.ipynb   # Google Colab notebook
├── requirements.txt
├── modules/
│   ├── conversation.py           # Clinical chatbot, differential diagnosis
│   ├── frameworks.py             # SOAP, SBAR, NEWS2, referral letters
│   └── admin.py                  # Clinic letters, MDT summaries, ICD-10
├── utils/
│   └── llm_client.py             # OpenAI API wrapper with demo mode
└── tests/
    └── test_clinical_ai.py       # Pytest test suite
```

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

All tests run without an API key (demo mode).

---

## Configuration

| Environment variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key for live AI responses |

The default model is `gpt-4o`. Pass `model="gpt-3.5-turbo"` to `LLMClient()` to use a cheaper model.

---

## Disclaimer

This tool is intended to **support** clinical decision-making, not replace it.  
All output should be reviewed by a qualified clinician before acting upon it.  
Always follow your institution's guidelines and GMC good medical practice.
