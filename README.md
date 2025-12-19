# MD-GPT Clinical Assistant

MD-GPT is an end-to-end demo of an AI-assisted clinical workflow. A Streamlit front end lets clinicians authenticate, pick a patient, manage appointments, and launch specialty agents. A single FastAPI backend (`md_gpt/api.py`) exposes those agents—built with LangChain/LangGraph, OpenAI models, and Langfuse tracing—as `/clinical-history`, `/appointment-summary`, and `/exams-comparison` endpoints that all read/write to the structured `clinical_data.json` database.

## Feature Highlights
- **Secure sign-in & session management** – Streamlit UI guard plus backend token issuance via `/login`.
- **Clinical dashboard** – Visual schedule, key metrics, and patient selector powering every view.
- **Appointment summarization** – Handles note-taking or audio uploads, calls `/appointment-summary`, and renders editable structured fields (symptoms, therapeutics, follow-up). Audio transcription uses Whisper via `OPENAI_TRANSCRIPTION_MODEL`.
- **Healthcare guardrails** – Guardrails’ `RestrictToTopic` validator enforces that both the appointment input and the generated summary stay strictly within clinical/medical topics.
- **Patient clinical history agent** – Lets providers pick categories (symptoms, labs, therapeutics, etc.) and detail level, then calls `/clinical-history` to synthesize the longitudinal story.
- **Exam comparison agent** – Reads lab/scan reports from the repo’s `exams/` folder (or another `EXAMS_DIR`) and produces a markdown table summarizing biomarker deltas through `/exams-comparison`.
- **MCP server** – `md_gpt/exams_mcp_server.py` exposes the repo exams over MCP for compatibility with other clients.

## Repository Layout
```
app.py                       # Streamlit UI entry point
md_gpt/api.py                # FastAPI app aggregating all agents + auth
md_gpt/clinical_data.json    # Sample structured EMR database
md_gpt/*_agent.py            # Agent logic (LangGraph / LangChain)
views/                       # Streamlit pages (dashboard, appointments, exams...)
appointment_recordings/      # Demo audio files mapped by patient_id
exams/                       # Text/PDF exam samples used by the comparison view
requirements.txt             # Shared runtime requirements
```

## Prerequisites
- Python 3.10+ (the agents rely on `langchain`, `fastapi`, and `streamlit` features that expect 3.10).
- An OpenAI API key with access to the configured chat & Whisper models.
- (Optional) Langfuse workspace credentials if you want trace logging.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Environment Configuration
Create a `.env` file in the project root (both the backend and Streamlit entry point call `load_dotenv()`):

```ini
# Front end login
STREAMLIT_UI_USERNAME=doctor
STREAMLIT_UI_PASSWORD=doctor-demo

# Backend auth (Streamlit calls /login using these)
AGENT_API_USERNAME=doctor
AGENT_API_PASSWORD=doctor-demo

# OpenAI + LangChain defaults
OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-4o-mini
OPENAI_TRANSCRIPTION_MODEL=whisper-1
OPENAI_TRANSCRIPTION_URL=https://api.openai.com/v1/audio/transcriptions

# Data & assets
PATIENTS_FILE=/absolute/path/to/md_gpt/clinical_data.json
EXAMS_DIR=/absolute/path/to/exams
DEFAULT_DOCTOR_NAME=Dr. MD-GPT
EXAM_COMPARISON_MAX_CHARS=6000

# Front end API routing
CLINICAL_API_URL=http://localhost:8000        # or set AGENT_API_BASE

# Langfuse (optional)
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_TRACING_ENVIRONMENT=development
```


## Running the Stack Locally
1. **Start the unified FastAPI backend**
   ```bash
   uvicorn md_gpt.api:app --host 0.0.0.0 --port 8000 --reload
   ```
   - Hosts `/login`, `/clinical-history`, `/appointment-summary`, `/exams-comparison`, and exposes the MCP server at `/mcp`.
   - Uses `AGENT_API_USERNAME` / `AGENT_API_PASSWORD` to validate credentials sent by the UI.

2. **Launch the Streamlit app**
   ```bash
   streamlit run app.py
   ```
   - Log in with the `STREAMLIT_UI_*` credentials. Successful login also requests an API token from the backend, which is stored in `st.session_state`.
   - Use the sidebar to select a patient, switch between the Dashboard, Appointment Summarization, Clinical History, or Exams Comparison views, and configure the backend base URL if it is not `http://localhost:8000`.

3. **Interact with the demo**
   - **Appointment Summarization**: Choose consent mode (audio vs. notes). In recording mode you may upload one of the sample `.m4a` files under `appointment_recordings/` (mapped via `APPOINTMENT_AUDIO_MAP`), transcribe via Whisper, and send to the agent. In notes mode, type notes and click *End Appointment* to create a structured summary stored in `clinical_data.json`.
   - **Clinical History**: Pick categories and detail level, then generate a longitudinal summary. Results persist in Streamlit session state for the selected patient.
   - **Exams Comparison**: Select any two reports present in `exams/`, optionally list biomarkers to focus on, and generate a markdown summary/table created by the comparison agent.

## Sample Data & Assets
- `md_gpt/clinical_data.json` – Synthetic patient + appointment + medication + exam tables. Extend it or point `PATIENTS_FILE` elsewhere to use real data.
- `appointment_recordings/` – Pre-recorded consultations used when consent allows audio.
- `exams/` – Example lab reports consumed by the comparison agent and the MCP server. Add more files (txt/pdf) to broaden the repository.

## Useful Commands
- `uvicorn md_gpt.api:app --reload` – Start/stop the backend quickly during development.
- `streamlit run app.py --server.port 8501` – Run the UI on a different port if needed.
- `pytest` (if you add tests) or `python -m md_gpt.exams_mcp_server` – Run the MCP server standalone for debugging.
- `curl -X POST http://localhost:8000/appointment-summary ...` – Smoke-test the Guardrails behavior (see below).

## Guardrails Validation
- The appointment agent (`md_gpt/appointment_summary_agent.py`) instantiates a Guard that allows only health/clinical topics and explicitly blocks unrelated domains (sports, entertainment, politics, etc.).
- The Guard validates two places:
  1. **Incoming notes/transcripts** – non-medical requests fail before the LLM runs.
  2. **Generated fields** – `summary`, `symptoms`, `diagnosis`, `therapeutics`, and `follow_up` all have to pass the health-topic check.
- Any violation raises `ValueError`, which FastAPI turns into `400 Bad Request` so clients can surface the error message.
- Test locally:
  ```bash
  # Obtain a token
  TOKEN=$(curl -s http://localhost:8000/login \
      -H "Content-Type: application/json" \
      -d '{"username":"doctor","password":"doctor-demo"}' | jq -r .token)

  # Valid (should return 200)
  curl -X POST http://localhost:8000/appointment-summary \
       -H "Content-Type: application/json" \
       -H "x_auth_token: $TOKEN" \
       -d '{"patient_id":"P001","input_mode":"notes","content":"Patient reports mild chest discomfort and elevated BP."}'

  # Invalid (should return 400 with Guardrails error)
  curl -X POST http://localhost:8000/appointment-summary \
       -H "Content-Type: application/json" \
       -H "x_auth_token: $TOKEN" \
       -d '{"patient_id":"P001","input_mode":"notes","content":"I love rock and roll and lasagna."}'
  ```

You should now have a fully local MD-GPT assistant capable of summarizing appointments, generating patient histories, and comparing exams end-to-end.
