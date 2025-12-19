"""
FastAPI appointment summarization service with a LangGraph agent
over the clinical_data.json (new schema).

It supports two input modes:
- "transcript": speech-to-text transcript of the consultation
- "notes": free-form notes typed by the doctor

The agent produces a brief structured summary for the appointment and
persists it into the clinical_data.json "appointment" table:
- appointment_summary
- appointment_symptoms
- diagnosis
- therapeutics
- follow_up
"""

import os
import json
from uuid import uuid4
from typing import Any, Dict, List, Literal, Optional, TypedDict

from pathlib import Path
from datetime import date

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

# --- Langfuse imports -------------------------------------------------------
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from guardrails import Guard
from guardrails.hub import RestrictToTopic


# ---------------------------------------------------------------------------
# Environment & basic setup
# ---------------------------------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

AGENT_API_USERNAME = os.getenv("AGENT_API_USERNAME")
AGENT_API_PASSWORD = os.getenv("AGENT_API_PASSWORD")

# Where to load clinical DB from (new schema)
PATIENTS_FILE = os.getenv("PATIENTS_FILE", "clinical_data.json")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set.")

if not AGENT_API_USERNAME or not AGENT_API_PASSWORD:
    print(
        "[Auth] Warning: AGENT_API_USERNAME or AGENT_API_PASSWORD is missing. "
        "Login will fail until both are set."
    )

# Langfuse environment label (optional)
LANGFUSE_TRACING_ENVIRONMENT = os.getenv("LANGFUSE_TRACING_ENVIRONMENT", "development")


# ---------------------------------------------------------------------------
# FastAPI app & CORS
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Appointment Summarization Agent API",
    description="Appointment summarization agent for MD-GPT.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adapt for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory token store (for demo / course project)
app.state.active_tokens: dict[str, str] = {}

# Langfuse client (it will no-op if keys/host are not correctly set)
langfuse_client = get_client()


# ---------------------------------------------------------------------------
# Helpers for clinical_data.json
# ---------------------------------------------------------------------------

def load_db_tables() -> Dict[str, Any]:
    """
    Load the clinical database structure from PATIENTS_FILE.

    Expected JSON shape:
    {
      "database_tables": {
        "patient": [...],
        "medication_history": [...],
        "appointment": [...],
        "exams": [...]
      }
    }
    """
    path = Path(PATIENTS_FILE)
    if not path.exists():
        raise RuntimeError(f"Clinical DB file not found at {path.resolve()}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    tables = raw.get("database_tables", {})
    return {
        "patient": tables.get("patient", []),
        "medication_history": tables.get("medication_history", []),
        "appointment": tables.get("appointment", []),
        "exams": tables.get("exams", []),
    }


def save_db_tables(tables: Dict[str, Any]) -> None:
    """
    Persist updated tables back into PATIENTS_FILE.
    """
    path = Path(PATIENTS_FILE)
    if not path.exists():
        raise RuntimeError(f"Clinical DB file not found at {path.resolve()}")

    # Load the whole JSON to preserve any extra metadata fields
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    raw["database_tables"] = raw.get("database_tables", {})
    raw["database_tables"]["patient"] = tables.get("patient", [])
    raw["database_tables"]["medication_history"] = tables.get("medication_history", [])
    raw["database_tables"]["appointment"] = tables.get("appointment", [])
    raw["database_tables"]["exams"] = tables.get("exams", [])

    with path.open("w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)


def _split_csv_field(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def get_patient_record(patient_id: str) -> Dict[str, Any]:
    """
    Build a small "patient record" for context:

    {
      "patient": {...},     # basic demographics, chronic_conditions, allergies, tags
      "appointments": [...],
      "current_medications": [...],   # meds with no end date
    }
    """
    tables = load_db_tables()
    patients = tables["patient"]
    meds = tables["medication_history"]
    appts = tables["appointment"]

    patient_row = next(
        (p for p in patients if p.get("patient_id") == patient_id),
        None,
    )
    if not patient_row:
        return {}

    patient_appts = [a for a in appts if a.get("patient_id") == patient_id]
    patient_meds = [m for m in meds if m.get("patient_id") == patient_id]

    current_meds = [
        m
        for m in patient_meds
        if m.get("medication_end_date") in (None, "", "null")
    ]

    return {
        "patient": patient_row,
        "appointments": patient_appts,
        "current_medications": current_meds,
    }


def persist_appointment_summary(
    patient_id: str,
    summary_text: str,
    symptoms: str,
    diagnosis: str,
    therapeutics: str,
    follow_up: str,
    appointment_id: Optional[str],
    appointment_date: Optional[str],
    appointment_doctor: Optional[str],
    reason_for_visit: Optional[str],
) -> str:
    """
    Update or create an appointment row in the clinical DB with the generated data.

    Writes:
    - appointment_summary
    - appointment_symptoms
    - diagnosis
    - therapeutics
    - follow_up

    Returns the appointment_id that was updated/created.
    """
    tables = load_db_tables()
    appts = tables["appointment"]

    # If appointment_id provided, try to update that row
    target_row = None
    if appointment_id:
        for row in appts:
            if row.get("appointment_id") == appointment_id:
                target_row = row
                break

    # If no matching row, create a new one
    if not target_row:
        new_id = appointment_id or f"APPT-{uuid4().hex[:8]}"
        today_str = date.today().isoformat()
        target_row = {
            "appointment_id": new_id,
            "patient_id": patient_id,
            "appointment_date": appointment_date or today_str,
            "appointment_doctor": appointment_doctor or "Unknown doctor",
            "reason_for_visit": reason_for_visit or "Unspecified reason",
            "appointment_symptoms": symptoms,
            "diagnosis": diagnosis,
            "therapeutics": therapeutics,
            "follow_up": follow_up,
            "appointment_summary": summary_text,
        }
        appts.append(target_row)
        appointment_id = new_id
    else:
        # Update fields on existing row
        target_row["appointment_summary"] = summary_text
        target_row["appointment_symptoms"] = symptoms
        target_row["diagnosis"] = diagnosis
        target_row["therapeutics"] = therapeutics
        target_row["follow_up"] = follow_up

        if appointment_date:
            target_row["appointment_date"] = appointment_date
        if appointment_doctor:
            target_row["appointment_doctor"] = appointment_doctor
        if reason_for_visit:
            target_row["reason_for_visit"] = reason_for_visit

    # Save tables back
    tables["appointment"] = appts
    save_db_tables(tables)

    return appointment_id


# ---------------------------------------------------------------------------
# LangGraph agent for appointment summarization
# ---------------------------------------------------------------------------

InputMode = Literal["transcript", "notes"]


class AppointmentSummaryState(TypedDict, total=False):
    # Input
    patient_id: str
    input_mode: InputMode
    raw_input: str
    appointment_id: Optional[str]
    appointment_date: Optional[str]
    appointment_doctor: Optional[str]
    reason_for_visit: Optional[str]
    # Intermediate
    patient_context: str
    # Output (structured)
    summary: str
    symptoms: str
    diagnosis: str
    therapeutics: str
    follow_up: str
    final_appointment_id: str


llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0,
    api_key=OPENAI_API_KEY,
)

# Guardrails setup to ensure generated summaries stay in the medical domain
HEALTH_TOPIC_GUARD = Guard().use(
    RestrictToTopic(
        valid_topics=[
            "health",
            "medicine",
            "patient care",
            "clinical diagnostics",
            "pharmacology",
            "symptoms and therapeutics",
            "treatment recommendations",
        ],
        invalid_topics=[
            "sports",
            "entertainment",
            "politics",
            "finance",
            "music",
            "general news",
        ],
        disable_classifier=True,
        disable_llm=False,
    )
)


def _validate_health_topic(field_name: str, value: Optional[str]) -> str:
    """
    Ensure Guardrails approves the generated text as health-related.
    Raising ValueError aborts the request upstream (FastAPI returns 400/500).
    """
    if not value or not value.strip():
        return value or ""

    try:
        HEALTH_TOPIC_GUARD.validate(value)
    except Exception as exc:
        raise ValueError(
            f"Guardrails rejected the {field_name} content as non-health related: {exc}"
        ) from exc

    return value


def build_patient_context(patient_id: str) -> str:
    """
    Build a short patient context (overview + key problems + current meds)
    from the clinical database.
    """
    record = get_patient_record(patient_id)
    if not record:
        return f"No patient found with id {patient_id}."

    patient = record["patient"]
    appointments = record["appointments"]
    current_meds = record["current_medications"]

    name = patient.get("patient_name")
    sex = patient.get("sex")  # "M"/"F"
    dob = patient.get("date_of_birth")

    chronic_list = _split_csv_field(patient.get("chronic_conditions"))
    allergy_list = _split_csv_field(patient.get("allergies"))
    tags_list = _split_csv_field(patient.get("patient_tags"))
    current_med_names = [m.get("medication_name") for m in current_meds if m.get("medication_name")]

    lines: List[str] = []

    lines.append(
        f"Patient: {name} ({sex}), DOB: {dob}.\n"
        f"Chronic conditions: {', '.join(chronic_list) or 'none'}.\n"
        f"Allergies: {', '.join(allergy_list) or 'none'}.\n"
        f"Current medication: {', '.join(current_med_names) or 'none'}.\n"
        f"Patient tags: {', '.join(tags_list) or 'none'}.\n\n"
    )

    if appointments:
        lines.append("Recent appointments (most recent last):\n")
        for appt in sorted(appointments, key=lambda x: x.get("appointment_date", ""))[-3:]:
            lines.append(
                f"- {appt.get('appointment_date')} with {appt.get('appointment_doctor')}: "
                f"{appt.get('reason_for_visit')} "
            )
            prev_summary = appt.get("appointment_summary")
            if prev_summary:
                lines.append(f"  Summary: {prev_summary[:200]}...\n")
        lines.append("\n")

    return "".join(lines)


def retrieve_context_node(state: AppointmentSummaryState) -> AppointmentSummaryState:
    ctx = build_patient_context(state["patient_id"])
    return {"patient_context": ctx}


def summarize_appointment_node(state: AppointmentSummaryState) -> AppointmentSummaryState:
    input_mode = state["input_mode"]
    raw_input = state["raw_input"]
    patient_context = state["patient_context"]

    if input_mode == "transcript":
        input_description = "This is a speech-to-text transcript of the consultation."
    else:
        input_description = "These are free-form notes entered by the doctor during the visit."

    system_msg = SystemMessage(
        content=(
            "You are MD-GPT, a clinical assistant that writes brief, structured appointment "
            "summaries for medical doctors.\n\n"
            "You receive:\n"
            "1) Stable patient context (past history, medications, tags).\n"
            "2) A single-visit transcript or doctor's notes.\n\n"
            "Your goal is to create a short summary for the current appointment, "
            "similar in length and style to the examples stored in an EHR field, "
            "and to extract concise text for these fields:\n"
            "- appointment_symptoms\n"
            "- diagnosis\n"
            "- therapeutics\n"
            "- follow_up\n\n"
            "IMPORTANT:\n"
            "- Keep the summary brief (roughly 3–6 sentences or bullets total).\n"
            "- Use clinical language but avoid unnecessary detail.\n"
            "- Only use information in the provided context; do not invent data.\n"
            "- If a field is not clearly present, return an empty string for that field.\n\n"
            "Return your answer ONLY as a JSON object with these keys:\n"
            "{\n"
            '  \"summary\": \"...\",\n'
            '  \"symptoms\": \"...\",\n'
            '  \"diagnosis\": \"...\",\n'
            '  \"therapeutics\": \"...\",\n'
            '  \"follow_up\": \"...\"\n'
            "}\n"
        )
    )

    user_msg = HumanMessage(
        content=(
            f"{input_description}\n\n"
            "=== Patient context ===\n"
            f"{patient_context}\n\n"
            "=== Current appointment content ===\n"
            f"{raw_input}\n\n"
            "Generate the brief summary and structured fields for THIS appointment only."
        )
    )

    response = llm.invoke([system_msg, user_msg])

    # Try to parse JSON output; fall back gracefully if needed
    try:
        parsed = json.loads(response.content)
    except Exception:
        # Fallback: put everything into summary, leave others blank
        return {
            "summary": response.content,
            "symptoms": "",
            "diagnosis": "",
            "therapeutics": "",
            "follow_up": "",
        }

    summary_text = str(parsed.get("summary", "")).strip()
    symptoms = str(parsed.get("symptoms", "")).strip()
    diagnosis = str(parsed.get("diagnosis", "")).strip()
    therapeutics = str(parsed.get("therapeutics", "")).strip()
    follow_up = str(parsed.get("follow_up", "")).strip()

    return {
        "summary": summary_text or response.content,
        "symptoms": symptoms,
        "diagnosis": diagnosis,
        "therapeutics": therapeutics,
        "follow_up": follow_up,
    }


def save_summary_node(state: AppointmentSummaryState) -> AppointmentSummaryState:
    patient_id = state["patient_id"]
    summary = state["summary"]
    symptoms = state.get("symptoms", "")
    diagnosis = state.get("diagnosis", "")
    therapeutics = state.get("therapeutics", "")
    follow_up = state.get("follow_up", "")
    appointment_id = state.get("appointment_id")
    appointment_date = state.get("appointment_date")
    appointment_doctor = state.get("appointment_doctor")
    reason_for_visit = state.get("reason_for_visit")

    final_id = persist_appointment_summary(
        patient_id=patient_id,
        summary_text=summary,
        symptoms=symptoms,
        diagnosis=diagnosis,
        therapeutics=therapeutics,
        follow_up=follow_up,
        appointment_id=appointment_id,
        appointment_date=appointment_date,
        appointment_doctor=appointment_doctor,
        reason_for_visit=reason_for_visit,
    )

    return {"final_appointment_id": final_id}


def build_appointment_graph():
    graph = StateGraph(AppointmentSummaryState)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("summarize", summarize_appointment_node)
    graph.add_node("save_summary", save_summary_node)

    graph.set_entry_point("retrieve_context")
    graph.add_edge("retrieve_context", "summarize")
    graph.add_edge("summarize", "save_summary")
    graph.add_edge("save_summary", END)
    return graph.compile()


appointment_graph = build_appointment_graph()


def run_appointment_agent(
    patient_id: str,
    input_mode: InputMode,
    raw_input: str,
    appointment_id: Optional[str] = None,
    appointment_date: Optional[str] = None,
    appointment_doctor: Optional[str] = None,
    reason_for_visit: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the appointment summarization agent, with optional Langfuse tracing.

    - patient_id: patient identifier
    - input_mode: 'transcript' | 'notes'
    - raw_input: STT transcript or doctor's notes
    - appointment_id: optional; if provided, that row is updated in the DB
    - appointment_date, appointment_doctor, reason_for_visit: optional metadata
    - session_id: optional session key to group traces in Langfuse
    """
    _validate_health_topic("appointment input", raw_input)

    initial: AppointmentSummaryState = {
        "patient_id": patient_id,
        "input_mode": input_mode,
        "raw_input": raw_input,
        "appointment_id": appointment_id,
        "appointment_date": appointment_date,
        "appointment_doctor": appointment_doctor,
        "reason_for_visit": reason_for_visit,
    }

    if langfuse_client.auth_check():
        print("[Langfuse] Client authenticated. Tracing appointment summarization run.")
        handler = CallbackHandler()
        config = {"callbacks": [handler]}

        with langfuse_client.start_as_current_span(name="🩺-appointment-summary-agent") as span:
            span.update_trace(
                input={
                    "patient_id": patient_id,
                    "input_mode": input_mode,
                    "appointment_id": appointment_id,
                    "appointment_date": appointment_date,
                    "appointment_doctor": appointment_doctor,
                    "reason_for_visit": reason_for_visit,
                },
                session_id=session_id,
                metadata={"environment": LANGFUSE_TRACING_ENVIRONMENT},
            )
            result = appointment_graph.invoke(initial, config=config)
            span.update_trace(output=result.get("summary", ""))
    else:
        print("[Langfuse] Not configured or auth failed. Running without tracing.")
        result = appointment_graph.invoke(initial)

    validated_fields = {
        name: _validate_health_topic(name, result.get(name, ""))
        for name in ("summary", "symptoms", "diagnosis", "therapeutics", "follow_up")
    }

    return {
        "patient_id": patient_id,
        "summary": validated_fields["summary"],
        "symptoms": validated_fields["symptoms"],
        "diagnosis": validated_fields["diagnosis"],
        "therapeutics": validated_fields["therapeutics"],
        "follow_up": validated_fields["follow_up"],
        "appointment_id": result["final_appointment_id"],
    }


# ---------------------------------------------------------------------------
# Auth: simple token-based login (same pattern as patient_history_agent)
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    message: str
    token: str


def create_token(username: str) -> str:
    return f"token-{uuid4()}-{username}"


def save_token(token: str, username: str) -> None:
    app.state.active_tokens[token] = username


def verify_token(
    x_auth_token: str = Header(..., convert_underscores=False),
) -> str:
    username = app.state.active_tokens.get(x_auth_token)
    if not username:
        raise HTTPException(status_code=401, detail="Missing or invalid authentication token")
    return username


# ---------------------------------------------------------------------------
# Request / response models for the appointment summary endpoint
# ---------------------------------------------------------------------------

class AppointmentSummaryRequest(BaseModel):
    patient_id: str
    input_mode: InputMode  # "transcript" | "notes"
    content: str           # STT transcript or doctor's notes
    appointment_id: Optional[str] = None
    appointment_date: Optional[str] = None  # ISO string "YYYY-MM-DD"
    appointment_doctor: Optional[str] = None
    reason_for_visit: Optional[str] = None
    session_id: Optional[str] = None


class AppointmentSummaryResponse(BaseModel):
    patient_id: str
    summary: str
    symptoms: str
    diagnosis: str
    therapeutics: str
    follow_up: str
    appointment_id: str  # final id after persistence


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/login", response_model=LoginResponse)
def login(credentials: LoginRequest) -> LoginResponse:
    if not AGENT_API_USERNAME or not AGENT_API_PASSWORD:
        raise HTTPException(
            status_code=500,
            detail="Server credentials are not configured",
        )

    if credentials.username != AGENT_API_USERNAME or credentials.password != AGENT_API_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_token(credentials.username)
    save_token(token, credentials.username)
    return LoginResponse(message=f"Welcome back, {credentials.username}!", token=token)


@app.post("/appointment-summary", response_model=AppointmentSummaryResponse)
def appointment_summary(
    payload: AppointmentSummaryRequest,
    username: str = Depends(verify_token),
) -> AppointmentSummaryResponse:
    """
    Main endpoint:
    - Streamlit passes patient_id, input_mode, content, optional metadata.
    - Requires valid x_auth_token header from /login.
    - Returns the generated summary and the appointment_id written to the DB.
    """
    try:
        result = run_appointment_agent(
            patient_id=payload.patient_id,
            input_mode=payload.input_mode,
            raw_input=payload.content,
            appointment_id=payload.appointment_id,
            appointment_date=payload.appointment_date,
            appointment_doctor=payload.appointment_doctor,
            reason_for_visit=payload.reason_for_visit,
            session_id=payload.session_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return AppointmentSummaryResponse(
        patient_id=result["patient_id"],
        summary=result["summary"],
        symptoms=result["symptoms"],
        diagnosis=result["diagnosis"],
        therapeutics=result["therapeutics"],
        follow_up=result["follow_up"],
        appointment_id=result["appointment_id"],
    )
