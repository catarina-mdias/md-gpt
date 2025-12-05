"""
FastAPI clinical history summarization service with a LangGraph agent
over patient JSON data (no external tools).
"""

import os
from uuid import uuid4
from typing import Any, Dict, List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

import json
from pathlib import Path

# --- Langfuse imports -------------------------------------------------------
from langfuse import get_client
from langfuse.langchain import CallbackHandler


# ---------------------------------------------------------------------------
# Environment & basic setup
# ---------------------------------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

AGENT_API_USERNAME = os.getenv("AGENT_API_USERNAME")
AGENT_API_PASSWORD = os.getenv("AGENT_API_PASSWORD")

# Where to load patients from (your simplified JSON)
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
    title="Clinical History Agent API",
    description="Clinical history summarization agent for MD-GPT.",
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
# Patients loading helpers (from JSON)
# ---------------------------------------------------------------------------
def load_db_tables() -> Dict[str, Any]:
    """
    Load the new clinical database structure from PATIENTS_FILE.

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


def _split_csv_field(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def get_patient_record(patient_id: str) -> Dict[str, Any]:
    """
    Build a denormalized "patient record" from the new tables:

    {
      "patient": {...},                # basic demographics, chronic_conditions, allergies, tags
      "appointments": [...],          # list of appointments for this patient
      "medications": [...],           # list of meds for this patient
      "exams": [...],                 # list of exams for this patient
      "current_medications": [...],   # derived: meds with no end date
    }
    """
    tables = load_db_tables()
    patients = tables["patient"]
    meds = tables["medication_history"]
    appts = tables["appointment"]
    exams = tables["exams"]

    patient_row = next(
        (p for p in patients if p.get("patient_id") == patient_id),
        None,
    )
    if not patient_row:
        return {}

    patient_appts = [a for a in appts if a.get("patient_id") == patient_id]
    patient_meds = [m for m in meds if m.get("patient_id") == patient_id]
    patient_exams = [e for e in exams if e.get("patient_id") == patient_id]

    current_meds = [
        m
        for m in patient_meds
        if m.get("medication_end_date") in (None, "", "null")
    ]

    return {
        "patient": patient_row,
        "appointments": patient_appts,
        "medications": patient_meds,
        "exams": patient_exams,
        "current_medications": current_meds,
    }


# ---------------------------------------------------------------------------
# LangGraph agent for clinical history summarization
# ---------------------------------------------------------------------------

# Only two levels now: "low" and "high"
DetailLevel = Literal["low", "high"]
HistoryCategory = Literal["symptoms", "exams", "diagnosis", "therapeutics", "lab_results"]


class ClinicalHistoryState(TypedDict, total=False):
    # Input
    patient_id: str
    categories: List[HistoryCategory]
    detail_level: DetailLevel
    # Intermediate
    history_context: str
    # Output
    summary: str


llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0,
    api_key=OPENAI_API_KEY,
)


def build_history_context(
    patient_id: str,
    categories: List[HistoryCategory],
    detail_level: DetailLevel,
) -> str:
    """
    Build a textual context for the agent based on the NEW clinical_database.json schema.

    Uses:
      - database_tables.patient
      - database_tables.appointment
      - database_tables.medication_history
      - database_tables.exams
    """
    record = get_patient_record(patient_id)
    if not record:
        return f"No patient found with id {patient_id}."

    patient = record["patient"]
    appointments = record["appointments"]
    exams = record["exams"]
    current_meds = record["current_medications"]

    # ---- Patient overview ----
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

    # ---- Appointments + Exams ----
    lines.append("Previous appointments:\n")

    for appt in sorted(appointments, key=lambda x: x.get("appointment_date", "")):
        appt_id = appt.get("appointment_id")
        appt_date = appt.get("appointment_date")
        doctor = appt.get("appointment_doctor")
        reason = appt.get("reason_for_visit")

        lines.append(
            f"- Appointment {appt_id} on {appt_date} with {doctor}: {reason}\n"
        )

        # High-level summary
        summary = appt.get("appointment_summary")
        if summary:
            lines.append(f"  Summary: {summary}\n")

        # Symptoms
        if "symptoms" in categories:
            symptoms_text = appt.get("appointment_symptoms") or ""
            if symptoms_text:
                if detail_level == "low":
                    lines.append("  Symptoms: described in brief.\n")
                else:
                    # Split on ';' or ',' to make bullets if possible
                    raw_parts = [s.strip() for s in symptoms_text.replace(";", ",").split(",") if s.strip()]
                    if len(raw_parts) <= 1:
                        lines.append(f"  Symptoms: {symptoms_text}\n")
                    else:
                        lines.append("  Symptoms:\n")
                        for s in raw_parts:
                            lines.append(f"    - {s}\n")

        # Exams (from exams table, matched by patient_id + appointment_id)
        related_exams = [
            e for e in exams
            if e.get("appointment_id") == appt_id and e.get("patient_id") == patient_id
        ]

        if "exams" in categories and related_exams:
            if detail_level == "low":
                lines.append(f"  Exams: {len(related_exams)} exams mentioned.\n")
            else:
                lines.append("  Exams:\n")
                for ex in related_exams:
                    lines.append(
                        f"    - {ex.get('exam_type')} "
                        f"(date: {ex.get('exam_date')}, revised: {ex.get('exam_revised')})\n"
                    )

        # Lab results: we reuse exam entries as lab-type investigations
        if "lab_results" in categories and related_exams:
            if detail_level == "low":
                lines.append(f"  Lab results: {len(related_exams)} entries.\n")
            else:
                lines.append("  Lab results (from exam records):\n")
                for ex in related_exams:
                    lines.append(f"    - {ex.get('exam_type')}\n")

        # Diagnosis (string, may contain multiple codes separated by commas)
        if "diagnosis" in categories:
            diag_text = appt.get("diagnosis") or ""
            if diag_text:
                diag_items = _split_csv_field(diag_text)
                if detail_level == "low":
                    lines.append(f"  Diagnosis: {len(diag_items) or 1} items.\n")
                else:
                    lines.append("  Diagnosis / problems:\n")
                    if diag_items:
                        for d in diag_items:
                            lines.append(f"    - {d}\n")
                    else:
                        lines.append(f"    - {diag_text}\n")

        # Therapeutics (string)
        if "therapeutics" in categories:
            tx_text = appt.get("therapeutics") or ""
            if tx_text:
                tx_items = _split_csv_field(tx_text)
                if detail_level == "low":
                    lines.append(f"  Therapeutics: {len(tx_items) or 1} interventions.\n")
                else:
                    lines.append("  Therapeutics:\n")
                    if tx_items:
                        for t in tx_items:
                            lines.append(f"    - {t}\n")
                    else:
                        lines.append(f"    - {tx_text}\n")

        follow_up = appt.get("follow_up")
        if follow_up and detail_level != "low":
            lines.append(f"  Follow-up: {follow_up}\n")

        lines.append("\n")

    return "".join(lines)


def retrieve_history_node(state: ClinicalHistoryState) -> ClinicalHistoryState:
    patient_id = state["patient_id"]
    categories = state.get("categories") or [
        "symptoms",
        "exams",
        "diagnosis",
        "therapeutics",
        "lab_results",
    ]
    # Default to "high" detail if not provided
    detail_level = state.get("detail_level", "high")
    ctx = build_history_context(patient_id, categories, detail_level)
    return {"history_context": ctx}


def summarize_history_node(state: ClinicalHistoryState) -> ClinicalHistoryState:
    # Default to "high" detail if not provided
    detail_level = state.get("detail_level", "high")
    categories = state.get("categories") or [
        "symptoms",
        "exams",
        "diagnosis",
        "therapeutics",
        "lab_results",
    ]
    history_context = state["history_context"]

    # Only two levels now:
    # - "low": very concise + explicit patient overview instruction.
    # - "high": old "medium" behaviour (3–5 key bullet points per category).
    if detail_level == "low":
        detail_instruction = (
            "Start with a short patient overview (1–2 sentences). "
            "Then be very concise: 1–3 short bullet points per selected category."
        )
    else:  # "high"
        detail_instruction = (
            "Provide a succinct but informative summary. 3–5 key bullet points "
            "per selected category."
        )

    categories_str = ", ".join(categories)

    system_msg = SystemMessage(
        content=(
            "You are MD-GPT, a clinical assistant for medical doctors.\n"
            "You receive structured information about a patient's past appointments and must "
            "produce a clear, clinically oriented summary that can be quickly reviewed "
            "before or during an appointment.\n\n"
            "Only use information in the provided context. Do not invent diagnoses, exams, "
            "dates, medications, or follow-up plans that are not present.\n\n"
            "Organize the answer using these headings, but only if relevant to the data:\n"
            "- Patient overview\n"
            "- Symptoms\n"
            "- Exams\n"
            "- Lab results\n"
            "- Diagnosis / problems list\n"
            "- Therapeutics\n"
            "- Timeline of key events\n\n"
            f"{detail_instruction}\n"
            "Highlight obvious contraindications and safety-relevant aspects (e.g. allergies, "
            "kidney disease) and relate them to treatments when relevant."
        )
    )

    user_msg = HumanMessage(
        content=(
            "Generate a clinical history summary for this patient.\n\n"
            f"Selected categories: {categories_str}\n"
            f"Detail level: {detail_level}\n\n"
            "Here is the patient's history context:\n\n"
            f"{history_context}"
        )
    )

    response = llm.invoke([system_msg, user_msg])
    return {"summary": response.content}


def build_clinical_history_graph():
    graph = StateGraph(ClinicalHistoryState)
    graph.add_node("retrieve_history", retrieve_history_node)
    graph.add_node("summarize_history", summarize_history_node)
    graph.set_entry_point("retrieve_history")
    graph.add_edge("retrieve_history", "summarize_history")
    graph.add_edge("summarize_history", END)
    return graph.compile()


clinical_history_graph = build_clinical_history_graph()


def run_clinical_history_agent(
    patient_id: str,
    categories: Optional[List[HistoryCategory]] = None,
    detail_level: DetailLevel = "high",
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the LangGraph clinical history agent, with optional Langfuse tracing.

    - patient_id: patient identifier
    - categories: list of categories to include
    - detail_level: 'low' | 'high'
    - session_id: optional session key to group traces in Langfuse
    """
    initial: ClinicalHistoryState = {
        "patient_id": patient_id,
        "categories": categories
        or ["symptoms", "exams", "diagnosis", "therapeutics", "lab_results"],
        "detail_level": detail_level,
    }

    # If Langfuse is correctly configured, use it; otherwise run without callbacks
    if langfuse_client.auth_check():
        print("[Langfuse] Client authenticated. Tracing clinical-history run.")
        handler = CallbackHandler()
        config = {"callbacks": [handler]}

        with langfuse_client.start_as_current_span(name="🩺-clinical-history-agent") as span:
            span.update_trace(
                input={
                    "patient_id": patient_id,
                    "categories": initial["categories"],
                    "detail_level": detail_level,
                },
                session_id=session_id,
                metadata={"environment": LANGFUSE_TRACING_ENVIRONMENT},
            )
            result = clinical_history_graph.invoke(initial, config=config)
            span.update_trace(output=result.get("summary", ""))
    else:
        # No valid Langfuse config → just run the agent normally
        print("[Langfuse] Not configured or auth failed. Running without tracing.")
        result = clinical_history_graph.invoke(initial)

    return {
        "patient_id": patient_id,
        "categories": initial["categories"],
        "detail_level": detail_level,
        "summary": result["summary"],
    }


# ---------------------------------------------------------------------------
# Auth: simple token-based login
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
# Request / response models for the clinical history endpoint
# ---------------------------------------------------------------------------

class ClinicalHistoryRequest(BaseModel):
    patient_id: str
    categories: Optional[List[HistoryCategory]] = None
    detail_level: DetailLevel = "high"
    session_id: Optional[str] = None  # optional, for tracing grouping


class ClinicalHistoryResponse(BaseModel):
    patient_id: str
    categories: List[HistoryCategory]
    detail_level: DetailLevel
    summary: str


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


@app.post("/clinical-history", response_model=ClinicalHistoryResponse)
def clinical_history(
    payload: ClinicalHistoryRequest,
    username: str = Depends(verify_token),
) -> ClinicalHistoryResponse:
    """
    Main endpoint:
    - Streamlit passes patient_id, categories, detail_level ('low' | 'high'),
      and optionally session_id.
    - Requires valid x_auth_token header from /login.
    """
    result = run_clinical_history_agent(
        patient_id=payload.patient_id,
        categories=payload.categories,
        detail_level=payload.detail_level,
        session_id=payload.session_id,
    )

    return ClinicalHistoryResponse(
        patient_id=result["patient_id"],
        categories=result["categories"],
        detail_level=result["detail_level"],
        summary=result["summary"],
    )
