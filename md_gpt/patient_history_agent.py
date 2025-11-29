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
PATIENTS_FILE = os.getenv("PATIENTS_FILE", "patients.json")

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

def load_patients_raw() -> Dict[str, Any]:
    """Load the raw JSON structure from PATIENTS_FILE."""
    path = Path(PATIENTS_FILE)
    if not path.exists():
        raise RuntimeError(f"Patients file not found at {path.resolve()}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_patients_by_id() -> Dict[str, Dict[str, Any]]:
    """
    Returns a dict keyed by patient_id:
    {
      "P001": {...},
      "P002": {...},
      ...
    }
    """
    data = load_patients_raw()
    patients_list: List[Dict[str, Any]] = data.get("patients", [])
    return {p["patient_id"]: p for p in patients_list}


# ---------------------------------------------------------------------------
# LangGraph agent for clinical history summarization
# ---------------------------------------------------------------------------

DetailLevel = Literal["low", "medium", "high"]
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
    Build a textual context for the agent based on JSON data.

    Expects the simplified schema:
      - patients[].appointments[].summary
      - ... .symptoms (list[str]), .exams, .diagnosis, .therapeutics, .lab_results, follow_up
    """
    patients = load_patients_by_id()
    patient = patients.get(patient_id)
    if not patient:
        return f"No patient found with id {patient_id}."

    lines: List[str] = []

    lines.append(
        f"Patient: {patient.get('name')} ({patient.get('sex')}), "
        f"DOB: {patient.get('date_of_birth')}.\n"
        f"Chronic conditions: {', '.join(patient.get('chronic_conditions', [])) or 'none'}.\n"
        f"Allergies: {', '.join(patient.get('allergies', [])) or 'none'}.\n"
        f"Current medication: {', '.join(patient.get('current_medication', [])) or 'none'}.\n\n"
    )

    lines.append("Previous appointments:\n")

    appointments: List[Dict[str, Any]] = patient.get("appointments", [])
    for appt in appointments:
        lines.append(
            f"- Appointment {appt.get('appointment_id')} on {appt.get('date')} "
            f"with {appt.get('doctor')}: {appt.get('reason_for_visit')}\n"
        )

        # Always include the appointment's high-level summary as first line
        summary = appt.get("summary")
        if summary:
            lines.append(f"  Summary: {summary}\n")

        if "symptoms" in categories:
            symptoms = appt.get("symptoms") or []
            if symptoms:
                if detail_level == "low":
                    lines.append(f"  Symptoms: {len(symptoms)} items described.\n")
                else:
                    lines.append("  Symptoms:\n")
                    for s in symptoms:
                        lines.append(f"    - {s}\n")

        if "exams" in categories:
            exams = appt.get("exams") or []
            if exams:
                if detail_level == "low":
                    lines.append(f"  Exams: {len(exams)} exams mentioned.\n")
                else:
                    lines.append("  Exams:\n")
                    for e in exams:
                        lines.append(f"    - {e}\n")

        if "lab_results" in categories:
            labs = appt.get("lab_results") or []
            if labs:
                if detail_level == "low":
                    lines.append(f"  Lab results: {len(labs)} entries.\n")
                else:
                    lines.append("  Lab results:\n")
                    for l in labs:
                        lines.append(f"    - {l}\n")

        if "diagnosis" in categories:
            diag = appt.get("diagnosis") or []
            if diag:
                if detail_level == "low":
                    lines.append(f"  Diagnosis: {len(diag)} items.\n")
                else:
                    lines.append("  Diagnosis / problems:\n")
                    for d in diag:
                        lines.append(f"    - {d}\n")

        if "therapeutics" in categories:
            tx = appt.get("therapeutics") or []
            if tx:
                if detail_level == "low":
                    lines.append(f"  Therapeutics: {len(tx)} interventions.\n")
                else:
                    lines.append("  Therapeutics:\n")
                    for t in tx:
                        lines.append(f"    - {t}\n")

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
    detail_level = state.get("detail_level", "medium")
    ctx = build_history_context(patient_id, categories, detail_level)
    return {"history_context": ctx}


def summarize_history_node(state: ClinicalHistoryState) -> ClinicalHistoryState:
    detail_level = state.get("detail_level", "medium")
    categories = state.get("categories") or [
        "symptoms",
        "exams",
        "diagnosis",
        "therapeutics",
        "lab_results",
    ]
    history_context = state["history_context"]

    if detail_level == "low":
        detail_instruction = (
            "Be very concise. 1–3 short bullet points per selected category."
        )
    elif detail_level == "high":
        detail_instruction = (
            "Be detailed, include relevant numbers, dates, and evolution when available, "
            "but stay clinically focused. Up to 5–8 bullets per category."
        )
    else:
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
    detail_level: DetailLevel = "medium",
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the LangGraph clinical history agent, with optional Langfuse tracing.

    - patient_id: patient identifier
    - categories: list of categories to include
    - detail_level: 'low' | 'medium' | 'high'
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
    detail_level: DetailLevel = "medium"
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
    - Streamlit passes patient_id, categories, detail_level, and optionally session_id.
    - Requires valid X-Auth-Token header from /login.
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
