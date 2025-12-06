"""
FastAPI service for comparing two lab exam reports using a LangChain LLM agent.
"""

import os
import json
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

AGENT_API_USERNAME = os.getenv("AGENT_API_USERNAME")
AGENT_API_PASSWORD = os.getenv("AGENT_API_PASSWORD")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set.")

if not AGENT_API_USERNAME or not AGENT_API_PASSWORD:
    print(
        "[Auth] Warning: AGENT_API_USERNAME or AGENT_API_PASSWORD is missing. "
        "Login will fail until both are set."
    )

app = FastAPI(
    title="Exam Comparison Agent API",
    description="Compare two exam reports and summarize key differences.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.active_tokens: dict[str, str] = {}

llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0,
    api_key=OPENAI_API_KEY,
)


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    message: str
    token: str


MAX_EXAM_CHARS = int(os.getenv("EXAM_COMPARISON_MAX_CHARS", "6000"))


class ExamDocument(BaseModel):
    label: str
    exam_date: Optional[str] = None
    content: str


class ExamComparisonRequest(BaseModel):
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    exam_a: ExamDocument
    exam_b: ExamDocument
    focus_markers: Optional[List[str]] = None
    session_id: Optional[str] = None


class ExamComparisonResponse(BaseModel):
    patient_id: Optional[str]
    summary: str
    key_improvements: List[str]
    key_declines: List[str]
    urgent_flags: List[str]
    recommendations: List[str]


def create_token(username: str) -> str:
    from uuid import uuid4

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


def _truncate_text(text: str) -> str:
    if len(text) <= MAX_EXAM_CHARS:
        return text
    return text[:MAX_EXAM_CHARS] + "\n... [truncated]"


def _format_exam_document(doc: ExamDocument) -> str:
    header = f"{doc.label} (Date: {doc.exam_date or 'unknown'})"
    body = _truncate_text(doc.content.strip())
    return f"{header}\n{body}\n"


def run_exam_comparison_agent(payload: ExamComparisonRequest) -> Dict[str, Any]:
    focus_clause = ""
    if payload.focus_markers:
        focus_clause = (
            "Focus especially on these lab markers when comparing: "
            + ", ".join(payload.focus_markers)
            + ".\n"
        )

    system_msg = SystemMessage(
        content=(
            "You are a clinical data analyst specializing in longitudinal lab comparisons. "
            "Given two exam reports for the same patient, highlight the most important differences, "
            "flag urgent issues, and recommend follow-up steps. Respond in valid JSON with the keys: "
            "summary (string), key_improvements (array), key_declines (array), urgent_flags (array), "
            "recommendations (array). Keep outputs concise but actionable."
        )
    )

    human_content = (
        f"Patient ID: {payload.patient_id or 'N/A'}\n"
        f"Patient Name: {payload.patient_name or 'N/A'}\n"
        f"{focus_clause}\n"
        f"--- Exam A ---\n{_format_exam_document(payload.exam_a)}\n"
        f"--- Exam B ---\n{_format_exam_document(payload.exam_b)}\n"
        "Compare Exam B against Exam A."
    )

    response = llm.invoke([system_msg, HumanMessage(content=human_content)])

    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError:
        parsed = {
            "summary": response.content,
            "key_improvements": [],
            "key_declines": [],
            "urgent_flags": [],
            "recommendations": [],
        }

    return {
        "patient_id": payload.patient_id,
        "summary": parsed.get("summary", ""),
        "key_improvements": parsed.get("key_improvements", []),
        "key_declines": parsed.get("key_declines", []),
        "urgent_flags": parsed.get("urgent_flags", []),
        "recommendations": parsed.get("recommendations", []),
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/login", response_model=LoginResponse)
def login(credentials: LoginRequest) -> LoginResponse:
    if not AGENT_API_USERNAME or not AGENT_API_PASSWORD:
        raise HTTPException(status_code=500, detail="Server credentials are not configured")

    if credentials.username != AGENT_API_USERNAME or credentials.password != AGENT_API_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_token(credentials.username)
    save_token(token, credentials.username)
    return LoginResponse(message=f"Welcome back, {credentials.username}!", token=token)


@app.post("/exams-comparison", response_model=ExamComparisonResponse)
def compare_exams(
    payload: ExamComparisonRequest,
    username: str = Depends(verify_token),
) -> ExamComparisonResponse:
    result = run_exam_comparison_agent(payload)
    return ExamComparisonResponse(**result)