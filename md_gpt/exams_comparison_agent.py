"""
FastAPI service for comparing two exam reports using an LLM agent.

- Exposes /login and /exams-comparison endpoints.
- Mounts an MCP server at /mcp (see exams_mcp_server.py).
- Accepts exam_a and exam_b as objects: {label, exam_date, content}.
- Uses header x_auth_token for authentication (matching Streamlit).
"""

import json
import os
import secrets
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# MCP server (lists repo exams, not strictly required for comparison endpoint)
from md_gpt.exams_mcp_server import mcp

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in the environment")

AGENT_API_USERNAME = os.getenv("AGENT_API_USERNAME", "doctor")
AGENT_API_PASSWORD = os.getenv("AGENT_API_PASSWORD", "password123")

TOKENS_FILE = Path(os.getenv("TOKENS_FILE", ".tokens.json"))

# ---------------------------------------------------------------------------
# TOKEN MANAGEMENT
# ---------------------------------------------------------------------------


def _load_tokens() -> Dict[str, str]:
    if TOKENS_FILE.exists():
        try:
            return json.loads(TOKENS_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_tokens(tokens: Dict[str, str]) -> None:
    TOKENS_FILE.write_text(json.dumps(tokens))


def create_token(username: str) -> str:
    token = secrets.token_urlsafe(32)
    tokens = _load_tokens()
    tokens[token] = username
    _save_tokens(tokens)
    return token


def verify_token(
    x_auth_token: str = Header(..., alias="x_auth_token", convert_underscores=False)
) -> str:
    """
    Dependency: verify the x_auth_token header.

    Streamlit calls:
        headers={"x_auth_token": token}
    """
    token = x_auth_token.strip()
    tokens = _load_tokens()
    username = tokens.get(token)
    if not username:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return username



# ---------------------------------------------------------------------------
# REQUEST / RESPONSE MODELS
# ---------------------------------------------------------------------------


class ExamData(BaseModel):
    label: Optional[str] = Field(
        None, description="Human-friendly label for this exam (e.g. 'Baseline')."
    )
    exam_date: Optional[str] = Field(
        None, description="Date string for exam (free format, e.g. '2025-03-21')."
    )
    content: str = Field(..., description="Full text of the exam report.")


class ExamComparisonRequest(BaseModel):
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    exam_a: ExamData
    exam_b: ExamData
    focus_markers: Optional[List[str]] = None
    session_id: Optional[str] = None


class ExamComparisonResponse(BaseModel):
    summary: str
    key_improvements: List[str]
    key_declines: List[str]
    urgent_flags: List[str]
    recommendations: List[str]


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    message: str
    token: str


# ---------------------------------------------------------------------------
# LLM AGENT LOGIC
# ---------------------------------------------------------------------------


def run_exam_comparison_agent(payload: ExamComparisonRequest) -> Dict[str, Any]:
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0.0,
    )

    focus_clause = ""
    if payload.focus_markers:
        focus_clause = (
            "Clinical focus markers (biomarkers / parameters of interest): "
            + ", ".join(payload.focus_markers)
            + "\n"
        )
    else:
        focus_clause = "No specific biomarkers provided; focus on clinically relevant changes.\n"

    system_msg = SystemMessage(
        content=(
            "You are a clinical data analyst specializing in longitudinal lab comparisons. "
            "Given two exam reports for (presumably) the same patient, highlight the most "
            "important differences, flag urgent issues, and recommend follow-up steps.\n\n"
            "Reply ONLY with valid JSON and the keys:\n"
            "- summary (string)\n"
            "- key_improvements (array of strings)\n"
            "- key_declines (array of strings)\n"
            "- urgent_flags (array of strings)\n"
            "- recommendations (array of strings)\n"
        )
    )

    human_content = (
        f"Patient ID: {payload.patient_id or 'N/A'}\n"
        f"Patient Name: {payload.patient_name or 'N/A'}\n"
        f"{focus_clause}\n"
        f"--- Exam A (previous/baseline) ---\n"
        f"Label: {payload.exam_a.label or 'Baseline'}\n"
        f"Date: {payload.exam_a.exam_date or 'N/A'}\n"
        f"Content:\n{payload.exam_a.content}\n\n"
        f"--- Exam B (current) ---\n"
        f"Label: {payload.exam_b.label or 'Current'}\n"
        f"Date: {payload.exam_b.exam_date or 'N/A'}\n"
        f"Content:\n{payload.exam_b.content}\n\n"
        "Compare Exam B to Exam A and produce the JSON structure described above."
    )

    human_msg = HumanMessage(content=human_content)

    result = llm.invoke([system_msg, human_msg])
    content = result.content

    # Handle some models returning a list of chunks
    if isinstance(content, list):
        content = "".join(str(chunk) for chunk in content)

    # Parse JSON robustly
    try:
        parsed = json.loads(content)
    except Exception:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise HTTPException(
                status_code=500,
                detail="LLM returned non-JSON content that could not be parsed.",
            )
        parsed = json.loads(content[start : end + 1])

    def _ensure_list(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(v) for v in value]
        if value is None:
            return []
        return [str(value)]

    response = ExamComparisonResponse(
        summary=str(parsed.get("summary", "")),
        key_improvements=_ensure_list(parsed.get("key_improvements")),
        key_declines=_ensure_list(parsed.get("key_declines")),
        urgent_flags=_ensure_list(parsed.get("urgent_flags")),
        recommendations=_ensure_list(parsed.get("recommendations")),
    )

    return response.model_dump()


# ---------------------------------------------------------------------------
# FASTAPI + MCP MOUNT
# ---------------------------------------------------------------------------

mcp_app = mcp.streamable_http_app()

app = FastAPI(
    title="Exams Comparison Agent API with MCP",
    description=(
        "Compare two lab exam reports using an LLM agent and expose an MCP server "
        "for listing/loading exam files from the repository."
    ),
    version="1.0.0",
    lifespan=mcp_app.router.lifespan_context,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/mcp", mcp_app, name="mcp")


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------


@app.post("/login", response_model=LoginResponse)
def login(credentials: LoginRequest) -> LoginResponse:
    if (
        credentials.username != AGENT_API_USERNAME
        or credentials.password != AGENT_API_PASSWORD
    ):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_token(credentials.username)
    return LoginResponse(message=f"Welcome back, {credentials.username}!", token=token)


@app.post("/exams-comparison", response_model=ExamComparisonResponse)
def compare_exams(
    payload: ExamComparisonRequest,
    username: str = Depends(verify_token),
) -> ExamComparisonResponse:
    """
    Compare two exams using the LLM agent.
    The authenticated username (from x_auth_token) is available as `username`.
    """
    result = run_exam_comparison_agent(payload)
    return ExamComparisonResponse(**result)


