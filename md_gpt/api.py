import os
from uuid import uuid4
from typing import Dict

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Import ONLY the agent logic + pydantic models, not their FastAPI apps
from md_gpt.appointment_summary_agent import (
    run_appointment_agent,
    AppointmentSummaryRequest,
    AppointmentSummaryResponse,
)  # noqa: E402
from md_gpt.patient_history_agent import (
    run_clinical_history_agent,
    ClinicalHistoryRequest,
    ClinicalHistoryResponse,
)  # noqa: E402
from md_gpt.exams_comparison_agent import (
    run_exam_comparison_agent,
    ExamComparisonRequest,
    ExamComparisonResponse,
)  # noqa: E402
from md_gpt.exams_mcp_server import mcp  # MCP server (repo exams)  # noqa: E402

load_dotenv()

# --------------------------------------------------------------------
# Shared env + auth configuration
# --------------------------------------------------------------------
AGENT_API_USERNAME = os.getenv("AGENT_API_USERNAME")
AGENT_API_PASSWORD = os.getenv("AGENT_API_PASSWORD")

if not AGENT_API_USERNAME or not AGENT_API_PASSWORD:
    print(
        "[Auth] Warning: AGENT_API_USERNAME or AGENT_API_PASSWORD is missing. "
        "Login will fail until both are set."
    )

# --------------------------------------------------------------------
# Build MCP ASGI app and reuse its lifespan context
# --------------------------------------------------------------------
mcp_app = mcp.streamable_http_app()

app = FastAPI(
    title="MD-GPT Unified API",
    description="Single FastAPI service exposing all MD-GPT agents.",
    lifespan=mcp_app.router.lifespan_context,
)

# Expose MCP under /mcp (for MCP clients)
app.mount("/mcp", mcp_app, name="mcp")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared in-memory token store
app.state.active_tokens: dict[str, str] = {}


# --------------------------------------------------------------------
# Auth: single /login + token dependency
# --------------------------------------------------------------------
class LoginRequest:
    """Simple request body for /login.

    We don't reuse the per-agent LoginRequest to keep API concerns centralized.
    """
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password


class LoginResponseModel:
    def __init__(self, message: str, token: str):
        self.message = message
        self.token = token


def create_token(username: str) -> str:
    return f"token-{uuid4()}-{username}"


def save_token(token: str, username: str) -> None:
    app.state.active_tokens[token] = username


def verify_token(
    x_auth_token: str = Header(..., convert_underscores=False),
) -> str:
    """
    Common auth dependency for all endpoints.
    Matches frontend headers: headers={"x_auth_token": token}
    """
    username = app.state.active_tokens.get(x_auth_token)
    if not username:
        raise HTTPException(status_code=401, detail="Missing or invalid authentication token")
    return username


# --------------------------------------------------------------------
# Health + login endpoints
# --------------------------------------------------------------------
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/login")
def login(credentials: dict) -> Dict[str, str]:
    """
    Unified login endpoint used by Streamlit (app.py).
    Body: {"username": "...", "password": "..."}
    Returns: {"message": "...", "token": "..."}
    """
    username = credentials.get("username")
    password = credentials.get("password")

    if not AGENT_API_USERNAME or not AGENT_API_PASSWORD:
        raise HTTPException(
            status_code=500,
            detail="Server credentials are not configured",
        )

    if username != AGENT_API_USERNAME or password != AGENT_API_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_token(username)
    save_token(token, username)
    return {
        "message": f"Welcome back, {username}!",
        "token": token,
    }


# --------------------------------------------------------------------
# Clinical history endpoint (reusing patient_history_agent logic)
# --------------------------------------------------------------------
@app.post("/clinical-history", response_model=ClinicalHistoryResponse)
def clinical_history(
    payload: ClinicalHistoryRequest,
    username: str = Depends(verify_token),
) -> ClinicalHistoryResponse:
    """
    Wrapper around run_clinical_history_agent.

    Request body schema & response schema are those defined in patient_history_agent.py.
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


# --------------------------------------------------------------------
# Appointment summarization endpoint (reusing appointment_summary_agent logic)
# --------------------------------------------------------------------
@app.post("/appointment-summary", response_model=AppointmentSummaryResponse)
def appointment_summary(
    payload: AppointmentSummaryRequest,
    username: str = Depends(verify_token),
) -> AppointmentSummaryResponse:
    """
    Wrapper around run_appointment_agent.

    Request/response models imported from appointment_summary_agent.py.
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


# --------------------------------------------------------------------
# Exams comparison endpoint (reusing exams_comparison_agent logic)
# --------------------------------------------------------------------
@app.post("/exams-comparison", response_model=ExamComparisonResponse)
def compare_exams(
    payload: ExamComparisonRequest,
    username: str = Depends(verify_token),
) -> ExamComparisonResponse:
    """
    Wrapper around run_exam_comparison_agent.

    Request/response models imported from exams_comparison_agent.py.
    """
    result = run_exam_comparison_agent(payload)
    return ExamComparisonResponse(**result)
