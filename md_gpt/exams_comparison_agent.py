"""
Exam comparison agent logic.

This module provides the pure LLM / LangChain logic to compare two lab exam
reports and return a single markdown summary (ideally a table).

It is intentionally **framework-agnostic**: there is no FastAPI app, no
authentication, and no MCP mounting here. All HTTP / auth concerns are handled
in the unified `api.py` module.

Public entry point:
    - run_exam_comparison_agent(payload: ExamComparisonRequest) -> dict
"""

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# --- Langfuse imports -------------------------------------------------------
from langfuse import get_client
from langfuse.langchain import CallbackHandler


# ---------------------------------------------------------------------------
# Environment & basic setup
# ---------------------------------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set.")

# Langfuse environment label (optional)
LANGFUSE_TRACING_ENVIRONMENT = os.getenv("LANGFUSE_TRACING_ENVIRONMENT", "development")

# Langfuse client (it will no-op if keys/host are not correctly set)
langfuse_client = get_client()

# Global LLM instance reused across calls
llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0,
    api_key=OPENAI_API_KEY,
)

# Maximum characters from each exam that will be sent to the LLM
MAX_EXAM_CHARS = int(os.getenv("EXAM_COMPARISON_MAX_CHARS", "6000"))


# ---------------------------------------------------------------------------
# Pydantic models (shared with FastAPI layer in api.py)
# ---------------------------------------------------------------------------


class ExamDocument(BaseModel):
    """
    A single exam report as free text, with some metadata used for context.
    """

    label: str
    exam_date: Optional[str] = None
    content: str


class ExamComparisonRequest(BaseModel):
    """
    Request schema for the exam comparison agent.

    - patient_id / patient_name: optional, used only as context in the prompt.
    - exam_a / exam_b: two exam documents to compare.
    - focus_markers: optional list of biomarkers to pay special attention to.
    - session_id: optional session key used only for Langfuse trace grouping.
    """

    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    exam_a: ExamDocument
    exam_b: ExamDocument
    focus_markers: Optional[List[str]] = None
    session_id: Optional[str] = None


class ExamComparisonResponse(BaseModel):
    """
    Response schema from the exam comparison agent.

    - patient_id: echoes the input patient_id (if provided).
    - summary: markdown string, ideally a table with the most relevant
      marker variations between Exam A and B.
    """

    patient_id: Optional[str]
    summary: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _truncate_text(text: str) -> str:
    """
    Truncate the exam text to MAX_EXAM_CHARS characters to keep the context
    manageable for the LLM.
    """
    if len(text) <= MAX_EXAM_CHARS:
        return text
    return text[:MAX_EXAM_CHARS] + "\n... [truncated]"


def _format_exam_document(doc: ExamDocument) -> str:
    """
    Format an exam document into a labeled block for the prompt.
    """
    header = f"{doc.label} (Date: {doc.exam_date or 'unknown'})"
    body = _truncate_text(doc.content.strip())
    return f"{header}\n{body}\n"


# ---------------------------------------------------------------------------
# Public agent entry point
# ---------------------------------------------------------------------------


def run_exam_comparison_agent(payload: ExamComparisonRequest) -> Dict[str, Any]:
    """
    Compare Exam B against Exam A using an LLM and return a single markdown
    summary (ideally a table) describing the most relevant lab marker changes.

    This preserves the behavior of the original agent:
    - Output is **not** structured JSON – just a markdown `summary` string.
    - The table format is recommended but not strictly enforced, so the agent
      can fall back to concise bullet points if necessary.

    Langfuse:
    - If Langfuse is correctly configured (auth_check passes), this function
      will trace the LLM call with a span named "🧪-exams-comparison-agent".
    - Otherwise, it just calls the LLM without callbacks.
    """
    # Build an optional clause focusing on specific biomarkers
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
            "Compare Exam B against Exam A and describe only the most significant lab marker variations. "
            "Focus on numerical or clearly stated changes, quantify the delta when possible, and avoid "
            "diagnoses or treatment recommendations. Respond with a single Markdown section that makes "
            "it easy to scan the differences. Prefer a TABLE with columns:\n"
            "Marker | Exam A | Exam B | Change (use arrows ⬆️/⬇️/→ and numeric differences when clear).\n"
            "If you cannot build a good table, you may fall back to concise bullet points, "
            "but try to output a markdown table whenever possible.\n"
            "Do NOT output JSON or additional sections."
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

    messages = [system_msg, HumanMessage(content=human_content)]

    # Default: no callbacks / no tracing
    summary: str

    if langfuse_client.auth_check():
        # Langfuse tracing path
        print("[Langfuse] Client authenticated. Tracing exams-comparison run.")
        handler = CallbackHandler()
        config = {"callbacks": [handler]}

        with langfuse_client.start_as_current_span(
            name="🧪-exams-comparison-agent"
        ) as span:
            span.update_trace(
                input={
                    "patient_id": payload.patient_id,
                    "patient_name": payload.patient_name,
                    "exam_a_label": payload.exam_a.label,
                    "exam_b_label": payload.exam_b.label,
                    "focus_markers": payload.focus_markers,
                },
                session_id=payload.session_id,
                metadata={"environment": LANGFUSE_TRACING_ENVIRONMENT},
            )
            response = llm.invoke(messages, config=config)
            summary = (response.content or "").strip()
            span.update_trace(output=summary)
    else:
        # No valid Langfuse config → just run the LLM normally
        print("[Langfuse] Not configured or auth failed. Running without tracing.")
        response = llm.invoke(messages)
        summary = (response.content or "").strip()

    return {
        "patient_id": payload.patient_id,
        "summary": summary,
    }
