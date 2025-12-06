# views/exams_comparison.py
import os
from pathlib import Path

import requests
import streamlit as st

# ---- Config -----------------------------------------------------------------

EXAMS_API_BASE = os.getenv("EXAMS_COMPARISON_API_URL", "http://localhost:10002")
EXAMS_TEXT_MAX_CHARS = int(os.getenv("EXAM_COMPARISON_MAX_CHARS", "6000"))

# Folder in the repo where the “hospital system” exams live.
# This should match EXAMS_DIR used by your MCP server.
EXAMS_REPO_DIR = Path(os.getenv("EXAMS_DIR", "exams")).resolve()


# ---- Patient helpers --------------------------------------------------------

def _get_active_patient(patients_today):
    pid = st.session_state.get("selected_patient_id")
    for p in patients_today:
        if p["id"] == pid:
            return p
    return None


def _patient_header(active_patient):
    with st.container(border=True):
        if active_patient:
            st.markdown(f"### Active Patient: {active_patient['name']}")
            st.caption(
                f"ID: {active_patient['id']}  •  "
                f"Age: {active_patient['age']}  •  "
                f"Sex: {active_patient['sex']}  •  "
                "Key Tags: " + ", ".join(active_patient["tags"])
            )
        else:
            st.markdown("### No patient selected")
            st.caption(
                "Select an active patient from the sidebar "
                "to personalize this view."
            )


# ---- Exams API auth + calling -----------------------------------------------

def _get_exam_api_base() -> str:
    return st.session_state.get("exams_api_base", EXAMS_API_BASE)


def _login_exam_api():
    if "exams_api_token" in st.session_state and st.session_state.exams_api_token:
        st.success("Logged in to Exams API")
        return

    base_url = _get_exam_api_base()
    with st.expander("Login to Exams Comparison API", expanded=True):
        username = st.text_input(
            "API Username (exam agent)", key="exam_api_username"
        )
        password = st.text_input(
            "API Password (exam agent)",
            type="password",
            key="exam_api_password",
        )
        if st.button("Login to Exams API", key="login_exam_api_button"):
            if not username or not password:
                st.error("Please enter username and password.")
                return
            try:
                resp = requests.post(
                    f"{base_url}/login",
                    json={"username": username, "password": password},
                    timeout=10,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    token = data.get("token")
                    if not token:
                        st.error("Login succeeded but no token returned.")
                        return
                    st.session_state["exams_api_token"] = token
                    st.session_state["exams_api_base"] = base_url
                    st.success("Logged in to Exams API")
                else:
                    st.error(
                        f"Login failed ({resp.status_code}): {resp.text}"
                    )
            except Exception as exc:
                st.error(f"Error calling Exams API /login: {exc}")


def _call_exam_comparison_api(payload: dict) -> dict | None:
    token = st.session_state.get("exams_api_token")
    if not token:
        st.error("No Exams API token. Please log in first.")
        return None

    base_url = _get_exam_api_base()
    try:
        resp = requests.post(
            f"{base_url}/exams-comparison",
            json=payload,
            headers={"x_auth_token": token},
            timeout=60,
        )
    except Exception as exc:
        st.error(f"Error calling /exams-comparison endpoint: {exc}")
        return None

    if resp.status_code != 200:
        st.error(f"Exams API error {resp.status_code}: {resp.text}")
        return None

    try:
        return resp.json()
    except Exception as exc:
        st.error(f"Error parsing Exams API response JSON: {exc}")
        return None


# ---- Repo exam helpers (replace upload-based flow) --------------------------

def _list_repo_exams() -> list[Path]:
    """
    List all exam files in the repo folder.

    By default, this looks under ./exams relative to the project root,
    but you can override via EXAMS_DIR env var.
    """
    if not EXAMS_REPO_DIR.exists():
        return []

    # You can restrict to specific extensions if you prefer, e.g. "*.pdf".
    # Here we allow any file, but skip directories.
    files = [
        p for p in sorted(EXAMS_REPO_DIR.rglob("*"))
        if p.is_file()
    ]
    return files


def _read_repo_exam_text(path: Path) -> str:
    """
    Read a repo exam file as text, using the same decoding strategy
    that the old upload-based flow used.
    """
    try:
        raw = path.read_bytes()
    except Exception as exc:
        st.error(f"Error reading exam file {path.name}: {exc}")
        return ""

    # path.read_bytes() always returns bytes, but keep the old logic shape.
    if isinstance(raw, str):
        return raw

    for encoding in ("utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except Exception:
            continue

    return ""


# ---- Comparison pipeline ----------------------------------------------------

def _process_exam_comparison(
    active_patient,
    exam_a_path: Path,
    exam_b_path: Path,
    exam_a_label: str,
    exam_b_label: str,
    exam_a_date: str,
    exam_b_date: str,
    focus_markers_str: str,
):
    if not exam_a_path or not exam_b_path:
        st.error("Please select both Exam A and Exam B files.")
        return None

    if not exam_a_path.exists():
        st.error(f"Exam A file not found: {exam_a_path}")
        return None
    if not exam_b_path.exists():
        st.error(f"Exam B file not found: {exam_b_path}")
        return None

    exam_a_text = _read_repo_exam_text(exam_a_path)
    exam_b_text = _read_repo_exam_text(exam_b_path)

    if not exam_a_text or not exam_b_text:
        st.error(
            "Could not read one of the selected files. "
            "Ensure they are text-based (txt/csv or text-based PDF)."
        )
        return None

    if len(exam_a_text) > EXAMS_TEXT_MAX_CHARS:
        st.info(
            f"Exam A text exceeds {EXAMS_TEXT_MAX_CHARS} characters; "
            "it will be truncated for analysis."
        )
    if len(exam_b_text) > EXAMS_TEXT_MAX_CHARS:
        st.info(
            f"Exam B text exceeds {EXAMS_TEXT_MAX_CHARS} characters; "
            "it will be truncated for analysis."
        )

    # Truncate to max length
    exam_a_text = exam_a_text[:EXAMS_TEXT_MAX_CHARS]
    exam_b_text = exam_b_text[:EXAMS_TEXT_MAX_CHARS]

    markers = [
        m.strip()
        for m in (focus_markers_str or "").split(",")
        if m.strip()
    ]

    payload = {
        "patient_id": active_patient["id"] if active_patient else None,
        "patient_name": active_patient["name"] if active_patient else None,
        "exam_a": {
            "label": exam_a_label or "Baseline Exam",
            "exam_date": exam_a_date or None,
            "content": exam_a_text,
        },
        "exam_b": {
            "label": exam_b_label or "Current Exam",
            "exam_date": exam_b_date or None,
            "content": exam_b_text,
        },
        "focus_markers": markers or None,
        "session_id": st.session_state.get("session_id"),
    }

    with st.spinner("Comparing exam reports from hospital repository..."):
        return _call_exam_comparison_api(payload)


def _render_comparison_results(result: dict) -> None:
    st.markdown("### Summary")
    st.write(result.get("summary") or "No summary provided.")

    def _render_list(title: str, items: list[str], empty_text: str):
        st.markdown(f"#### {title}")
        if not items:
            st.caption(empty_text)
            return
        for item in items:
            st.markdown(f"- {item}")

    _render_list(
        "Improvements / Positive Trends",
        result.get("key_improvements", []),
        "No improvements detected.",
    )
    _render_list(
        "Declines / Worsening Findings",
        result.get("key_declines", []),
        "No declines detected.",
    )
    _render_list(
        "Urgent Flags",
        result.get("urgent_flags", []),
        "No urgent flags.",
    )
    _render_list(
        "Recommendations",
        result.get("recommendations", []),
        "No specific recommendations.",
    )


# ---- Main view --------------------------------------------------------------

def show_exams_comparison(patients_today):
    active_patient = _get_active_patient(patients_today)

    _patient_header(active_patient)

    st.markdown("## Exams Comparison")
    st.caption(
        "Select the baseline and most recent exams from the hospital repository "
        "(read-only exams folder in this project) to generate a summary of changes "
        "using the MD-GPT exam comparison agent."
    )

    # Login to backend comparison API
    _login_exam_api()

    # List repo exams for selection
    exam_files = _list_repo_exams()
    if not exam_files:
        st.warning(
            f"No exam files found in repository folder: {EXAMS_REPO_DIR}\n\n"
            "Add exam files to that folder in the repo and redeploy."
        )
        return

    # Build nice labels relative to EXAMS_REPO_DIR
    labels_to_paths = {
        str(p.relative_to(EXAMS_REPO_DIR)): p for p in exam_files
    }
    all_labels = list(labels_to_paths.keys())

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Exam A (previous / baseline)")
        default_a_index = 0
        exam_a_label_choice = st.selectbox(
            "Select baseline exam from repository",
            options=all_labels,
            index=default_a_index,
            key="exam_a_repo_choice",
        )
        exam_a_label = st.text_input(
            "Exam A Label",
            value="Previous Blood Exam",
            key="exam_a_label",
        )
        exam_a_date = st.text_input(
            "Exam A Date",
            value="",
            key="exam_a_date",
        )

    with col_b:
        st.subheader("Exam B (current)")
        # Try to pick a different default (e.g., the next file) if possible
        default_b_index = 1 if len(all_labels) > 1 else 0
        exam_b_label_choice = st.selectbox(
            "Select current exam from repository",
            options=all_labels,
            index=default_b_index,
            key="exam_b_repo_choice",
        )
        exam_b_label = st.text_input(
            "Exam B Label",
            value="Latest Blood Exam",
            key="exam_b_label",
        )
        exam_b_date = st.text_input(
            "Exam B Date",
            value="",
            key="exam_b_date",
        )

    focus_markers = st.text_input(
        "Optional: biomarkers to focus on (comma separated)",
        value="LDL, HDL, HbA1c, Creatinine",
    )

    comparison_result = None
    if st.button("Compare Exams", use_container_width=True):
        exam_a_path = labels_to_paths.get(exam_a_label_choice)
        exam_b_path = labels_to_paths.get(exam_b_label_choice)

        comparison_result = _process_exam_comparison(
            active_patient,
            exam_a_path,
            exam_b_path,
            exam_a_label,
            exam_b_label,
            exam_a_date,
            exam_b_date,
            focus_markers,
        )

    if comparison_result:
        st.success("Exam comparison generated successfully.")
        _render_comparison_results(comparison_result)

    st.markdown("---")
    st.caption(
        "The exam comparison agent uses uploaded blood test exams and highlights the main differences between two lab tests."
    )

