# exams_comparison.py (Streamlit view)
import os
from pathlib import Path

import requests
import streamlit as st

EXAMS_API_BASE = os.getenv("EXAMS_COMPARISON_API_URL", "http://localhost:10002")
EXAMS_TEXT_MAX_CHARS = int(os.getenv("EXAM_COMPARISON_MAX_CHARS", "6000"))

# Folder where the public repo exams live (same as EXAMS_DIR used by MCP)
EXAMS_REPO_DIR = Path(os.getenv("EXAMS_DIR", "exams")).resolve()


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


def _get_exam_api_base() -> str:
    return st.session_state.get("exams_api_base", EXAMS_API_BASE)


def _login_exam_api():
    if "exams_api_token" in st.session_state and st.session_state.exams_api_token:
        st.success("Logged in to Exams API")
        return

    base_url = _get_exam_api_base()
    with st.expander("Login to Exams Comparison API", expanded=True):
        username = st.text_input("API Username (exam agent)", key="exam_api_username")
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
                    token = resp.json().get("token")
                    st.session_state["exams_api_token"] = token
                    st.session_state["exams_api_base"] = base_url
                    st.success("Login to Exams API successful.")
                else:
                    st.error(f"Login failed: {resp.status_code} - {resp.text}")
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


# ---------- Repo exams helpers (instead of upload) ----------

def _list_repo_exams() -> list[Path]:
    """
    List all exam files from the repo folder (e.g., ./exams).
    This simulates the hospital system's stored exams.
    """
    if not EXAMS_REPO_DIR.exists():
        return []
    files = [
        p for p in sorted(EXAMS_REPO_DIR.rglob("*"))
        if p.is_file()
    ]
    return files


def _read_repo_exam_text(path: Path) -> str:
    """
    Read exam file content as text, using the same decoding
    strategy you previously used for uploaded files.
    """
    try:
        raw = path.read_bytes()
    except Exception as exc:
        st.error(f"Error reading exam file {path.name}: {exc}")
        return ""

    # For symmetry with the old upload logic
    if isinstance(raw, str):
        return raw

    for encoding in ("utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except Exception:
            continue

    return ""


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
            "Could not read one of the files. "
            "Ensure they are text-based (txt/csv or text-based PDF)."
        )
        return None

    exam_a_text = exam_a_text[:EXAMS_TEXT_MAX_CHARS]
    exam_b_text = exam_b_text[:EXAMS_TEXT_MAX_CHARS]

    markers = [m.strip() for m in (focus_markers_str or "").split(",") if m.strip()]

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

    with st.spinner("Comparing exam reports from repository..."):
        return _call_exam_comparison_api(payload)


def _render_comparison_results(result: dict) -> None:
    """
    Keep the same rendering as your original version:
    show only the markdown `summary` from the API.
    That summary is where the LLM will output the table.
    """
    st.markdown("### Summary")
    summary = result.get("summary") or "No summary provided."
    st.markdown(summary)


def show_exams_comparison(patients_today):
    active_patient = _get_active_patient(patients_today)
    _patient_header(active_patient)

    st.markdown("## Exams Comparison")
    st.caption(
        "Select the baseline exam and the most recent exam from the hospital repository "
        "(public exams folder in this project) to generate a tabular summary of changes "
        "using the MD-GPT exam comparison agent."
    )

    _login_exam_api()

    exam_files = _list_repo_exams()
    if not exam_files:
        st.warning(
            f"No exam files found in repository folder: {EXAMS_REPO_DIR}\n\n"
            "Add exam files to that folder in the repo and redeploy."
        )
        return

    # Build labels relative to EXAMS_REPO_DIR for nicer display
    labels_to_paths = {
        str(p.relative_to(EXAMS_REPO_DIR)): p for p in exam_files
    }
    all_labels = list(labels_to_paths.keys())

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Exam A (previous / baseline)")
        default_a_index = 0
        exam_a_choice = st.selectbox(
            "Select baseline exam from repository",
            options=all_labels,
            index=default_a_index,
            key="exam_a_repo_choice",
        )
        exam_a_label = st.text_input("Exam A Label", value="Previous Blood Exam")
        exam_a_date = st.text_input("Exam A Date", value="")
    with col_b:
        st.subheader("Exam B (current)")
        default_b_index = 1 if len(all_labels) > 1 else 0
        exam_b_choice = st.selectbox(
            "Select latest exam from repository",
            options=all_labels,
            index=default_b_index,
            key="exam_b_repo_choice",
        )
        exam_b_label = st.text_input("Exam B Label", value="Latest Blood Exam")
        exam_b_date = st.text_input("Exam B Date", value="")

    focus_markers = st.text_input(
        "Optional: biomarkers to focus on (comma separated)",
        value="LDL, HDL, HbA1c, Creatinine",
    )

    comparison_result = None
    if st.button("Compare Exams", use_container_width=True):
        exam_a_path = labels_to_paths.get(exam_a_choice)
        exam_b_path = labels_to_paths.get(exam_b_choice)

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
        "The exam comparison agent now uses exams stored in the repository "
        "(hospital-like system) and outputs a markdown table summarizing the "
        "lab marker differences."
    )
