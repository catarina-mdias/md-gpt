# views/exams_comparison.py
import os
import requests
import streamlit as st

EXAMS_API_BASE = os.getenv("EXAMS_COMPARISON_API_URL", "http://localhost:10002")
EXAMS_TEXT_MAX_CHARS = int(os.getenv("EXAM_COMPARISON_MAX_CHARS", "6000"))


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
        password = st.text_input("API Password (exam agent)", type="password", key="exam_api_password")
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
                    st.success("Login to Exams API successful.")
                else:
                    st.error(f"Login failed: {resp.status_code} - {resp.text}")
            except Exception as exc:
                st.error(f"Error calling Exams API /login: {exc}")


def _read_uploaded_text(uploaded_file) -> str:
    if not uploaded_file:
        return ""
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    if isinstance(raw, str):
        return raw
    for encoding in ("utf-8", "latin-1"):
        try:
            text = raw.decode(encoding)
            return text
        except Exception:
            continue
    return ""


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


def _process_exam_comparison(
    active_patient,
    exam_a_file,
    exam_b_file,
    exam_a_label,
    exam_b_label,
    exam_a_date,
    exam_b_date,
    focus_markers_str,
):
    if not exam_a_file or not exam_b_file:
        st.error("Please upload both Exam A and Exam B files.")
        return None

    exam_a_text = _read_uploaded_text(exam_a_file)
    exam_b_text = _read_uploaded_text(exam_b_file)

    if not exam_a_text or not exam_b_text:
        st.error("Could not read one of the files. Ensure they are text-based (txt/csv/pdf).")
        return None

    if len(exam_a_text) > EXAMS_TEXT_MAX_CHARS:
        st.info(
            f"Exam A text exceeds {EXAMS_TEXT_MAX_CHARS} characters; it will be truncated for analysis."
        )
    if len(exam_b_text) > EXAMS_TEXT_MAX_CHARS:
        st.info(
            f"Exam B text exceeds {EXAMS_TEXT_MAX_CHARS} characters; it will be truncated for analysis."
        )

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

    with st.spinner("Comparing exam reports..."):
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

    _render_list("Improvements / Positive Trends", result.get("key_improvements", []), "No improvements detected.")
    _render_list("Declines / Worsening Findings", result.get("key_declines", []), "No declines detected.")
    _render_list("Urgent Flags", result.get("urgent_flags", []), "No urgent flags.")
    _render_list("Recommendations", result.get("recommendations", []), "No specific recommendations.")


def show_exams_comparison(patients_today):
    active_patient = _get_active_patient(patients_today)

    _patient_header(active_patient)

    st.markdown("## Exams Comparison")
    st.caption(
        "Upload the baseline exam and the most recent exam (text, CSV, or simple PDF text) "
        "to generate a summary of changes using the MD-GPT exam comparison agent."
    )

    _login_exam_api()

    col_upload_a, col_upload_b = st.columns(2)
    with col_upload_a:
        st.subheader("Exam A (previous / baseline)")
        exam_a_file = st.file_uploader(
            "Upload baseline lab report",
            type=["txt", "csv", "pdf", "md"],
            key="exam_a_upload",
        )
        exam_a_label = st.text_input("Exam A Label", value="Previous Blood Exam")
        exam_a_date = st.text_input("Exam A Date", value="")
    with col_upload_b:
        st.subheader("Exam B (current)")
        exam_b_file = st.file_uploader(
            "Upload latest lab report",
            type=["txt", "csv", "pdf", "md"],
            key="exam_b_upload",
        )
        exam_b_label = st.text_input("Exam B Label", value="Latest Blood Exam")
        exam_b_date = st.text_input("Exam B Date", value="")

    focus_markers = st.text_input(
        "Optional: biomarkers to focus on (comma separated)",
        value="LDL, HDL, HbA1c, Creatinine",
    )

    comparison_result = None
    if st.button("Compare Exams", use_container_width=True):
        comparison_result = _process_exam_comparison(
            active_patient,
            exam_a_file,
            exam_b_file,
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
        "The exam comparison agent uses uploaded report text. Ensure documents are exported to text or CSV format."
    )