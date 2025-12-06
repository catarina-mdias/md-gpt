# views/clinical_history.py
import os
import requests
import streamlit as st

API_BASE_URL = os.getenv("CLINICAL_API_URL", "http://localhost:8000")


def _get_active_patient(patients_today):
    pid = st.session_state.get("selected_patient_id")
    for p in patients_today:
        if p["id"] == pid:
            return p
    return None


def _get_editable_tags(patient):
    """
    Keep per-patient editable key tags in session_state.

    Returns the current list of tags and also updates from any text input.
    """
    key = f"patient_tags_{patient['id']}"

    # Initialize from the original patient definition, the first time
    if key not in st.session_state:
        st.session_state[key] = list(patient.get("tags", []))

    current_tags = st.session_state[key]
    tags_str_default = ", ".join(current_tags)

    # Text input to edit tags (comma-separated)
    new_tags_str = st.text_input(
        "Key Tags (edit as needed, comma-separated)",
        value=tags_str_default,
        key=f"{key}_input",
        help="Example: Hypertension, Penicillin Allergy, Mild Gastritis",
    )

    # Normalize input back into a list
    new_tags = [t.strip() for t in new_tags_str.split(",") if t.strip()]

    # Update state if it changed
    if new_tags != current_tags:
        st.session_state[key] = new_tags
        current_tags = new_tags

    return current_tags


def _patient_header(active_patient):
    with st.container(border=True):
        if not active_patient:
            st.markdown("### No active patient")
            st.caption("Select a patient from the dashboard to view history.")
            return

        st.markdown(f"### Active Patient: {active_patient['name']}")
        st.caption(f"Age: {active_patient['age']}  •  Sex: {active_patient['sex']}")

        # 🔹 Editable Key Tags
        editable_tags = _get_editable_tags(active_patient)
        st.caption("Current Key Tags: " + (", ".join(editable_tags) or "None"))


def _call_clinical_history_api(
    patient_id: str,
    categories: list[str],
    detail_level: str,
    session_id: str | None = None,
) -> str:
    """Call the FastAPI /clinical-history endpoint and return the summary text."""
    token = st.session_state.get("api_token")
    if not token:
        st.error(
            "No API token available. The unified MD-GPT API login is done on the first page; "
            "please log out and sign in again if this persists."
        )
        return ""

    # Use the same base URL configured in app.py sidebar
    base_url = st.session_state.get("api_base", API_BASE_URL)

    payload = {
        "patient_id": patient_id,
        "categories": categories,
        "detail_level": detail_level,  # "low" | "high"
        "session_id": session_id,
    }

    try:
        resp = requests.post(
            f"{base_url}/clinical-history",
            json=payload,
            headers={"x_auth_token": token},
            timeout=30,
        )
    except Exception as e:
        st.error(f"Error calling clinical-history endpoint: {e}")
        return ""

    if resp.status_code != 200:
        st.error(f"API error {resp.status_code}: {resp.text}")
        return ""

    data = resp.json()
    return data.get("summary", "")


def show_clinical_history_page(patients_today):
    active_patient = _get_active_patient(patients_today)
    _patient_header(active_patient)

    if not active_patient:
        return

    st.markdown("## Patient Clinical History Summarization")
    st.caption(
        "Summarize the patient's clinical history based on selected categories and detail level, "
        "using the MD-GPT unified API."
    )

    col_left, col_right = st.columns([2, 2])

    # ---------- Category selection (matches backend categories) ----------
    with col_left:
        with st.container(border=True):
            st.subheader("Select Categories to Include")

            include_symptoms = st.checkbox("Symptoms", value=True)
            include_exams = st.checkbox("Exams", value=True)
            include_diagnosis = st.checkbox("Diagnosis / Problems", value=True)
            include_therapeutics = st.checkbox("Therapeutics", value=True)
            include_lab_results = st.checkbox("Lab Results", value=True)

    # ---------- Detail level ----------
    with col_right:
        with st.container(border=True):
            st.subheader("Level of Detail")

            detail_level_label = st.radio(
                "",
                options=[
                    "Low detail",
                    "High detail",
                ],
                index=1,  # default: High detail
            )

    # Map UI label to API detail level ("low" | "high")
    if detail_level_label == "Low detail":
        api_detail_level = "low"
    else:
        api_detail_level = "high"

    st.markdown("")

    # ---------- Generate Summary button ----------
    if st.button(
        "Generate Summary",
        key="generate_history_summary_real",
        use_container_width=True,
    ):
        categories = []
        if include_symptoms:
            categories.append("symptoms")
        if include_exams:
            categories.append("exams")
        if include_diagnosis:
            categories.append("diagnosis")
        if include_therapeutics:
            categories.append("therapeutics")
        if include_lab_results:
            categories.append("lab_results")

        if not categories:
            st.warning("Please select at least one category.")
        else:
            with st.spinner("Generating patient clinical history summary..."):
                session_id = f"clinical-history-{active_patient['id']}"
                summary_text = _call_clinical_history_api(
                    patient_id=active_patient["id"],
                    categories=categories,
                    detail_level=api_detail_level,
                    session_id=session_id,
                )

            if summary_text:
                st.session_state[f"history_summary_{active_patient['id']}"] = summary_text

    # ---------- Show summary if available ----------
    if active_patient:
        summary_key = f"history_summary_{active_patient['id']}"
        if summary_key in st.session_state:
            st.markdown("---")
            st.markdown("### Generated History Summary")
            st.markdown(
                st.session_state[summary_key],
                unsafe_allow_html=True,
            )
