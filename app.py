# app.py
import os
import json
from pathlib import Path
from datetime import date, datetime
import uuid

import requests
import streamlit as st
from dotenv import load_dotenv

from views.dashboard import show_dashboard
from views.exams_comparison import show_exams_comparison
from views.appointment import show_appointment_page
from views.clinical_history import show_clinical_history_page

load_dotenv()

st.set_page_config(
    page_title="MD-GPT Clinical Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------- UI Login Config ---------
STREAMLIT_USERNAME = os.getenv("STREAMLIT_UI_USERNAME", "doctor")
STREAMLIT_PASSWORD = os.getenv("STREAMLIT_UI_PASSWORD", "doctor-demo")

# --------- Session State Initialization (auth-related) ---------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "api_token" not in st.session_state:
    st.session_state.api_token = None
if "api_base" not in st.session_state:
    # prefer CLINICAL_API_URL; fallback to AGENT_API_BASE; then localhost
    st.session_state.api_base = os.getenv(
        "CLINICAL_API_URL",
        os.getenv("AGENT_API_BASE", "http://localhost:8000"),
    )
if "login_notice" not in st.session_state:
    st.session_state.login_notice = None
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


def authenticate_api(username: str, password: str) -> bool:
    """Request an auth token from the FastAPI backend."""
    url = f"{st.session_state.api_base}/login"
    try:
        response = requests.post(
            url,
            json={"username": username, "password": password},
            timeout=10,
        )
        response.raise_for_status()
        token = response.json().get("token")
        if not token:
            st.warning(
                "Login to API succeeded but no token returned. "
                "Agent calls may fail; check your backend."
            )
            st.session_state.api_token = None
            return False
        st.session_state.api_token = token
        return True
    except Exception as exc:
        st.warning(
            f"Could not authenticate with the FastAPI service ({exc}). "
            "Agent calls may not work until the API is reachable."
        )
        st.session_state.api_token = None
        return False


def require_login() -> None:
    """Prompt for credentials before rendering the main app."""
    if st.session_state.authenticated:
        return

    login_placeholder = st.empty()
    feedback_placeholder = st.empty()

    with login_placeholder:
        st.title("MD-GPT Login")
        st.write("Enter your credentials to access the MD-GPT Clinical Assistant.")

        with st.form("login-form", clear_on_submit=False):
            username_input = st.text_input("Username")
            password_input = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in")

    if submitted:
        if username_input == STREAMLIT_USERNAME and password_input == STREAMLIT_PASSWORD:
            st.session_state.username = username_input
            st.session_state.session_id = str(uuid.uuid4())
            api_ready = authenticate_api(username_input, password_input)
            if api_ready:
                st.session_state.login_notice = (
                    "success",
                    "Login successful. Loading the MD-GPT interface…",
                )
            else:
                st.session_state.login_notice = (
                    "info",
                    "Logged in. Backend API could not be fully authenticated; "
                    "check API credentials or availability.",
                )
            st.session_state.authenticated = True
            login_placeholder.empty()
            feedback_placeholder.empty()
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()
        else:
            feedback_placeholder.error("Invalid username or password. Try again.")

    # Stop rendering the rest of the app until authenticated
    st.stop()


# Require login before anything else
require_login()

# --------- Show login notice once, if any ---------
notice = st.session_state.login_notice
if notice:
    level, message = notice
    if level == "success":
        st.success(message)
    elif level == "info":
        st.info(message)
    elif level == "warning":
        st.warning(message)
    elif level == "error":
        st.error(message)
    st.session_state.login_notice = None

# --------- Helpers to load real patients from JSON ---------
def _compute_age(dob_str: str) -> int:
    try:
        dob = datetime.strptime(dob_str, "%Y-%m-%d").date()
        today = date.today()
        age = today.year - dob.year - (
            (today.month, today.day) < (dob.month, dob.day)
        )
        return age
    except Exception:
        return 0


def _split_csv_field(value: str | None) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def load_patients_for_ui():
    """
    Load clinical_database.json and adapt the patient table
    to the structure the views expect.

    Uses: database_tables.patient
    """
    patients_file_env = os.getenv("PATIENTS_FILE")
    if patients_file_env:
        path = Path(patients_file_env)
    else:
        # Fallbacks relative to this file
        root = Path(__file__).resolve().parent
        candidate_root = root / "clinical_data.json"
        candidate_md_gpt = root / "md_gpt" / "clinical_data.json"

        if candidate_root.exists():
            path = candidate_root
        elif candidate_md_gpt.exists():
            path = candidate_md_gpt
        else:
            raise RuntimeError(
                "clinical_data.json not found. Looked for:\n"
                f"- {candidate_root}\n"
                f"- {candidate_md_gpt}\n"
                "Set PATIENTS_FILE in your .env if it's stored elsewhere."
            )

    if not path.exists():
        raise RuntimeError(f"clinical_data.json not found at {path.resolve()}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    tables = raw.get("database_tables", {})
    patients_table = tables.get("patient", [])

    ui_patients = []
    base_hour = 9

    for idx, p in enumerate(patients_table):
        pid = p.get("patient_id")
        name = p.get("patient_name")
        dob = p.get("date_of_birth", "1970-01-01")
        sex_raw = p.get("sex", "U")
        sex = {"M": "Male", "F": "Female"}.get(sex_raw, "Unknown")

        chronic_list = _split_csv_field(p.get("chronic_conditions"))
        allergy_list = _split_csv_field(p.get("allergies"))
        tags_list = _split_csv_field(p.get("patient_tags"))

        # Simple fake schedule just for UI
        time_str = f"{base_hour + idx:02d}:00"
        main_reason = chronic_list[0] if chronic_list else "General check-up"
        reason = f"Follow-up: {main_reason}"

        ui_patients.append(
            {
                "id": pid,
                "name": name,
                "age": _compute_age(dob),
                "sex": sex,
                # Tags used by dashboard + clinical history view
                "tags": tags_list or (chronic_list + allergy_list),
                "time": time_str,
                "reason": reason,
            }
        )

    return ui_patients


PATIENTS = load_patients_for_ui()

# --------- Session State Initialization (existing app state) ---------
if "page" not in st.session_state:
    st.session_state.page = "Clinical Dashboard"

if "selected_patient_id" not in st.session_state and PATIENTS:
    st.session_state.selected_patient_id = PATIENTS[0]["id"]  # default first


def get_active_patient():
    pid = st.session_state.get("selected_patient_id")
    for p in PATIENTS:
        if p["id"] == pid:
            return p
    return None


# --------- Sidebar Layout ---------
with st.sidebar:
    st.markdown("### MD-GPT")
    st.caption("Clinical Assistant")

    pages = (
        "Clinical Dashboard",
        "Appointment Summarization",
        "Clinical History",
        "Exams Comparison (OCR)",
    )

    if "page" not in st.session_state:
        st.session_state["page"] = pages[0]

    nav_choice = st.radio(
        "Navigation",
        pages,
        index=pages.index(st.session_state["page"]),
    )
    st.session_state["page"] = nav_choice

    st.markdown("---")
    active_patient = get_active_patient()
    if active_patient:
        st.markdown("**Active Patient**")
        st.markdown(f"{active_patient['name']}")
        st.caption(f"Age: {active_patient['age']}  •  Sex: {active_patient['sex']}")
        st.caption("Key Tags: " + ", ".join(active_patient["tags"]))

    st.markdown("---")
    st.header("Account")
    st.caption(f"Signed in as: {st.session_state.username or STREAMLIT_USERNAME}")
    if st.button("Log out"):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.session_state.api_token = None
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.page = "Clinical Dashboard"
        st.rerun()

    st.header("API Setup")
    st.session_state.api_base = st.text_input(
        label="FastAPI base URL",
        value=st.session_state.api_base,
        help="Use your clinical-history API URL (e.g. http://localhost:8000).",
    )
    st.caption(f"Current session: {st.session_state.session_id}")

# --------- Main Content Routing ---------
page = st.session_state["page"]

if page == "Clinical Dashboard":
    show_dashboard(PATIENTS)
elif page == "Appointment Summarization":
    show_appointment_page(PATIENTS)
elif page == "Clinical History":
    show_clinical_history_page(PATIENTS)
elif page == "Exams Comparison (OCR)":
    show_exams_comparison(PATIENTS)
