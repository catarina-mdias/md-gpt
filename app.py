# app.py
import streamlit as st

from views.dashboard import show_dashboard
from views.exams_comparison import show_exams_comparison
from views.appointment import show_appointment_page
from views.clinical_history import show_clinical_history_page

st.set_page_config(
    page_title="MD-GPT Clinical Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------- Mock Patients for Today ---------
MOCK_PATIENTS = [
    {
        "id": "101",
        "name": "Jane Doe",
        "age": 45,
        "sex": "Female",
        "tags": ["Hypertension", "Penicillin Allergy", "Mild Gastritis", "Lisinopril"],
        "time": "09:00",
        "reason": "Hypertension follow-up",
    },
    {
        "id": "102",
        "name": "John Smith",
        "age": 58,
        "sex": "Male",
        "tags": ["Type 2 Diabetes", "Statin Intolerance"],
        "time": "10:30",
        "reason": "Diabetes review",
    },
    {
        "id": "103",
        "name": "Ana García",
        "age": 39,
        "sex": "Female",
        "tags": ["Asthma", "NSAID Allergy"],
        "time": "11:15",
        "reason": "Asthma control",
    },
    {
        "id": "104",
        "name": "Luca Rossi",
        "age": 67,
        "sex": "Male",
        "tags": ["CHF", "CKD", "ACEi"],
        "time": "14:00",
        "reason": "Heart failure check",
    },
]

# --------- Session State Initialization ---------
if "page" not in st.session_state:
    st.session_state.page = "Clinical Dashboard"

if "selected_patient_id" not in st.session_state:
    st.session_state.selected_patient_id = MOCK_PATIENTS[0]["id"]  # default first


def get_active_patient():
    pid = st.session_state.get("selected_patient_id")
    for p in MOCK_PATIENTS:
        if p["id"] == pid:
            return p
    return None


# --------- Sidebar Layout ---------
# --------- Sidebar Layout ---------
with st.sidebar:
    st.markdown("### MD-GPT")
    st.caption("Clinical Assistant")

    # ---- NAVIGATION (decoupled from session_state key) ----
    pages = ("Clinical Dashboard", "Appointment Summarization",  "Clinical History", "Exams Comparison (OCR)")

    if "page" not in st.session_state:
        st.session_state["page"] = pages[0]

    nav_choice = st.radio(
        "Navigation",
        pages,
        index=pages.index(st.session_state["page"])
    )
    st.session_state["page"] = nav_choice

    st.markdown("---")
    # Patient selector LIVES in the sidebar
    st.markdown("**Active Patient**")

    patient_options = {p["id"]: f"{p['name']} (ID: {p['id']})" for p in MOCK_PATIENTS}
    st.selectbox(
        "Choose patient",
        options=list(patient_options.keys()),
        format_func=lambda pid: patient_options[pid],
        key="selected_patient_id",
    )

    # Display details for the active patient
    active_patient = get_active_patient()
    if active_patient:
        st.markdown(f"{active_patient['name']}  •  ID: {active_patient['id']}")
        st.caption(f"Age: {active_patient['age']}  •  Sex: {active_patient['sex']}")
        st.caption("Key Tags: " + ", ".join(active_patient["tags"]))


# --------- Main Content Routing ---------
page = st.session_state["page"]

if page == "Clinical Dashboard":
    show_dashboard(MOCK_PATIENTS)
elif page == "Appointment Summarization":
    show_appointment_page(MOCK_PATIENTS)
elif page == "Clinical History":
    show_clinical_history_page(MOCK_PATIENTS)
elif page == "Exams Comparison (OCR)":
    show_exams_comparison(MOCK_PATIENTS)
