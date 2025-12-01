# views/appointment.py
import os
import requests
import streamlit as st
import pandas as pd  # kept in case you add tables later
from datetime import date

# Base URL for the appointment summarization API
APPOINTMENT_API_BASE = os.getenv("APPOINTMENT_API_URL", "http://localhost:10001")
DEFAULT_DOCTOR_NAME = os.getenv("DEFAULT_DOCTOR_NAME", "Dr. MD-GPT")


def _get_active_patient(patients_today):
    pid = st.session_state.get("selected_patient_id")
    for p in patients_today:
        if p["id"] == pid:
            return p
    return None


def _get_patient_appt_state(patient_id: str):
    """
    Per-patient appointment state stored in session_state.

    consent: 'yes' | 'no' | None
    mode: 'recording' | 'notes' | None
    in_progress: bool
    notes: str
    summary: str | None
    editing_summary: bool
    appointment_id: str | None
    """
    key = f"appt_state_{patient_id}"
    if key not in st.session_state:
        st.session_state[key] = {
            "consent": None,
            "mode": None,
            "in_progress": False,
            "notes": "",
            "summary": None,
            "editing_summary": False,
            "appointment_id": None,
        }
    return st.session_state[key]


def _mock_summarize_notes(notes: str, patient_name: str) -> str:
    notes = (notes or "").strip()
    if not notes:
        return f"Appointment with {patient_name}. No detailed notes entered yet."
    return f"Structured summary for {patient_name}:\n\n{notes}"


def _patient_header(active_patient):
    with st.container(border=True):
        if not active_patient:
            st.markdown("### No active patient")
            st.caption("Select a patient from the sidebar to start an appointment.")
            return

        st.markdown(f"### Active Patient: {active_patient['name']}")
        st.caption(
            f"Age: {active_patient['age']}  •  Sex: {active_patient['sex']}  •  "
            "Key Tags: " + ", ".join(active_patient["tags"])
        )


# ---------- API helpers for appointment summarization ----------


def _login_appointment_api():
    """
    Login to the Appointment Summarization API and store its token in session_state.

    Uses AGENT_API_USERNAME / AGENT_API_PASSWORD (same as other agents),
    but this FastAPI app has its own token store, so we keep a separate token.
    """
    # If we already have a token, don't log in again
    if "appointment_api_token" in st.session_state and st.session_state.appointment_api_token:
        st.success("Logged in to Appointment API")
        return

    # Allow overriding base URL via session_state if you ever set it from app.py
    base_url = st.session_state.get("appointment_api_base", APPOINTMENT_API_BASE)

    with st.expander("Login to Appointment API", expanded=True):
        username = st.text_input("API Username (appointment agent)", key="appt_api_username")
        password = st.text_input("API Password (appointment agent)", type="password", key="appt_api_password")
        if st.button("Login to Appointment API", key="login_appt_api_button"):
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
                    st.session_state["appointment_api_token"] = token
                    st.success("Login to Appointment API successful. Token stored.")
                else:
                    st.error(f"Login failed: {resp.status_code} - {resp.text}")
            except Exception as e:
                st.error(f"Error calling Appointment API /login: {e}")


def _call_appointment_summary_api(
    patient_id: str,
    patient_name: str,
    input_mode: str,  # "transcript" | "notes"
    content: str,
    reason_for_visit: str | None = None,
    appointment_id: str | None = None,
) -> dict | None:
    """
    Call the FastAPI /appointment-summary endpoint and return the parsed JSON,
    or None on failure.

    The backend will:
    - generate a brief structured summary
    - fill symptoms/diagnosis/therapeutics/follow_up fields
    - persist them into clinical_data.json
    """
    token = st.session_state.get("appointment_api_token")
    if not token:
        st.error("No Appointment API token. Please log in to the Appointment API first.")
        return None

    base_url = st.session_state.get("appointment_api_base", APPOINTMENT_API_BASE)

    # Basic metadata – you can refine this later to use real schedule/doctor data
    today_str = date.today().isoformat()
    reason = reason_for_visit or f"Consultation with {patient_name}"
    doctor_name = DEFAULT_DOCTOR_NAME

    payload = {
        "patient_id": patient_id,
        "input_mode": input_mode,          # "transcript" | "notes"
        "content": content,
        "appointment_id": appointment_id,  # may be None for new appointment
        "appointment_date": today_str,
        "appointment_doctor": doctor_name,
        "reason_for_visit": reason,
        "session_id": f"appointment-{patient_id}-{today_str}",
    }

    try:
        resp = requests.post(
            f"{base_url}/appointment-summary",
            json=payload,
            headers={"x_auth_token": token},
            timeout=40,
        )
    except Exception as e:
        st.error(f"Error calling /appointment-summary endpoint: {e}")
        return None

    if resp.status_code != 200:
        st.error(f"Appointment API error {resp.status_code}: {resp.text}")
        return None

    try:
        data = resp.json()
    except Exception as e:
        st.error(f"Error parsing Appointment API response JSON: {e}")
        return None

    return data


# ---------- Main Streamlit view ----------


def show_appointment_page(patients_today):
    active_patient = _get_active_patient(patients_today)
    _patient_header(active_patient)

    if not active_patient:
        return

    state = _get_patient_appt_state(active_patient["id"])

    # ---------- API Login ----------
    _login_appointment_api()

    st.markdown("## Appointment Summarization")
    st.caption(
        "Aid the patient consultation with live transcription (when consent is given) "
        "and structured summary generation using the MD-GPT appointment agent."
    )

    # ------------------ STEP 1: START & CONSENT ------------------
    st.markdown("### Step 1: Start & Consent")

    col_left, col_right = st.columns([2, 2])

    # --- Right column: consent card ---
    with col_right:
        with st.container(border=True):
            st.subheader("Patient Consent")

            # default selection based on stored consent
            if state["consent"] == "yes":
                default_index = 0
            elif state["consent"] == "no":
                default_index = 1
            else:
                default_index = 1  # default to "No" until explicitly given

            consent_choice = st.radio(
                "Has the patient given consent for audio recording for this visit?",
                options=[
                    "Yes - Recording Mode",
                    "No - Use Note-Taking Mode",
                ],
                index=default_index,
            )

            new_consent = "yes" if consent_choice.startswith("Yes") else "no"
            if new_consent != state["consent"]:
                state["consent"] = new_consent
                state["mode"] = "recording" if new_consent == "yes" else "notes"
                state["in_progress"] = False
                state["summary"] = None
                state["notes"] = ""
                state["appointment_id"] = None

            if state["consent"] == "yes":
                st.success("Patient consent recorded: audio recording is allowed.")
            else:
                st.info("No audio recording consent. Use note-taking mode for this visit.")

    # --- Left column: main action (start button / controls) ---
    with col_left:
        # Button label depends on mode
        if state["mode"] == "recording":
            main_label = "Start Recording"
        elif state["mode"] == "notes":
            main_label = "Start Appointment in Note-Taking Mode"
        else:
            main_label = "Start New Appointment"

        # If appointment not started yet
        if not state["in_progress"] and state["summary"] is None:
            if st.button(main_label, use_container_width=True):
                state["in_progress"] = True
                state["summary"] = None

        # If in progress, show appropriate UI
        if state["in_progress"]:
            if state["mode"] == "recording":
                # --- Recording mode UI (placeholder for real mic component) ---
                with st.container(border=True):
                    st.subheader("Recording in progress")
                    st.markdown(
                        "🎙️ Audio recording has **started**. "
                        "Here you can plug in your actual microphone component."
                    )
                    st.caption(
                        "For now this is a placeholder; integrate your own audio "
                        "capture and streaming to MD-GPT here."
                    )

                    if st.button("End Appointment / Stop Recording", use_container_width=True):
                        state["in_progress"] = False

                        # Placeholder transcript for now
                        transcript_text = "Auto-generated transcript from recording."

                        with st.spinner("Generating appointment summary from transcript..."):
                            api_data = _call_appointment_summary_api(
                                patient_id=active_patient["id"],
                                patient_name=active_patient["name"],
                                input_mode="transcript",
                                content=transcript_text,
                                reason_for_visit=active_patient.get("reason"),
                                appointment_id=state.get("appointment_id"),
                            )

                        if api_data:
                            state["summary"] = api_data.get("summary", "")
                            state["appointment_id"] = api_data.get("appointment_id")
                        else:
                            # Fallback to mock summary if API fails
                            state["summary"] = _mock_summarize_notes(
                                transcript_text,
                                active_patient["name"],
                            )

            else:
                # --- Note-taking mode UI ---
                with st.container(border=True):
                    st.subheader("Note-Taking Mode (no recording)")
                    notes_key = f"notes_{active_patient['id']}"
                    state["notes"] = st.text_area(
                        "Type your notes during the appointment:",
                        value=state["notes"],
                        height=200,
                        key=notes_key,
                    )

                    if st.button("End Appointment (Generate Summary)", use_container_width=True):
                        state["in_progress"] = False
                        notes_text = state["notes"]

                        if not notes_text.strip():
                            st.warning("No notes entered. A very minimal summary will be generated.")
                        with st.spinner("Generating appointment summary from notes..."):
                            api_data = _call_appointment_summary_api(
                                patient_id=active_patient["id"],
                                patient_name=active_patient["name"],
                                input_mode="notes",
                                content=notes_text,
                                reason_for_visit=active_patient.get("reason"),
                                appointment_id=state.get("appointment_id"),
                            )

                        if api_data:
                            state["summary"] = api_data.get("summary", "")
                            state["appointment_id"] = api_data.get("appointment_id")
                        else:
                            # Fallback to mock summary if API fails
                            state["summary"] = _mock_summarize_notes(
                                notes_text,
                                active_patient["name"],
                            )

    # ------------------ STEP 2: REVIEW STRUCTURED SUMMARY ------------------
    st.markdown("### Step 2: Review Structured Summary")

    if state["summary"] is None:
        st.info("Once you end the appointment, a structured summary will appear here.")
        return

    # Editable summary area
    summary_key = f"summary_text_{active_patient['id']}"
    summary_text = st.text_area(
        "Structured Summary",
        value=state["summary"],
        height=260,
        key=summary_key,
        disabled=not state["editing_summary"],
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if not state["editing_summary"]:
            if st.button("Edit Summary", use_container_width=True):
                state["editing_summary"] = True
        else:
            if st.button("Save Summary", use_container_width=True):
                state["summary"] = summary_text
                state["editing_summary"] = False

    with col_b:
        if st.button("Validate Summary", use_container_width=True):
            # Here you would persist the final edited summary to your backend / EHR again
            state["editing_summary"] = False
            state["summary"] = summary_text
            st.success("Summary validated and saved for this appointment.")
