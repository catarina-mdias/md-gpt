# views/appointment.py
import os
import requests
import streamlit as st
import pandas as pd  # kept in case you add tables later
from datetime import date

# We now use the unified API base from app.py (st.session_state.api_base)
DEFAULT_DOCTOR_NAME = os.getenv("DEFAULT_DOCTOR_NAME", "Dr. MD-GPT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TRANSCRIPTION_MODEL = os.getenv("OPENAI_TRANSCRIPTION_MODEL", "whisper-1")
TRANSCRIPTION_ENDPOINT = os.getenv(
    "OPENAI_TRANSCRIPTION_URL", "https://api.openai.com/v1/audio/transcriptions"
)


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
            "transcript": None,
            "summary_fields": None,
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


def _transcribe_audio_file(uploaded_file) -> str:
    """
    Send the uploaded audio file to the OpenAI transcription endpoint
    and return the transcript text.
    """
    if not uploaded_file:
        st.error("Please upload an audio file before transcribing.")
        return ""

    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY is not configured. Cannot transcribe audio.")
        return ""

    audio_bytes = uploaded_file.getvalue()
    files = {
        "file": (
            uploaded_file.name,
            audio_bytes,
            uploaded_file.type or "audio/mpeg",
        )
    }
    data = {"model": TRANSCRIPTION_MODEL}
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    try:
        resp = requests.post(
            TRANSCRIPTION_ENDPOINT,
            headers=headers,
            data=data,
            files=files,
            timeout=90,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        details = ""
        if getattr(exc, "response", None) is not None:
            try:
                details = exc.response.text
            except Exception:
                details = ""
        error_msg = f"Transcription failed: {exc}"
        if details:
            error_msg = f"{error_msg}\n{details}"
        st.error(error_msg)
        st.info(
            "Ensure your OpenAI account has access to the selected model or override "
            "OPENAI_TRANSCRIPTION_MODEL (default 'whisper-1')."
        )
        return ""

    payload = resp.json()
    transcript_text = (payload.get("text") or "").strip()
    if not transcript_text:
        st.warning("Transcription completed but returned empty text.")
    return transcript_text


def _call_appointment_summary_api(
    patient_id: str,
    patient_name: str,
    input_mode: str,  # "transcript" | "notes"
    content: str,
    reason_for_visit: str | None = None,
    appointment_id: str | None = None,
) -> dict | None:
    """
    Call the unified FastAPI /appointment-summary endpoint and return the parsed JSON,
    or None on failure.

    The backend will:
    - generate a brief structured summary
    - fill symptoms/diagnosis/therapeutics/follow_up fields
    - persist them into clinical_data.json
    """
    token = st.session_state.get("api_token")
    if not token:
        st.error(
            "No API token available. The MD-GPT API login is done at the app entry page; "
            "please log out and sign in again if this persists."
        )
        return None

    base_url = st.session_state.get("api_base", "http://localhost:8000")

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

    # Attach metadata used for persistence so the UI can show/edit it later.
    data.setdefault("appointment_date", today_str)
    data.setdefault("appointment_doctor", doctor_name)
    data.setdefault("reason_for_visit", reason)

    return data


# ---------- Main Streamlit view ----------


def show_appointment_page(patients_today):
    active_patient = _get_active_patient(patients_today)
    _patient_header(active_patient)

    if not active_patient:
        return

    state = _get_patient_appt_state(active_patient["id"])

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
                state["transcript"] = None

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
                state["transcript"] = None

        # If in progress, show appropriate UI
        if state["in_progress"]:
            if state["mode"] == "recording":
                # --- Recording mode UI with audio upload/transcription ---
                with st.container(border=True):
                    st.subheader("Recording Mode")
                    st.markdown(
                        "Upload the consultation audio file once recording concludes. "
                        "The file will be transcribed using your configured speech-to-text model "
                        "before generating the appointment summary."
                    )

                    audio_key = f"audio_upload_{active_patient['id']}"
                    audio_file = st.file_uploader(
                        "Audio file (.mp3, .wav, .m4a, .mp4)",
                        type=["mp3", "wav", "m4a", "mp4"],
                        key=audio_key,
                    )

                    if audio_file:
                        st.caption(f"Loaded file: **{audio_file.name}** ({audio_file.size / 1024:.1f} KB)")

                    process_disabled = audio_file is None
                    if st.button(
                        "Transcribe Audio & Generate Summary",
                        use_container_width=True,
                        disabled=process_disabled,
                        key=f"process_audio_{active_patient['id']}",
                    ):
                        if not audio_file:
                            st.error("Please upload an audio file before processing.")
                        else:
                            with st.spinner("Transcribing audio..."):
                                transcript_text = _transcribe_audio_file(audio_file)

                            if transcript_text:
                                state["transcript"] = transcript_text
                                state["in_progress"] = False
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
                                    state["summary_fields"] = {
                                        "appointment_id": api_data.get("appointment_id"),
                                        "patient_id": api_data.get("patient_id", active_patient["id"]),
                                        "appointment_date": api_data.get("appointment_date", today_str := date.today().isoformat()),
                                        "appointment_doctor": api_data.get("appointment_doctor", DEFAULT_DOCTOR_NAME),
                                        "reason_for_visit": api_data.get("reason_for_visit") or active_patient.get("reason") or "",
                                        "appointment_symptoms": api_data.get("symptoms", ""),
                                        "diagnosis": api_data.get("diagnosis", ""),
                                        "therapeutics": api_data.get("therapeutics", ""),
                                        "follow_up": api_data.get("follow_up", ""),
                                        "appointment_summary": api_data.get("summary", ""),
                                    }
                                    st.success("Audio processed and summary generated.")
                                else:
                                    state["summary"] = _mock_summarize_notes(
                                        transcript_text,
                                        active_patient["name"],
                                    )
                                    state["summary_fields"] = {
                                        "appointment_id": state.get("appointment_id"),
                                        "patient_id": active_patient["id"],
                                        "appointment_date": date.today().isoformat(),
                                        "appointment_doctor": DEFAULT_DOCTOR_NAME,
                                        "reason_for_visit": active_patient.get("reason") or "",
                                        "appointment_symptoms": "",
                                        "diagnosis": "",
                                        "therapeutics": "",
                                        "follow_up": "",
                                        "appointment_summary": transcript_text,
                                    }
                                    st.warning("Appointment API unavailable; showing fallback summary.")
                            else:
                                st.error("Unable to transcribe the provided audio file.")

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
                        state["transcript"] = None

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
                            state["summary_fields"] = {
                                "appointment_id": api_data.get("appointment_id"),
                                "patient_id": api_data.get("patient_id", active_patient["id"]),
                                "appointment_date": api_data.get("appointment_date", date.today().isoformat()),
                                "appointment_doctor": api_data.get("appointment_doctor", DEFAULT_DOCTOR_NAME),
                                "reason_for_visit": api_data.get("reason_for_visit") or active_patient.get("reason") or "",
                                "appointment_symptoms": api_data.get("symptoms", ""),
                                "diagnosis": api_data.get("diagnosis", ""),
                                "therapeutics": api_data.get("therapeutics", ""),
                                "follow_up": api_data.get("follow_up", ""),
                                "appointment_summary": api_data.get("summary", ""),
                            }
                        else:
                            # Fallback to mock summary if API fails
                            state["summary"] = _mock_summarize_notes(
                                notes_text,
                                active_patient["name"],
                            )
                            state["summary_fields"] = {
                                "appointment_id": state.get("appointment_id"),
                                "patient_id": active_patient["id"],
                                "appointment_date": date.today().isoformat(),
                                "appointment_doctor": DEFAULT_DOCTOR_NAME,
                                "reason_for_visit": active_patient.get("reason") or "",
                                "appointment_symptoms": "",
                                "diagnosis": "",
                                "therapeutics": "",
                                "follow_up": "",
                                "appointment_summary": notes_text,
                            }

    # ------------------ STEP 2: REVIEW STRUCTURED SUMMARY ------------------
    st.markdown("### Step 2: Review Structured Summary")

    if state["summary"] is None:
        st.info("Once you end the appointment, a structured summary will appear here.")
        return

    summary_fields = state.get("summary_fields") or {}
    if not summary_fields:
        summary_fields = {
            "appointment_id": state.get("appointment_id"),
            "patient_id": active_patient["id"],
            "appointment_date": date.today().isoformat(),
            "appointment_doctor": DEFAULT_DOCTOR_NAME,
            "reason_for_visit": active_patient.get("reason") or "",
            "appointment_symptoms": "",
            "diagnosis": "",
            "therapeutics": "",
            "follow_up": "",
            "appointment_summary": state["summary"] or "",
        }
        state["summary_fields"] = summary_fields

    editable = state["editing_summary"]

    with st.container(border=True):
        st.subheader("Appointment Metadata")
        col_meta_a, col_meta_b = st.columns(2)
        appointment_id_input = col_meta_a.text_input(
            "Appointment ID",
            value=summary_fields.get("appointment_id") or "",
            key=f"appointment_id_{active_patient['id']}",
            disabled=True,
        )
        patient_id_display = col_meta_b.text_input(
            "Patient ID",
            value=summary_fields.get("patient_id") or active_patient["id"],
            key=f"patient_id_{active_patient['id']}",
            disabled=True,
        )
        col_meta_c, col_meta_d = st.columns(2)
        appointment_date_input = col_meta_c.text_input(
            "Appointment Date",
            value=summary_fields.get("appointment_date") or date.today().isoformat(),
            key=f"appointment_date_{active_patient['id']}",
            disabled=True,
        )
        appointment_doctor_input = col_meta_d.text_input(
            "Doctor",
            value=summary_fields.get("appointment_doctor") or DEFAULT_DOCTOR_NAME,
            key=f"appointment_doctor_{active_patient['id']}",
            disabled=True,
        )
        reason_input = st.text_input(
            "Reason for Visit",
            value=summary_fields.get("reason_for_visit") or "",
            key=f"reason_{active_patient['id']}",
            disabled=not editable,
        )

    with st.container(border=True):
        st.subheader("Clinical Summary Fields")
        appointment_summary_input = st.text_area(
            "Appointment Summary",
            value=summary_fields.get("appointment_summary") or state["summary"] or "",
            height=220,
            key=f"summary_text_{active_patient['id']}",
            disabled=not editable,
        )
        symptoms_input = st.text_area(
            "Symptoms",
            value=summary_fields.get("appointment_symptoms") or "",
            key=f"symptoms_{active_patient['id']}",
            disabled=not editable,
        )
        diagnosis_input = st.text_area(
            "Diagnosis",
            value=summary_fields.get("diagnosis") or "",
            key=f"diagnosis_{active_patient['id']}",
            disabled=not editable,
        )
        therapeutics_input = st.text_area(
            "Therapeutics",
            value=summary_fields.get("therapeutics") or "",
            key=f"therapeutics_{active_patient['id']}",
            disabled=not editable,
        )
        follow_up_input = st.text_area(
            "Follow-up Plan",
            value=summary_fields.get("follow_up") or "",
            key=f"follow_up_{active_patient['id']}",
            disabled=not editable,
        )

    if state.get("transcript"):
        with st.expander("View transcript from uploaded audio"):
            st.text_area(
                "Audio transcript",
                value=state["transcript"],
                height=200,
                key=f"transcript_{active_patient['id']}",
                disabled=True,
            )

    updated_fields = {
        "appointment_id": appointment_id_input,
        "patient_id": patient_id_display,
        "appointment_date": appointment_date_input,
        "appointment_doctor": appointment_doctor_input,
        "reason_for_visit": reason_input,
        "appointment_symptoms": symptoms_input,
        "diagnosis": diagnosis_input,
        "therapeutics": therapeutics_input,
        "follow_up": follow_up_input,
        "appointment_summary": appointment_summary_input,
    }

    col_a, col_b = st.columns(2)
    with col_a:
        if not editable:
            if st.button("Edit Summary", use_container_width=True):
                state["editing_summary"] = True
        else:
            if st.button("Save Summary", use_container_width=True):
                state["summary_fields"] = updated_fields
                state["summary"] = appointment_summary_input
                state["appointment_id"] = appointment_id_input or state.get("appointment_id")
                state["editing_summary"] = False
                st.success("Summary updated.")

    with col_b:
        if st.button("Validate Summary", use_container_width=True):
            state["summary_fields"] = updated_fields
            state["summary"] = appointment_summary_input
            state["appointment_id"] = appointment_id_input or state.get("appointment_id")
            state["editing_summary"] = False
            st.success("Summary validated and saved for this appointment.")
