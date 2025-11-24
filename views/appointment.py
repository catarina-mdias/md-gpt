# views/appointment.py
import streamlit as st
import pandas as pd


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


def show_appointment_page(patients_today):
    active_patient = _get_active_patient(patients_today)
    _patient_header(active_patient)

    if not active_patient:
        return

    state = _get_patient_appt_state(active_patient["id"])

    st.markdown("## Appointment Summarization")
    st.caption(
        "Aid the patient consultation with live transcription (when consent is given) "
        "and structured summary generation."
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
                        # TODO: replace this mocked summary with agent output
                        state["summary"] = _mock_summarize_notes(
                            "Auto-generated transcript from recording.",
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
                        state["summary"] = _mock_summarize_notes(
                            state["notes"], active_patient["name"]
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
            # Here you would persist the final summary to your backend / EHR
            state["editing_summary"] = False
            state["summary"] = summary_text
            st.success("Summary validated and saved for this appointment.")
