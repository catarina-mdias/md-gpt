# views/dashboard.py
import streamlit as st
import pandas as pd


def _card_container():
    return st.container(border=True)


def show_dashboard(patients_today):
    st.title("MD-GPT Clinical Dashboard")
    st.caption("AI-Powered Clinical Assistant")

    # ---------- Top Cards ----------
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        with _card_container():
            st.subheader("⚠️ Critical Action Items (All Patients)")
            st.markdown(
                """
                - 2 prescriptions flagged for interaction review
                - 1 abnormal lab result not yet acknowledged
                - 3 follow-up appointments overdue
                """
            )
            st.button("Review All Warnings", use_container_width=True)
    with col2:
        with _card_container():
            st.subheader("📈 Productivity Score")
            st.markdown("### 45 min")
            st.caption("Validation time saved today")
            st.markdown("⬆️ **15% WoW**")
            st.markdown("**Avg. Accuracy:** 98.1%")
    with col3:
        with _card_container():
            st.subheader("⚡ Quick Start")

            if st.button("Go to Exams Comparison", use_container_width=True):
                st.session_state["page"] = "Exams Comparison (OCR)"
                st.rerun()

            if st.button("Start New Appointment", use_container_width=True):
                st.session_state["page"] = "Appointment Summarization"
                st.rerun()

            if st.button("Generate Patient Summary", use_container_width=True):
                st.session_state["page"] = "Clinical History"
                st.rerun()

    st.markdown("")
    # ---------- Today’s Schedule ----------
    st.markdown("### Today’s Schedule")
    schedule_df = pd.DataFrame(
        [
            {
                "Time": p["time"],
                "Patient": p["name"],
                "ID": p["id"],
                "Reason": p["reason"],
            }
            for p in patients_today
        ]
    )
    st.dataframe(schedule_df, hide_index=True, use_container_width=True)

    st.markdown("### Select Patient for Appointment")

    # ---- Dropdown for patient selection by NAME ----
    patient_names = [p["name"] for p in patients_today]

    # Determine current name from selected_patient_id, if any
    current_id = st.session_state.get("selected_patient_id", patients_today[0]["id"])
    current_patient = next(
        (p for p in patients_today if p["id"] == current_id), patients_today[0]
    )
    current_name = current_patient["name"]

    selected_name = st.selectbox(
        "Choose patient:",
        options=patient_names,
        index=patient_names.index(current_name),
        key="dashboard_patient_select_by_name",
    )

    # ---- Select button sets global state (selected_patient_id) + goes to Appointment Summarization ----
    if st.button("Start Appointment with Selected Patient", key="start_appt_from_dashboard"):
        # cross the info: find patient by name, then store its ID
        chosen_patient = next(
            (p for p in patients_today if p["name"] == selected_name), None
        )
        if chosen_patient:
            st.session_state["selected_patient_id"] = chosen_patient["id"]
        st.session_state["page"] = "Appointment Summarization"
        st.rerun()

    # ---------- Summary Metrics ----------
    st.markdown("### Snapshot: Today’s Workload")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Patients Scheduled", len(patients_today))
    with c2:
        st.metric("Summaries Generated", 6, "+25%")
    with c3:
        st.metric("Warnings Raised", 5, "+1")
    with c4:
        st.metric("Avg. Review Time", "1.8 min", "-0.4")

    st.markdown("---")
