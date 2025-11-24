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
            # st.button("Start New Appointment", use_container_width=True)
            # st.button("Generate Patient Summary", use_container_width=True)

            # ✅ This now correctly switches page
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
        [{
            "Time": p["time"],
            "Patient": p["name"],
            "ID": p["id"],
            "Reason": p["reason"],
        } for p in patients_today]
    )
    st.dataframe(schedule_df, hide_index=True, use_container_width=True)
    st.markdown("### Select Patient for Appointment")
    # ---- Dropdown for patient selection (moved here from sidebar) ----
    patient_options = {p["id"]: f"{p['name']} (ID: {p['id']})" for p in patients_today}
    # Determine currently selected ID (if exists)
    current_id = st.session_state.get("selected_patient_id", list(patient_options.keys())[0])
    selected_patient_id = st.selectbox(
        "Choose patient:",
        options=list(patient_options.keys()),
        format_func=lambda pid: patient_options[pid],
        index=list(patient_options.keys()).index(current_id),
        key="dashboard_patient_select"
    )
    # ---- Select button sets global state + goes to Appointment Summarization ----
    if st.button("Start Appointment with Selected Patient", key="start_appt_from_dashboard"):
        st.session_state["selected_patient_id"] = selected_patient_id
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
    st.caption(
        "This dashboard remains global. Patient selection now happens here "
        "instead of the sidebar."
    )