import streamlit as st
import pandas as pd


def _get_active_patient(patients_today):
    pid = st.session_state.get("selected_patient_id")
    for p in patients_today:
        if p["id"] == pid:
            return p
    return None


def _patient_header(active_patient):
    with st.container(border=True):
        if not active_patient:
            st.markdown("### No active patient")
            st.caption("Select a patient from the sidebar to view history.")
            return

        st.markdown(f"### Active Patient: {active_patient['name']}")
        st.caption(
            f"Age: {active_patient['age']}  •  Sex: {active_patient['sex']}  •  "
            "Key Tags: " + ", ".join(active_patient["tags"])
        )


def show_clinical_history_page(patients_today):
    active_patient = _get_active_patient(patients_today)
    _patient_header(active_patient)

    if not active_patient:
        return

    st.markdown("## Patient Clinical History Summarization")
    st.caption(
        "Instantly summarize a patient's entire history based on chosen topics "
        "and level of detail."
    )

    col_left, col_right = st.columns([2, 2])

    # ---------- Topic selection ----------
    with col_left:
        with st.container(border=True):
            st.subheader("Select Topics to Show")

            # You can tune these defaults later
            prev_appts = st.checkbox("Previous Appointments", value=False)
            lab_results = st.checkbox("Lab Results", value=False)
            surgical_history = st.checkbox("Surgical History", value=False)
            medication_changes = st.checkbox("Medication Changes", value=True)
            key_diagnosis = st.checkbox("Key Diagnosis Timeline", value=True)

    # ---------- Detail level ----------
    with col_right:
        with st.container(border=True):
            st.subheader("Level of Detail")

            detail_level = st.radio(
                "",
                options=[
                    "Key Takeaways Only",
                    "Medium Detail",
                    "Comprehensive",
                ],
                index=1,  # default: Medium Detail
            )

    st.markdown("")

    # ---------- Generate Summary button ----------
    if st.button("Generate Summary (MD-GPT Summarizes)", use_container_width=True):
        # For now, build a mocked “LLM summary” based on selections
        selected_topics = []
        if prev_appts:
            selected_topics.append("previous appointments")
        if lab_results:
            selected_topics.append("lab results")
        if surgical_history:
            selected_topics.append("surgical history")
        if medication_changes:
            selected_topics.append("medication changes")
        if key_diagnosis:
            selected_topics.append("key diagnosis timeline")

        topics_text = ", ".join(selected_topics) if selected_topics else "no topics selected"

        summary_text = f"""
Structured history summary for **{active_patient['name']}**  
Detail level: **{detail_level}**  
Included topics: **{topics_text}**

> This is a mocked summary. Here you will later call your LLM with the
> patient's full record and the selected options.

Key Points:
- Long-standing hypertension with gradual medication adjustments.
- No recent hospitalizations; outpatient follow-ups stable.
- Latest labs within acceptable range, with mild dyslipidemia.
- Current treatment is well tolerated; no recent allergic reactions.
"""

        st.session_state[f"history_summary_{active_patient['id']}"] = summary_text

    # ---------- Show summary if available ----------
    summary_key = f"history_summary_{active_patient['id']}"
    if summary_key in st.session_state:
        st.markdown("---")
        st.markdown("### Generated History Summary (Mock)")

        st.markdown(
            st.session_state[summary_key],
            unsafe_allow_html=True,  # just to allow the bold/markdown we built
        )
