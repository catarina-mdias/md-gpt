# views/exams_comparison.py
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


def show_exams_comparison(patients_today):
    active_patient = _get_active_patient(patients_today)

    _patient_header(active_patient)

    st.markdown("## Exams Comparison (OCR)")
    st.caption(
        "Upload exam reports (PDF / images) for instant interpretation, "
        "comparison, and timeline insights."
    )

    st.markdown("### Upload Exam Report")
    uploaded_file = st.file_uploader(
        "Click to upload exam report",
        type=["pdf", "png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )

    col_left, col_right = st.columns([2, 2])

    # ---------- Left: Current Exam ----------
    with col_left:
        st.subheader("Current Exam – Key Findings")

        if uploaded_file is None:
            st.info(
                "No file uploaded yet. Upload a report to see automatic "
                "extraction of key findings."
            )
        else:
            st.success(f"File received: **{uploaded_file.name}**")

            name_for_text = active_patient["name"] if active_patient else "the patient"

            st.markdown(
                f"""
                **Mock interpretation (to be replaced by MD-GPT):**
                - Elevated LDL cholesterol compared to previous exam  
                - Blood pressure stable under current treatment  
                - Mild increase in fasting glucose (watch for pre-diabetes)  

                These findings are attached to **{name_for_text}**.
                """
            )
            st.text_area(
                "Doctor’s Notes",
                "Add your notes here…",
                height=150,
            )
            st.button("Save Notes to Patient Record", use_container_width=True)

    # ---------- Right: Timeline & Comparison ----------
    with col_right:
        st.subheader("Timeline Comparison")

        # Mock timeline summary table
        mock_data = pd.DataFrame(
            {
                "Date": ["2023-09-12", "2024-01-20", "2024-06-02"],
                "Exam": ["Blood Test", "MRI Abdomen", "Blood Test"],
                "Key Finding": [
                    "LDL slightly elevated",
                    "No relevant structural changes",
                    "LDL higher, glucose borderline",
                ],
                "Trend vs Previous": [
                    "Baseline",
                    "Stable",
                    "Worsening lipids",
                ],
            }
        )

        st.dataframe(
            mock_data,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Draft MD-GPT Summary (Mocked)")
        st.markdown(
            """
            - Lipid profile shows **progressive LDL increase** over the last 9 months.  
            - No major structural abnormalities on the last MRI.  
            - Consider intensifying lifestyle measures and evaluating need for statin adjustment.  
            """
        )
        st.button("Mark Summary as Reviewed", use_container_width=True)

    st.markdown("---")
    st.caption(
        "All interpretations shown here are placeholders. "
        "Once your agent and OCR pipeline are ready, call them where the "
        "mocked blocks are."
    )
