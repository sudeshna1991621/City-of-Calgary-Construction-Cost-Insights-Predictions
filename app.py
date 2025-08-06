import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import date
# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Calgary Building Permit Cost Estimator",
    page_icon="🏗️",
    layout="wide"
)

# ---------- CUSTOM CSS ----------
page_header_css = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #f9f9f9; /* Light background */
}

h1, h2, h3 {
    color: #003366 !important;
}

div.stButton > button {
    background-color: #FF4B4B;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: bold;
    border: none;
    cursor: pointer;
}

div.stButton > button:hover {
    background-color: #ff1a1a;
}
</style>
"""
st.markdown(page_header_css, unsafe_allow_html=True)

# ---------- HEADER IMAGE ----------
st.image("calgary.jpg", use_column_width=True)

# ---------- TITLE ----------
st.markdown("<h1 style='text-align: center;'>🏗️ Calgary Building Permit Cost Estimator</h1>", unsafe_allow_html=True)
st.write("Estimate your project cost based on building permit details. Fill in the form below and get predictions instantly.")
# ---------- LOAD MODEL ----------
group_classifier = joblib.load("xgb_model_pipeline.joblib")

# ---------- FORM ----------
with st.container():
    st.markdown("### 🔧 Permit Details Entry")
    col1, col2 = st.columns(2)

    with col1:
        permit_num = st.text_input("Permit Number", placeholder="e.g. BP2013-09623")
        selected_status = st.selectbox("Permit Status", joblib.load("StatusCurrent_Top_options.joblib"))
        permit_type = st.selectbox("Permit Type", joblib.load("PermitType_options.joblib"))
        permit_type_mapped = st.selectbox("Permit Type Mapped", joblib.load("PermitTypeMapped_options.joblib"))
        permit_class_top = st.selectbox("Permit Class Top", joblib.load("PermitClass_Top_options.joblib"))
        permit_class_group = st.selectbox("Permit Class Group", joblib.load("PermitClassGroup_options.joblib"))
        permit_class_mapped = st.selectbox("Permit Class Mapped", joblib.load("PermitClassMapped_options.joblib"))

    with col2:
        work_class = st.selectbox("Work Class", joblib.load("WorkClass_options.joblib"))
        work_class_group = st.selectbox("Work Class Group", joblib.load("WorkClassGroup_options.joblib"))
        work_class_mapped = st.selectbox("Work Class Mapped", joblib.load("WorkClassMapped_options.joblib"))

        applied_date = st.date_input("Applied Date", value=date(2020, 1, 1))
        issued_date = st.date_input("Issued Date", value=date(2020, 2, 1))
        completed_date = st.date_input("Completed Date", value=date(2020, 4, 1))

approval_duration = (issued_date - applied_date).days
completion_duration = (completed_date - issued_date).days
applied_year = applied_date.year
build_duration_ratio = completion_duration / approval_duration if approval_duration != 0 else np.nan

st.markdown("---")

col3, col4 = st.columns(2)

with col3:
    total_sqft = st.slider("Total Square Feet (TotalSqFt)", 0, 7000, 1732, 10)
    housing_units = st.slider("Number of Housing Units", 0, 300, 1, 1)
    sqft_per_unit = total_sqft / housing_units if housing_units != 0 else 0

    all_communities = joblib.load("CommunityName_all.joblib")
    top_communities = joblib.load("CommunityName_Top.joblib")
    selected_community = st.selectbox("Community Name", sorted(all_communities))
    community_top = selected_community if selected_community in top_communities else "Other"

with col4:
    all_contractors = joblib.load("ContractorName_all.joblib")
    top_contractors = joblib.load("ContractorName_Top.joblib")
    selected_contractor = st.selectbox("Contractor Name", sorted(all_contractors))
    contractor_top = selected_contractor if selected_contractor in top_contractors else "Other"

    location_count = st.slider("Number of Locations", 1, 410, 2, 1)
    applicant_name = st.text_input("Applicant Name")
    original_address = st.text_area("Original Address")
    description = st.text_area("Project Description")

# ---------- PREDICTION WITH VALIDATION ----------
if st.button("🔮 Predict Project Cost"):
    missing_fields = []

    if not permit_num.strip():
        missing_fields.append("Permit Number")
    if not selected_status:
        missing_fields.append("Permit Status")
    if not permit_type:
        missing_fields.append("Permit Type")
    if not permit_class_top:
        missing_fields.append("Permit Class Top")
    if not permit_class_group:
        missing_fields.append("Permit Class Group")
    if not work_class:
        missing_fields.append("Work Class")
    if total_sqft == 0:
        missing_fields.append("Total Square Feet")
    if housing_units == 0:
        missing_fields.append("Housing Units")
    if not selected_community:
        missing_fields.append("Community Name")
    if not selected_contractor:
        missing_fields.append("Contractor Name")

    if approval_duration <= 0:
        st.warning("⚠️ Issued Date must be after Applied Date.")
    if completion_duration <= 0:
        st.warning("⚠️ Completed Date must be after Issued Date.")

    if missing_fields:
        st.error(f"❌ Please fill in the following required fields: {', '.join(missing_fields)}")
    elif approval_duration <= 0 or completion_duration <= 0:
        pass  # Don't proceed if dates invalid
    else:
        model_input = pd.DataFrame([{
            'PermitType': permit_type,
            'PermitClass_Top': permit_class_top,
            'PermitClassGroup': permit_class_group,
            'WorkClass': work_class,
            'WorkClassGroup': work_class_group,
            'WorkClassMapped': work_class_mapped,
            'StatusCurrent_Top': selected_status,
            'TotalSqFt': total_sqft,
            'HousingUnits': housing_units,
            'SqFtPerUnit': sqft_per_unit,
            'AppliedYear': applied_year,
            'ApprovalDuration': approval_duration,
            'CompletionDuration': completion_duration,
            'LocationCount': location_count,
            'CommunityName_Top': community_top,
            'ContractorName_Top': contractor_top
        }])

        log_cost = group_classifier.predict(model_input)[0]
        predicted_cost = np.expm1(log_cost)

        st.markdown(
            f"""
            <div style="background-color:rgba(211, 211, 211, 0.7); padding:20px; border-radius:10px;">
            <h2 style="color:#FFA500;">💰 Estimated Project Cost: ${predicted_cost:,.2f}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("📋 Prediction Details"):
            st.json({
                "Permit Number": permit_num,
                "Log Cost (XGB)": round(log_cost, 4),
                "Estimated Cost": round(predicted_cost, 2)
            })
