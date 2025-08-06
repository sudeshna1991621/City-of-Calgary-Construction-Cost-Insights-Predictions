import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import date
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline


from xgboost import XGBRegressor    


st.title("🏗️ Calgary Building Permit Cost Estimator")

# --- Load all models once ---
group_classifier = joblib.load("xgb_model_pipeline.joblib")
model_small = joblib.load("model_small.joblib")
model_medium = joblib.load("model_medium.joblib")
model_large = joblib.load("model_large.joblib")

# --- Input Form ---
st.header("🔧 Permit Details Entry")

permit_num = st.text_input("Permit Number", placeholder="e.g. BP2013-09623")
status_options = joblib.load("StatusCurrent_Top_options.joblib")
selected_status = st.selectbox("Permit Status", status_options)

permit_type = st.selectbox("Permit Type", joblib.load("PermitType_options.joblib"))
permit_type_mapped = st.selectbox("Permit Type Mapped", joblib.load("PermitTypeMapped_options.joblib"))
permit_class_top = st.selectbox("Permit Class Top", joblib.load("PermitClass_Top_options.joblib"))
permit_class_group = st.selectbox("Permit Class Group", joblib.load("PermitClassGroup_options.joblib"))
permit_class_mapped = st.selectbox("Permit Class Mapped", joblib.load("PermitClassMapped_options.joblib"))
work_class = st.selectbox("Work Class", joblib.load("WorkClass_options.joblib"))
work_class_group = st.selectbox("Work Class Group", joblib.load("WorkClassGroup_options.joblib"))
work_class_mapped = st.selectbox("Work Class Mapped", joblib.load("WorkClassMapped_options.joblib"))

applied_date = st.date_input("Applied Date", value=date(2020, 1, 1))
issued_date = st.date_input("Issued Date", value=date(2020, 2, 1))
completed_date = st.date_input("Completed Date", value=date(2020, 4, 1))

approval_duration = (issued_date - applied_date).days
completion_duration = (completed_date - issued_date).days
build_duration_ratio = completion_duration / approval_duration if approval_duration != 0 else np.nan
applied_year = applied_date.year

total_sqft = st.slider("Total Square Feet (TotalSqFt)", 0, 7000, 1732, 10)
housing_units = st.slider("Number of Housing Units", 0, 300, 1, 1)
sqft_per_unit = total_sqft / housing_units if housing_units != 0 else 0

all_communities = joblib.load("CommunityName_all.joblib")
top_communities = joblib.load("CommunityName_Top.joblib")
selected_community = st.selectbox("Community Name", sorted(all_communities))
community_top = selected_community if selected_community in top_communities else "Other"

all_contractors = joblib.load("ContractorName_all.joblib")
top_contractors = joblib.load("ContractorName_Top.joblib")
selected_contractor = st.selectbox("Contractor Name", sorted(all_contractors))
contractor_top = selected_contractor if selected_contractor in top_contractors else "Other"

location_count = st.slider("Number of Locations", 1, 410, 2, 1)
applicant_name = st.text_input("Applicant Name")
original_address = st.text_area("Original Address")
description = st.text_area("Project Description")

# --- Prediction Trigger ---
if st.button("🔮 Predict Project Cost"):
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

    # --- Predict group ---
    log_cost = group_classifier.predict(model_input)[0]
    cost_xgb = np.expm1(log_cost)

    if cost_xgb < 14000:
        predicted_group = 'small'
        selected_model = model_small
    elif cost_xgb <= 170000:
        predicted_group = 'medium'
        selected_model = model_medium
    else:
        predicted_group = 'large'
        selected_model = model_large

    # --- Predict cost with group-specific regressor ---
    X_group = selected_model['preprocessor'].transform(model_input)
    log_final_cost = selected_model['model'].predict(X_group)[0]
    predicted_cost = np.expm1(log_final_cost)

    # --- Output Result ---
    st.success(f"✅ **Predicted Cost Group**: `{predicted_group.capitalize()}`")
    st.info(f"💰 **Estimated Project Cost**: ${predicted_cost:,.2f}")

    with st.expander("📋 Full Prediction Details"):
        st.json({
            "Permit Number": permit_num,
            "Predicted Cost Group": predicted_group,
            "Log Cost (Classifier)": round(log_cost, 4),
            "Cost (Classifier)": round(cost_xgb, 2),
            "Final Log Cost (Regressor)": round(log_final_cost, 4),
            "Final Predicted Cost": round(predicted_cost, 2)
        })