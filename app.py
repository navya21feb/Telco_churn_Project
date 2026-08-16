import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Loading the saved pipeline (preprocessing + model together)

model = joblib.load('churn_model_pipeline.pkl')

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn risk.")

# Input form — mirror the raw columns your pipeline expects

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

with col2:
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=840.0)

# Build a single-row dataframe matching the training data's raw format

input_df = pd.DataFrame([{
    'gender': gender,
    'SeniorCitizen': senior_citizen,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
}])

# NOTE: applying the SAME manual preprocessing steps I did in Phase 3
# before the ColumnTransformer step (binary mapping, ordinal mapping,
# tenure bucketing) — since those happened outside the pipeline object.
# If moved everything inside the pipeline/ColumnTransformer instead,
# skip this block entirely and pass input_df directly to model.predict().

binary_map = {
    'gender': {'Male': 1, 'Female': 0},
    'Partner': {'Yes': 1, 'No': 0},
    'Dependents': {'Yes': 1, 'No': 0},
    'PhoneService': {'Yes': 1, 'No': 0},
    'PaperlessBilling': {'Yes': 1, 'No': 0},
}
for col, mapping in binary_map.items():
    input_df[col] = input_df[col].map(mapping)

contract_order = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
input_df['Contract'] = input_df['Contract'].map(contract_order)

# Predict and display

if st.button("Predict Churn Risk"):
    CHOSEN_THRESHOLD = 0.4  

    proba = model.predict_proba(input_df)[0, 1]
    prediction = "Likely to Churn" if proba >= CHOSEN_THRESHOLD else "Not Likely to Churn"

    st.subheader(f"Prediction: {prediction}")
    st.metric("Churn Probability", f"{proba:.1%}")

    if proba >= CHOSEN_THRESHOLD:
        st.warning("This customer shows a high risk of churning.")
    else:
        st.success("This customer shows a low risk of churning.")

# ---- SHAP explanation ----
st.subheader("Why this prediction?")

try:
    fitted_preprocessor = model.named_steps['preprocessor']
    fitted_classifier = model.named_steps['classifier']

    # Transform input exactly as the model expects
    input_transformed = fitted_preprocessor.transform(input_df)

    # Convert sparse matrix to dense
    if hasattr(input_transformed, "toarray"):
        input_transformed = input_transformed.toarray()

    # Get feature names after encoding
    feature_names = fitted_preprocessor.get_feature_names_out()

    # Create SHAP explainer
    explainer = shap.TreeExplainer(fitted_classifier)

    # Calculate SHAP values
    shap_values = explainer.shap_values(input_transformed)

    # Take SHAP values for our single customer
    if isinstance(shap_values, list):
        customer_shap = shap_values[1][0]
    else:
        customer_shap = shap_values[0]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    shap.bar_plot(
        customer_shap,
        feature_names=feature_names,
        max_display=10,
        show=False
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

except Exception as e:
    st.error(f"SHAP explanation failed: {e}")

st.divider()
st.caption("Model: tuned pipeline trained on the Telco Customer Churn dataset.")