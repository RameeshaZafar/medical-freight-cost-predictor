import joblib
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Medical Cargo Freight Cost Predictor", page_icon="🚢", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("freight_cost_model.joblib")

model = load_model()

st.title("🚢 Medical Supply Chain Freight Cost Predictor")
st.markdown(
    "Estimate pre-dispatch international shipping costs (USD) for medical cargo orders "
    "using a leak-free `HistGradientBoosting` pipeline trained on USAID SCMS logistics data."
)

col1, col2 = st.columns(2)

with col1:
    country = st.selectbox("Destination Country", ["Nigeria", "South Africa", "Kenya", "Uganda", "Haiti", "Côte d'Ivoire", "Zambia", "Tanzania", "Mozambique", "Other"])
    shipment_mode = st.selectbox("Shipment Mode", ["Air", "Ocean", "Truck", "Air Charter"])
    weight_kg = st.number_input("Cargo Weight (kg)", min_value=1.0, max_value=200000.0, value=250.0, step=50.0)
    line_item_value = st.number_input("Total Cargo Value (USD)", min_value=10.0, max_value=5000000.0, value=15000.0, step=500.0)

with col2:
    product_group = st.selectbox("Product Group", ["ARV", "HRDT", "ACT", "ANTM", "MRDT"])
    fulfill_via = st.selectbox("Fulfillment Channel", ["Direct Drop", "From RDC"])
    inco_term = st.selectbox("Vendor INCO Term", ["EXW", "FCA", "CIP", "CIF", "DDP", "N/A - From RDC"])
    line_qty = st.number_input("Line Item Quantity (Packs)", min_value=1, max_value=500000, value=500, step=50)

if st.button("Predict Freight Cost", type="primary"):
    # Construct single-row DataFrame matching X_train schema
    input_row = pd.DataFrame([{
        "Country": country,
        "Managed By": "PMO - US",
        "Fulfill Via": fulfill_via,
        "Vendor INCO Term": inco_term,
        "Shipment Mode": shipment_mode,
        "Product Group": product_group,
        "Sub Classification": "Adult" if product_group == "ARV" else "HIV test",
        "Vendor": "SCMS from RDC",
        "Brand": "Generic",
        "Dosage": "Missing",
        "Dosage Form": "Tablet",
        "Unit of Measure (Per Pack)": 30,
        "Line Item Quantity": line_qty,
        "Line Item Value": line_item_value,
        "Pack Price": round(line_item_value / max(line_qty, 1), 2),
        "Unit Price": round((line_item_value / max(line_qty, 1)) / 30, 4),
        "Manufacturing Site": "Missing",
        "Weight_kg": weight_kg,
        "Planned_Lead_Time_Days": np.nan
    }])

    predicted_usd = model.predict(input_row)[0]
    st.success(f"### Estimated Freight Cost: **${predicted_usd:,.2f} USD**")
    st.caption(f"Expected model margin of error (Test MAE): ±$4,469 | Transport Mode: {shipment_mode}")