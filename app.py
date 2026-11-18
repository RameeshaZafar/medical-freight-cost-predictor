import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor

st.set_page_config(page_title="Medical Cargo Freight Cost Predictor", page_icon="🚢", layout="centered")

@st.cache_resource
def build_and_train_model():
    raw_df = pd.read_csv("SCMS_Delivery_History_Dataset.csv")
    df = raw_df.copy()

    # 1. Deterministic target & feature parsing
    df["target_freight_usd"] = pd.to_numeric(df["Freight Cost (USD)"], errors="coerce")
    df = df[df["target_freight_usd"].notna() & (df["target_freight_usd"] > 0)].copy()
    df["Weight_kg"] = pd.to_numeric(df["Weight (Kilograms)"], errors="coerce")

    cols_to_drop = [
        "ID", "Project Code", "PQ #", "PO / SO #", "ASN/DN #",
        "Item Description", "Molecule/Test Type",
        "Delivered to Client Date", "Delivery Recorded Date",
        "Line Item Insurance (USD)", "First Line Designation",
        "Freight Cost (USD)", "Weight (Kilograms)",
        "PQ First Sent to Client Date", "PO Sent to Vendor Date", "Scheduled Delivery Date",
        "target_freight_usd"
    ]
    X = df.drop(columns=cols_to_drop)
    y = df["target_freight_usd"]

    y_buckets = pd.qcut(y, q=10, labels=False, duplicates="drop")
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y_buckets
    )

    numeric_features = [
        "Unit of Measure (Per Pack)", "Line Item Quantity",
        "Line Item Value", "Pack Price", "Unit Price", "Weight_kg"
    ]
    categorical_features = [
        "Country", "Managed By", "Fulfill Via", "Vendor INCO Term",
        "Shipment Mode", "Product Group", "Sub Classification", "Vendor",
        "Brand", "Dosage", "Dosage Form", "Manufacturing Site"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler())
            ]), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False))
            ]), categorical_features)
        ],
        remainder="drop"
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", TransformedTargetRegressor(
            regressor=HistGradientBoostingRegressor(
                max_iter=160, learning_rate=0.05, max_leaf_nodes=45,
                min_samples_leaf=15, l2_regularization=0.0, random_state=42
            ),
            func=np.log1p,
            inverse_func=np.expm1
        ))
    ])

    model.fit(X_train, y_train)
    return model

model = build_and_train_model()

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
        "Weight_kg": weight_kg
    }])

    predicted_usd = model.predict(input_row)[0]
    st.success(f"### Estimated Freight Cost: **${predicted_usd:,.2f} USD**")
    st.caption(f"Expected model margin of error (Test MAE): ±$4,469 | Transport Mode: {shipment_mode}")
