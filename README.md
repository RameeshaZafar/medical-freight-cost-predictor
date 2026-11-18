\# Medical Supply Chain Freight Cost Predictor



An end-to-end, leak-free machine learning pipeline and interactive Streamlit web app that forecasts pre-dispatch international freight costs (USD) for global healthcare shipments.



\## Business Impact

\* \*\*49.6% Error Reduction:\*\* Cut per-shipment test MAE from \*\*$8,876.17\*\* (median baseline) down to \*\*$4,469.23\*\* across 1,240 unseen shipments.

\* \*\*Budget Variance Halved:\*\* Reduced overall freight spend variance (WAPE) from \*\*78.3%\*\* to \*\*39.4%\*\*.

\* \*\*Actionable Cost Screening:\*\* Flags heavy air-freight orders prior to dispatch so logistics teams can consolidate eligible cargo into Ocean/Truck transport.



\## Engineering Highlights

\* \*\*ERP Data Cleaning:\*\* Resolved 4,126 mixed-string target records (`"See DN-..."` multi-line invoice cross-references and bundled commodity contracts) to isolate 6,198 standalone freight bills.

\* \*\*Strict Zero-Leakage Architecture:\*\* Removed post-shipment fields (`Delivered to Client Date`, `Line Item Insurance`), locked a decile-stratified 80/20 split prior to learning, and encapsulated median imputation, missing-weight indicators, rare-category one-hot encoding, and log-target scaling (`TransformedTargetRegressor`) inside a `scikit-learn` `Pipeline`.



\## Key Visuals

!\[Feature Importance](feature\_importance.png)

!\[Predicted vs Actual](predicted\_vs\_actual.png)



\## Run Locally

```bash

pip install -r requirements.txt

streamlit run app.py

