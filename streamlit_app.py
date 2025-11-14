# streamlit_app.py
import streamlit as st
import requests
import pandas as pd

st.title("Kalpi Capital Portfolio Optimizer")
st.caption("FastAPI Backend + Streamlit UI | Quant Assignment")

API_URL = "http://localhost:8000"  # Change to deployed URL later

with st.sidebar:
    method = st.selectbox("Optimizer", [
        "mvo", "cvar", "risk_parity", "kelly",
        "tracking_error", "info_ratio", "sortino", "omega", "min_dd"
    ])
    tickers = st.text_input("Tickers", "RELIANCE.NS,TCS.NS,INFY.NS")
    start = st.date_input("Start", pd.to_datetime("2021-01-01"))
    end = st.date_input("End", pd.to_datetime("2023-12-31"))
    long_only = st.checkbox("Long Only", True)

if st.button("🚀 Optimize"):
    try:
        payload = {
            "tickers": [t.strip() for t in tickers.split(",")],
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "method": method,
            "long_only": long_only
        }
        with st.spinner("Calling FastAPI backend..."):
            res = requests.post(f"{API_URL}/optimize", json=payload, timeout=30)
            data = res.json()
        
        st.success("✅ Success!")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Weights")
            st.dataframe(pd.Series(data["weights"]).to_frame("Weight"))
        with col2:
            st.subheader("Metrics")
            st.json(data["metrics"])
            
    except Exception as e:
        st.error(f"❌ {e}")