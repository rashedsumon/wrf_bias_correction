# app.py
import streamlit as st
import pandas as pd
from pipeline import generate_forecast

st.set_page_config(page_title="WRF Bias-Correction", layout="wide")
st.title("🌤 WRF Temperature Bias-Correction System")

st.markdown("""
This application applies a trained AI bias-correction model to daily WRF temperature forecasts.
""")

# Upload today's WRF output
wrf_file = st.file_uploader("Upload today's WRF output CSV", type=['csv'])

if wrf_file is not None:
    df_forecast = pd.read_csv(wrf_file)
    
    st.write("Raw WRF Forecast Sample:")
    st.dataframe(df_forecast.head())
    
    corrected_forecast = generate_forecast(df_forecast)
    
    st.write("Corrected Forecast Sample:")
    st.dataframe(corrected_forecast.head())
    
    csv = corrected_forecast.to_csv(index=False).encode('utf-8')
    st.download_button("Download Corrected Forecast CSV", data=csv, file_name='corrected_forecast.csv', mime='text/csv')
