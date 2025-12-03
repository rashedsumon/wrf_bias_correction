# WRF Bias-Correction System

This project implements an automated bias-correction layer for daily WRF temperature forecasts.

## Features
- Ingests daily WRF outputs
- Pulls METAR and SYNOP observations
- Applies AI-driven bias correction (Random Forest / ML models)
- Computes RMSE, MAE, and correlation for validation
- Streamlit web interface for forecast correction

## Setup
1. Clone repository
2. Install dependencies:
```bash
pip install -r requirements.txt
