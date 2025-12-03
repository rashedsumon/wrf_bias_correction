# pipeline.py
import pandas as pd
from data_loader import load_data
from model import train_rf_model, load_model, evaluate_model

def preprocess_data(wrf_df, metar_df, synop_df):
    """
    Merge WRF outputs with observation data to create training dataset.
    Simplified example: match by station_id & date.
    """
    obs_df = pd.concat([metar_df, synop_df])
    df = pd.merge(wrf_df, obs_df, on=['station_id', 'date'], how='inner')
    X = df[['wrf_temperature']]
    y = df['obs_temperature']
    return X, y

def train_pipeline():
    wrf_df, metar_df, synop_df = load_data()
    X, y = preprocess_data(wrf_df, metar_df, synop_df)
    
    model = train_rf_model(X, y)
    
    metrics = evaluate_model(model, X, y)
    print("Training Metrics:", metrics)
    return model

def generate_forecast(wrf_forecast_df):
    """Apply trained bias-correction model to new WRF output"""
    model = load_model()
    if model is None:
        model = train_pipeline()  # Train if model doesn't exist
    
    X_new = wrf_forecast_df[['wrf_temperature']]
    corrected = model.predict(X_new)
    wrf_forecast_df['corrected_temperature'] = corrected
    return wrf_forecast_df
