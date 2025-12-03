# utils.py
import pandas as pd

def match_by_station_date(wrf_df, obs_df):
    """Helper to merge WRF outputs with observations."""
    return pd.merge(wrf_df, obs_df, on=['station_id', 'date'], how='inner')
