# data_loader.py
import os
import pandas as pd
import kagglehub

DATA_DIR = "data"

def download_data():
    """Download historical WRF outputs and observation data from KaggleHub."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    datasets = {
        "wrf_outputs": "user/wrf-historical-data",  # Replace with KaggleHub dataset
        "metar_obs": "user/metar-data",
        "synop_obs": "user/synop-data"
    }

    paths = {}
    for name, ds in datasets.items():
        path = kagglehub.dataset_download(ds, path=DATA_DIR)
        paths[name] = path
        print(f"{name} downloaded to: {path}")
    return paths

def load_data():
    """Load all datasets into Pandas DataFrames"""
    paths = download_data()
    wrf_df = pd.read_csv(paths['wrf_outputs'])
    metar_df = pd.read_csv(paths['metar_obs'])
    synop_df = pd.read_csv(paths['synop_obs'])
    return wrf_df, metar_df, synop_df
