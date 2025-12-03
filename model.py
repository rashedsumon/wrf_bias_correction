# model.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import joblib

MODEL_PATH = "bias_correction_model.pkl"

def train_rf_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    return model

def load_model():
    import os
    if os.path.exists(MODEL_PATH):
        import joblib
        return joblib.load(MODEL_PATH)
    return None

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    corr = np.corrcoef(y_test, preds)[0, 1]
    return {"RMSE": rmse, "MAE": mae, "Correlation": corr}
