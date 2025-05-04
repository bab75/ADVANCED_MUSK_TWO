import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from data_generator import generate_historical_data, generate_current_year_data

def load_uploaded_data(uploaded_file):
    """Load and validate uploaded CSV data."""
    try:
        data = pd.read_csv(uploaded_file)
        required_columns = ["Student_ID", "CA_Status", "Drop_Off"]
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        return data
    except Exception as e:
        raise ValueError(f"Error loading CSV: {str(e)}")

def compute_high_risk_baselines(data):
    """Compute baseline statistics for high-risk students."""
    if data is not None:
        high_risk = data[data["CA_Status"] == "CA"]
        if not high_risk.empty:
            return {
                "Attendance_Percentage": high_risk["Attendance_Percentage"].mean(),
                "Academic_Performance": high_risk["Academic_Performance"].mean(),
                "Suspensions": high_risk["Suspensions"].mean(),
                "Transportation": high_risk["Transportation"].mode().iloc[0] if not high_risk["Transportation"].empty else "Unknown"
            }
    return None

def preprocess_data(X, categorical_cols, numerical_cols):
    """Preprocess data with scaling and encoding."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_cols)
        ])
    X_processed = preprocessor.fit_transform(X)
    feature_names = numerical_cols + list(preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols))
    return X_processed, preprocessor, feature_names

def combine_datasets(datasets):
    """Combine multiple datasets, ensuring column consistency."""
    if not datasets:
        raise ValueError("No datasets provided.")
    combined = pd.concat(datasets, ignore_index=True)
    required_columns = ["Student_ID", "CA_Status", "Drop_Off"]
    missing_cols = [col for col in required_columns if col not in combined.columns]
    if missing_cols:
        raise ValueError(f"Combined dataset missing required columns: {missing_cols}")
    return combined