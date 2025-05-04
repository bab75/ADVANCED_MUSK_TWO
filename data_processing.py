import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import streamlit as st

def load_uploaded_data(file):
    """Load and validate uploaded CSV data."""
    try:
        data = pd.read_csv(file, dtype_backend="pandas", low_memory=False)
        if data.empty:
            raise ValueError("Uploaded CSV is empty.")
        if "Student_ID" in data.columns:
            data["Student_ID"] = data["Student_ID"].astype(str)
        if "School" in data.columns:
            data["School"] = data["School"].astype(str)
        return data
    except Exception as e:
        raise ValueError(f"Error loading CSV: {str(e)}")

def compute_high_risk_baselines(data):
    """Compute statistical baselines for high-risk identification."""
    try:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        baselines = {}
        for col in numeric_cols:
            if col not in ["Student_ID", "School", "Year"]:
                baselines

[col] = {
                    "mean": data[col].mean(),
                    "std": data[col].std()
                }
        return baselines
    except Exception as e:
        st.warning(f"Error computing high-risk baselines: {str(e)}")
        return {}

def preprocess_data(data, features, targets):
    """Preprocess data for training or prediction, handling all data types."""
    try:
        # Ensure data is a DataFrame
        data = pd.DataFrame(data)
        
        # Define columns to always treat as categorical (identifiers or non-features)
        categorical_cols = ["Student_ID", "School"]
        
        # Handle missing values
        for col in data.columns:
            if col in data.select_dtypes(include=[np.number]).columns:
                data[col] = data[col].fillna(data[col].mean())
            else:
                data[col] = data[col].fillna(data[col].mode().iloc[0] if not data[col].mode().empty else "Unknown")
        
        # Initialize X and y
        X = data[features].copy() if features else pd.DataFrame()
        y = pd.DataFrame()
        
        # Ensure School is treated as categorical if included in features
        for col in X.columns:
            if col in categorical_cols or X[col].dtype == "object" or X[col].dtype.name == "category":
                categorical_cols.append(col) if col not in categorical_cols else None
        
        # Remove duplicates in categorical_cols
        categorical_cols = list(set(categorical_cols) & set(X.columns))
        
        # Numerical columns are those not in categorical_cols
        numerical_cols = [col for col in X.columns if col not in categorical_cols]
        
        # Encode categorical features
        label_encoders = {}
        for col in categorical_cols:
            if col in X.columns:
                try:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    label_encoders[col] = le
                except Exception as e:
                    st.warning(f"Error encoding feature {col}: {str(e)}")
                    X[col] = 0
        
        # Scale numerical features only
        scaler = StandardScaler()
        if numerical_cols:
            try:
                X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
            except Exception as e:
                st.warning(f"Error scaling numerical features: {str(e)}")
                X[numerical_cols] = 0
        
        # Process targets
        if targets:
            y = data[targets].copy()
            for col in y.columns:
                if y[col].dtype == "object" or y[col].dtype.name == "category" or y[col].dtype == "bool":
                    try:
                        if y[col].isna().any():
                            raise ValueError(f"Target column {col} contains missing values.")
                        unique_values = y[col].dropna().unique()
                        if len(unique_values) < 2:
                            raise ValueError(f"Target column {col} has fewer than 2 unique values.")
                        le = LabelEncoder()
                        y[col] = le.fit_transform(y[col].astype(str))
                    except Exception as e:
                        st.warning(f"Error encoding target {col}: {str(e)}")
                        y[col] = 0
                else:
                    # Ensure target is binary or categorical
                    unique_values = y[col].dropna().unique()
                    if len(unique_values) > 2:
                        st.warning(f"Target column {col} has more than 2 unique values, treating as numeric may cause errors.")
        
        return X, y
    except Exception as e:
        raise ValueError(f"Error preprocessing data: {str(e)}")

def combine_datasets(datasets):
    """Combine multiple datasets into a single DataFrame."""
    try:
        if not datasets:
            return pd.DataFrame()
        combined = pd.concat(datasets, ignore_index=True)
        if "Student_ID" in combined.columns:
            combined["Student_ID"] = combined["Student_ID"].astype(str)
        if "School" in combined.columns:
            combined["School"] = combined["School"].astype(str)
        return combined
    except Exception as e:
        raise ValueError(f"Error combining datasets: {str(e)}")
