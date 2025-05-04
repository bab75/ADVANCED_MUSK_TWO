import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from model_utils import calculate_drop_off

def load_uploaded_data(file):
    """
    Load and validate uploaded CSV data.
    """
    try:
        data = pd.read_csv(file)
        required_columns = ['Student_ID', 'School', 'Grade', 'Gender', 'Meal_Code', 
                          'Academic_Performance', 'Transportation', 'Suspensions', 
                          'Present_Days', 'Absent_Days', 'Attendance_Percentage']
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
        return data
    except Exception as e:
        raise ValueError(f"Error loading CSV: {str(e)}")

def compute_high_risk_baselines(data):
    """
    Compute baseline thresholds for high-risk students based on historical data.
    """
    try:
        baselines = {}
        for col in ['Attendance_Percentage', 'Academic_Performance', 'Suspensions']:
            if col in data.columns:
                baselines[col] = {
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'high_risk_threshold': data[col].mean() - data[col].std()
                }
        return baselines
    except Exception as e:
        raise ValueError(f"Error computing baselines: {str(e)}")

def preprocess_data(data, features, target=None, encoder=None):
    """
    Preprocess data for model training or prediction.
    """
    try:
        # Drop rows with missing values in features or target
        columns_to_check = features + ([target] if target else [])
        data = data.dropna(subset=columns_to_check)
        
        # Separate features and target
        X = data[features]
        y = data[target] if target else None
        
        # Identify categorical and numerical columns
        categorical_cols = [col for col in features if X[col].dtype == "object" or X[col].dtype.name == "category"]
        numerical_cols = [col for col in features if col not in categorical_cols]
        
        # Create preprocessing pipeline
        if not encoder:
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_cols),
                    ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
                ])
            preprocessor.fit(X)
        else:
            preprocessor = encoder
        
        # Transform features
        X_transformed = preprocessor.transform(X)
        
        # Get feature names after encoding
        if categorical_cols:
            cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
            feature_names = numerical_cols + list(cat_features)
        else:
            feature_names = numerical_cols
        
        # Convert to DataFrame
        X_transformed = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)
        
        return X_transformed, y, preprocessor
    except Exception as e:
        raise ValueError(f"Error preprocessing data: {str(e)}")

def combine_datasets(datasets):
    """
    Combine multiple datasets into a single DataFrame.
    """
    try:
        if not datasets:
            raise ValueError("No datasets provided")
        
        # Ensure all datasets have the same columns
        columns = datasets[0].columns
        for i, ds in enumerate(datasets[1:], 1):
            if set(ds.columns) != set(columns):
                raise ValueError(f"Dataset {i} has different columns")
        
        # Concatenate datasets
        combined = pd.concat(datasets, ignore_index=True)
        
        # Drop duplicates based on Student_ID
        combined = combined.drop_duplicates(subset=['Student_ID'], keep='last')
        
        return combined
    except Exception as e:
        raise ValueError(f"Error combining datasets: {str(e)}")
