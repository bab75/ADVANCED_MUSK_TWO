import pandas as pd
import numpy as np
import uuid
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import plotly.graph_objects as go
from model_utils import train_model, tune_model, get_model_explanation
from data_processing import preprocess_data

def train_and_tune_model(data, features, targets, models_to_train, enable_tuning, tuning_params):
    """Train and tune models for multi-target learning."""
    X = data[features]
    y = data[targets]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    categorical_cols = [col for col in features if data[col].dtype == "object"]
    numerical_cols = [col for col in features if col not in categorical_cols]
    
    X_train_processed, preprocessor, feature_names = preprocess_data(X_train, categorical_cols, numerical_cols)
    X_test_processed = preprocessor.transform(X_test)
    
    models = {}
    model_versions = {}
    patterns = []
    
    for model_name in models_to_train:
        try:
            if enable_tuning and model_name in tuning_params:
                model, metrics, best_params = tune_model(
                    model_name, X_train_processed, y_train, X_test_processed, y_test, tuning_params[model_name]
                )
            else:
                model, metrics = train_model(model_name, X_train_processed, y_train, X_test_processed, y_test)
                best_params = None
            
            version_id = str(uuid.uuid4())
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            model_versions.setdefault(model_name, []).append({
                "version_id": version_id,
                "timestamp": timestamp,
                "metrics": metrics,
                "best_params": best_params,
                "model": model,
                "preprocessor": preprocessor,
                "feature_names": feature_names,
                "y_test": y_test,
                "y_pred": metrics["y_pred"],
                "X_test_processed": X_test_processed
            })
            
            models[model_name] = {
                "model": model,
                "metrics": metrics,
                "preprocessor": preprocessor,
                "feature_names": feature_names,
                "best_params": best_params,
                "y_test": y_test,
                "y_pred": metrics["y_pred"],
                "X_test_processed": X_test_processed
            }
            
            high_risk = data[data["CA_Status"] == "CA"]
            if not high_risk.empty:
                patterns.extend([
                    {"pattern": f"Average Attendance: {high_risk['Attendance_Percentage'].mean():.2f}%", "explanation": "Identified in high-risk students"},
                    {"pattern": f"Common Grades: {', '.join(map(str, high_risk['Grade'].mode().tolist()))}", "explanation": "Identified in high-risk students"}
                ])
        except Exception as e:
            raise ValueError(f"Error training {model_name}: {str(e)}")
    
    return {"models": models, "model_versions": model_versions}, patterns

def run_predictions(data, features, selected_model, models, drop_off_rules, patterns, high_risk_baselines):
    """Run predictions on current-year data."""
    model_info = models[selected_model]
    X = data[features]
    X_processed = model_info["preprocessor"].transform(X)
    predictions = model_info["model"].predict(X_processed)
    probabilities = model_info["model"].predict_proba(X_processed)[:, 1]
    
    prediction_data = data.copy()
    prediction_data["CA_Prediction"] = predictions[:, 0] if predictions.shape[1] > 1 else predictions
    prediction_data["CA_Probability"] = probabilities
    
    def apply_prediction_drop_off_rules(row):
        if row["CA_Prediction"] != "CA":
            return "N"
        attendance = row["Attendance_Percentage"]
        if not (drop_off_rules["attendance_min"] <= attendance <= drop_off_rules["attendance_max"]):
            return "N"
        for feature, values in drop_off_rules.get("features", {}).items():
            if feature in row and row[feature] not in values:
                return "N"
        return "Y"
    
    prediction_data["Drop_Off"] = prediction_data.apply(apply_prediction_drop_off_rules, axis=1)
    
    def identify_causes(row):
        causes = []
        if row["CA_Prediction"] == "CA":
            for pattern in patterns:
                pattern_text = pattern["pattern"].lower()
                if "attendance" in pattern_text and row["Attendance_Percentage"] < (high_risk_baselines["Attendance_Percentage"] if high_risk_baselines else 80):
                    causes.append(pattern["pattern"])
        return ", ".join(causes) if causes else "None"
    
    prediction_data["Prediction_Causes"] = prediction_data.apply(identify_causes, axis=1)
    return prediction_data

def plot_confusion_matrix(y_test, y_pred):
    """Plot confusion matrix."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=["Non-CA", "CA"], y=["Non-CA", "CA"],
        colorscale="Blues", showscale=True
    ))
    fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
    return fig

def plot_feature_importance(model, feature_names):
    """Plot feature importance."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        fig = go.Figure(go.Bar(
            x=importances, y=feature_names, orientation="h"
        ))
        fig.update_layout(title="Feature Importance", xaxis_title="Importance", yaxis_title="Feature")
        return fig
    return None