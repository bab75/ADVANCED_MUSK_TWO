import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from model_utils import train_model, tune_model, get_model_explanation
from data_processing import preprocess_data
import plotly.express as px
import plotly.figure_factory as ff
import uuid

def train_and_tune_model(data, features, targets, models_to_train, enable_tuning, tuning_params):
    """Train and tune models, return results and patterns."""
    try:
        X, y = preprocess_data(data, features, targets)
        if X.empty or y.empty:
            raise ValueError("Preprocessed data is empty. Check features and targets.")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model_results = {"models": {}, "model_versions": {}}
        patterns = []
        
        for model_name in models_to_train:
            try:
                model_info = {
                    "model": None,
                    "metrics": {},
                    "y_test": {},
                    "y_pred": {},
                    "feature_names": X.columns.tolist(),
                    "best_params": {}
                }
                
                if enable_tuning and model_name in tuning_params and tuning_params[model_name]:
                    model, metrics, best_params = tune_model(
                        model_name, X_train, y_train, X_test, y_test, tuning_params[model_name]
                    )
                    model_info["best_params"] = best_params
                else:
                    model, metrics = train_model(model_name, X_train, y_train, X_test, y_test)
                
                model_info["model"] = model
                model_info["metrics"] = metrics
                
                # Safely populate y_test and y_pred
                for target in y_train.columns:
                    try:
                        model_info["y_test"][target] = y_test[target].values
                        model_info["y_pred"][target] = metrics[target]["y_pred"] if "y_pred" in metrics[target] else []
                    except Exception as e:
                        st.warning(f"Error storing predictions for target {target} in {model_name}: {str(e)}")
                        model_info["y_pred"][target] = []
                
                model_version = str(uuid.uuid4())
                model_results["models"][model_name] = model_info
                model_results["model_versions"][model_name] = model_version
                
                explanation = get_model_explanation(model_name, X_test, model)
                patterns.append({
                    "pattern": f"{model_name} Explanation",
                    "explanation": explanation
                })
                
                # Discover patterns based on feature importance
                if model_name in ["Random Forest", "Decision Tree", "Gradient Boosting"]:
                    try:
                        importance = model.estimators_[0].feature_importances_
                        feature_importance = sorted(zip(X.columns, importance), key=lambda x: x[1], reverse=True)
                        top_features = [f"{feat}: {imp:.4f}" for feat, imp in feature_importance[:3]]
                        patterns.append({
                            "pattern": f"Top features for {model_name}",
                            "explanation": f"Most important features: {', '.join(top_features)}"
                        })
                    except Exception as e:
                        st.warning(f"Error computing feature importance for {model_name}: {str(e)}")
            
            except Exception as e:
                st.warning(f"Error in model training pipeline for {model_name}: {str(e)}")
                continue
        
        return model_results, patterns
    except Exception as e:
        raise ValueError(f"Error in model training pipeline: {str(e)}")

def run_predictions(data, features, selected_model, models, drop_off_rules, patterns, high_risk_baselines):
    """Run predictions on new data."""
    try:
        X, _ = preprocess_data(data, features, [])
        if X.empty:
            raise ValueError("Preprocessed features are empty.")
        
        model_info = models.get(selected_model)
        if not model_info:
            raise ValueError(f"Model {selected_model} not found.")
        
        model = model_info["model"]
        predictions = model.predict(X)
        probabilities = model.predict_proba(X) if hasattr(model, "predict_proba") else [np.zeros((X.shape[0], 2)) for _ in range(predictions.shape[1])]
        
        prediction_data = data.copy()
        for i, target in enumerate(model_info["metrics"].keys()):
            prediction_data[f"{target}_Prediction"] = predictions[:, i]
            prediction_data[f"{target}_Probability"] = probabilities[i][:, 1]
        
        prediction_data["Drop_Off"] = False
        prediction_data["Prediction_Causes"] = ""
        
        if drop_off_rules and "features" in drop_off_rules:
            for idx, row in prediction_data.iterrows():
                attendance = row.get("Attendance_Percentage", 100)
                if drop_off_rules["attendance_min"] <= attendance <= drop_off_rules["attendance_max"]:
                    for feature, values in drop_off_rules["features"].items():
                        if feature in row and row[feature] in values:
                            prediction_data.at[idx, "Drop_Off"] = True
                            prediction_data.at[idx, "Prediction_Causes"] += f"{feature}: {row[feature]}, "
        
        for idx, row in prediction_data.iterrows():
            causes = []
            for feature in features:
                if high_risk_baselines and feature in high_risk_baselines:
                    if row[feature] > high_risk_baselines[feature]["mean"] + high_risk_baselines[feature]["std"]:
                        causes.append(f"High {feature}")
                    elif row[feature] < high_risk_baselines[feature]["mean"] - high_risk_baselines[feature]["std"]:
                        causes.append(f"Low {feature}")
            prediction_data.at[idx, "Prediction_Causes"] += "; ".join(causes)
        
        return prediction_data
    except Exception as e:
        raise ValueError(f"Error running predictions: {str(e)}")

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix using Plotly."""
    try:
        cm = np.zeros((2, 2))
        for i in range(len(y_true)):
            cm[int(y_true[i]), int(y_pred[i])] += 1
        
        fig = ff.create_annotated_heatmap(
            z=cm,
            x=["Predicted 0", "Predicted 1"],
            y=["Actual 0", "Actual 1"],
            colorscale="Blues",
            showscale=True
        )
        fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
        return fig
    except Exception as e:
        st.warning(f"Error plotting confusion matrix: {str(e)}")
        return None

def plot_feature_importance(model, feature_names):
    """Plot feature importance for tree-based models."""
    try:
        importance = model.estimators_[0].feature_importances_
        df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
        df = df.sort_values("Importance", ascending=True)
        
        fig = px.bar(
            df,
            x="Importance",
            y="Feature",
            title="Feature Importance",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues"
        )
        return fig
    except Exception as e:
        st.warning(f"Error plotting feature importance: {str(e)}")
        return None
