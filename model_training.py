import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from model_utils import identify_patterns, explain_prediction
import plotly.graph_objects as go

def train_and_tune_model(data, features, targets, models_to_train, enable_tuning, tuning_params):
    """
    Train and tune machine learning models on the provided data.
    
    Args:
        data (pd.DataFrame): Input data
        features (list): List of feature column names
        targets (list): List of target column names
        models_to_train (list): List of model names to train
        enable_tuning (bool): Whether to perform hyperparameter tuning
        tuning_params (dict): Hyperparameter grids for tuning
    
    Returns:
        dict: Trained models and their metrics
        list: Identified patterns
    """
    X = data[features]
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "SVM": SVC(probability=True),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Neural Network": MLPClassifier(max_iter=1000)
    }
    
    results = {"models": {}, "model_versions": {}}
    patterns = []
    
    for target in targets:
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        for model_name in models_to_train:
            model = models[model_name]
            if enable_tuning and model_name in tuning_params:
                grid_search = GridSearchCV(
                    model, tuning_params[model_name], cv=5, scoring='f1', n_jobs=-1
                )
                grid_search.fit(X_train_scaled, y_train)
                model = grid_search.best_estimator_
                best_params = grid_search.best_params_
            else:
                model.fit(X_train_scaled, y_train)
                best_params = {}
            
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
            
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average='binary', pos_label='CA'),
                "recall": recall_score(y_test, y_pred, average='binary', pos_label='CA'),
                "f1": f1_score(y_test, y_pred, average='binary', pos_label='CA'),
                "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0
            }
            
            results["models"][model_name] = {
                "model": model,
                "feature_names": features,
                "metrics": {target: metrics},
                "y_test": {target: y_test},
                "y_pred": {target: y_pred},
                "best_params": best_params
            }
            
            model_version_id = str(uuid.uuid4())
            results["model_versions"][model_version_id] = {
                "model_name": model_name,
                "target": target,
                "features": features,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat()
            }
            
            if model_name in ["Random Forest", "Decision Tree", "Gradient Boosting"]:
                patterns.extend(identify_patterns(X_train, y_train, features, model))
    
    return results, patterns

def run_predictions(data, features, model_name, models, drop_off_rules, patterns, baselines):
    """
    Run predictions on new data using a trained model.
    
    Args:
        data (pd.DataFrame): Input data for predictions
        features (list): List of feature column names
        model_name (str): Name of the model to use
        models (dict): Dictionary of trained models
        drop_off_rules (dict): Rules for drop-off identification
        patterns (list): Identified patterns
        baselines (dict): High-risk baselines
    
    Returns:
        pd.DataFrame: Data with predictions and explanations
    """
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found.")
    
    model_info = models[model_name]
    model = model_info["model"]
    trained_features = model_info["feature_names"]
    
    # Validate features
    missing_features = [f for f in trained_features if f not in data.columns]
    extra_features = [f for f in features if f not in trained_features]
    
    if missing_features or extra_features:
        error_msg = ""
        if missing_features:
            error_msg += f"Missing features in data: {missing_features}. "
        if extra_features:
            error_msg += f"Extra features not in training: {extra_features}. "
        error_msg += "Ensure prediction features match training features."
        raise ValueError(error_msg)
    
    output = data.copy()
    
    # Add Drop_Off column if missing
    if "Drop_Off" not in output.columns:
        output["Drop_Off"] = False
    
    # Preprocess data
    X = output[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Make predictions
    output["CA_Prediction"] = model.predict(X_scaled)
    output["CA_Probability"] = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else 0.0
    
    # Apply drop-off rules
    if drop_off_rules:
        attendance_min = drop_off_rules.get("attendance_min", 0)
        attendance_max = drop_off_rules.get("attendance_max", 100)
        rule_features = drop_off_rules.get("features", {})
        
        drop_off_condition = (
            (output["Attendance_Percentage"] >= attendance_min) &
            (output["Attendance_Percentage"] <= attendance_max)
        )
        
        for feature, values in rule_features.items():
            if feature in output.columns:
                drop_off_condition &= output[feature].isin(values)
        
        output.loc[drop_off_condition, "Drop_Off"] = True
    
    # Generate prediction explanations
    output["Prediction_Causes"] = output.apply(
        lambda row: explain_prediction(row, features, patterns, baselines),
        axis=1
    )
    
    return output

def plot_confusion_matrix(y_true, y_pred):
    """
    Plot a confusion matrix for model evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        plotly.graph_objects.Figure: Confusion matrix plot
    """
    cm = confusion_matrix(y_true, y_pred, labels=['Non-CA', 'CA'])
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Non-CA', 'CA'],
        y=['Non-CA', 'CA'],
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 20}
    ))
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        width=400,
        height=400
    )
    return fig

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model
        feature_names (list): List of feature names
    
    Returns:
        plotly.graph_objects.Figure: Feature importance plot or None
    """
    if not hasattr(model, "feature_importances_"):
        return None
    
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({"Feature": feature_names, "Importance": importance})
    feature_importance = feature_importance.sort_values("Importance", ascending=True)
    
    fig = go.Figure(go.Bar(
        x=feature_importance["Importance"],
        y=feature_importance["Feature"],
        orientation='h'
    ))
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=400
    )
    return fig

def get_model_explanation(model_name, model_info):
    """
    Generate a text explanation of the model's performance.
    
    Args:
        model_name (str): Name of the model
        model_info (dict): Model information including metrics
    
    Returns:
        str: Explanation text
    """
    metrics = model_info["metrics"]
    explanation = f"Model: {model_name}\n"
    for target, metric in metrics.items():
        explanation += (
            f"Target: {target}\n"
            f"Accuracy: {metric['accuracy']:.2f}\n"
            f"Precision: {metric['precision']:.2f}\n"
            f"Recall: {metric['recall']:.2f}\n"
            f"F1 Score: {metric['f1']:.2f}\n"
            f"ROC AUC: {metric['roc_auc']:.2f}\n"
        )
    if model_info["best_params"]:
        explanation += "Best Parameters:\n" + "\n".join([f"{k}: {v}" for k, v in model_info["best_params"].items()])
    return explanation
