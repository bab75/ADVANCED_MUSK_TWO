import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import plotly.graph_objects as go

def train_model(model_name, X_train, y_train, X_test, y_test):
    """
    Train a specified machine learning model and return the model and metrics.
    
    Args:
        model_name (str): Name of the model to train.
        X_train (array): Training features.
        y_train (array): Training target.
        X_test (array): Test features.
        y_test (array): Test target.
    
    Returns:
        tuple: (trained model, metrics dictionary)
    """
    model_dict = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Neural Network": MLPClassifier(max_iter=1000, random_state=42)
    }
    
    if model_name not in model_dict:
        raise ValueError(f"Model {model_name} not supported.")
    
    model = model_dict[model_name]
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label="CA"),
        "recall": recall_score(y_test, y_pred, pos_label="CA"),
        "f1": f1_score(y_test, y_pred, pos_label="CA"),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        "y_pred": y_pred
    }
    
    return model, metrics

def tune_model(model_name, X_train, y_train, X_test, y_test, param_grid):
    """
    Perform hyperparameter tuning for a model using GridSearchCV.
    
    Args:
        model_name (str): Name of the model to tune.
        X_train (array): Training features.
        y_train (array): Training target.
        X_test (array): Test features.
        y_test (array): Test target.
        param_grid (dict): Hyperparameter grid for tuning.
    
    Returns:
        tuple: (best model, metrics dictionary, best parameters)
    """
    model_dict = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Neural Network": MLPClassifier(max_iter=1000, random_state=42)
    }
    
    if model_name not in model_dict:
        raise ValueError(f"Model {model_name} not supported.")
    
    model = model_dict[model_name]
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label="CA"),
        "recall": recall_score(y_test, y_pred, pos_label="CA"),
        "f1": f1_score(y_test, y_pred, pos_label="CA"),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        "y_pred": y_pred
    }
    
    return best_model, metrics, grid_search.best_params_

def get_model_explanation(model_name, X_sample, model):
    """
    Generate an explanation for a model's prediction on a sample.
    
    Args:
        model_name (str): Name of the model.
        X_sample (array): Sample input for explanation.
        model: Trained model object.
    
    Returns:
        str: Explanation of the model's prediction.
    """
    if model_name in ["Random Forest", "Decision Tree", "Gradient Boosting"]:
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
            top_feature_idx = np.argmax(importance)
            return f"The most important feature for {model_name} is feature {top_feature_idx} with importance {importance[top_feature_idx]:.4f}."
    elif model_name == "Logistic Regression":
        if hasattr(model, "coef_"):
            coef = model.coef_[0]
            top_feature_idx = np.argmax(np.abs(coef))
            return f"The feature with the highest coefficient in {model_name} is feature {top_feature_idx} with coefficient {coef[top_feature_idx]:.4f}."
    elif model_name in ["SVM", "Neural Network"]:
        return f"{model_name} is a complex model; feature importance is not directly available."
    return f"No explanation available for {model_name}."

def plot_confusion_matrix(y_true, y_pred):
    """
    Create a Plotly confusion matrix plot.
    
    Args:
        y_true (array): True labels.
        y_pred (array): Predicted labels.
    
    Returns:
        go.Figure: Plotly figure object.
    """
    cm = confusion_matrix(y_true, y_pred, labels=["NO-CA", "CA"])
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["NO-CA", "CA"],
        y=["NO-CA", "CA"],
        colorscale="Blues",
        showscale=True
    ))
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual"
    )
    return fig

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model object.
        feature_names (list): List of feature names.
    
    Returns:
        go.Figure: Plotly figure object or None if not applicable.
    """
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        sorted_idx = np.argsort(importance)[::-1]
        top_features = feature_names[:10] if len(feature_names) > 10 else feature_names
        top_importance = importance[sorted_idx][:10] if len(feature_names) > 10 else importance[sorted_idx]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[feature_names[i] for i in sorted_idx][:10],
            y=top_importance,
            marker_color="#3498db"
        ))
        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Feature",
            yaxis_title="Importance",
            xaxis_tickangle=45
        )
        return fig
    return None

def recommend_models(target_series):
    """
    Recommend models based on target variable characteristics.
    
    Args:
        target_series (pd.Series): Target variable data.
    
    Returns:
        dict: Dictionary of model recommendations with reasons.
    """
    class_balance = target_series.value_counts(normalize=True)
    is_imbalanced = any(class_balance < 0.3)
    
    recommendations = {
        "Logistic Regression": "Suitable for binary classification and interpretable results.",
        "Random Forest": "Robust to overfitting and handles complex patterns well."
    }
    
    if is_imbalanced:
        recommendations["Gradient Boosting"] = "Effective for imbalanced datasets with high predictive accuracy."
    else:
        recommendations["Decision Tree"] = "Simple and interpretable for balanced datasets."
    
    return recommendations
