from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
import plotly.graph_objects as go
from sklearn.multioutput import MultiOutputClassifier
import pandas as pd

def train_model(model_name, X_train, y_train, X_test, y_test, preprocessor):
    """
    Train a specified model and compute metrics.
    
    Args:
        model_name (str): Name of the model to train.
        X_train: Training features.
        y_train: Training target.
        X_test: Test features.
        y_test: Test target.
        preprocessor: Preprocessor for numerical and categorical features.
    
    Returns:
        Pipeline: Trained pipeline.
        dict: Metrics for each target.
    """
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_name == "SVM":
        model = SVC(probability=True, random_state=42)
    elif model_name == "Gradient Boosting":
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif model_name == "Neural Network":
        model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Handle multi-target
    if isinstance(y_train, pd.DataFrame) and y_train.shape[1] > 1:
        model = MultiOutputClassifier(model)
    
    # Create pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("estimator", model)
    ])
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    metrics = {}
    
    if isinstance(y_test, pd.DataFrame):
        for i, target in enumerate(y_test.columns):
            metrics[target] = {
                "accuracy": accuracy_score(y_test.iloc[:, i], y_pred[:, i] if y_pred.ndim > 1 else y_pred),
                "precision": precision_score(y_test.iloc[:, i], y_pred[:, i] if y_pred.ndim > 1 else y_pred, pos_label="CA" if target == "CA_Status" else "Y", zero_division=0),
                "recall": recall_score(y_test.iloc[:, i], y_pred[:, i] if y_pred.ndim > 1 else y_pred, pos_label="CA" if target == "CA_Status" else "Y", zero_division=0),
                "f1": f1_score(y_test.iloc[:, i], y_pred[:, i] if y_pred.ndim > 1 else y_pred, pos_label="CA" if target == "CA_Status" else "Y", zero_division=0),
            }
    else:
        metrics["target"] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, pos_label="CA", zero_division=0),
            "recall": recall_score(y_test, y_pred, pos_label="CA", zero_division=0),
            "f1": f1_score(y_test, y_pred, pos_label="CA", zero_division=0),
        }
    
    return pipeline, metrics

def tune_model(model_name, X_train, y_train, X_test, y_test, param_grid, preprocessor):
    """
    Tune a model using GridSearchCV with a preprocessor.
    
    Args:
        model_name (str): Name of the model to train.
        X_train: Training features.
        y_train: Training target.
        X_test: Test features.
        y_test: Test target.
        param_grid (dict): Hyperparameter grid for GridSearchCV.
        preprocessor: Preprocessor for numerical and categorical features.
    
    Returns:
        Pipeline: Trained pipeline with the best estimator.
        dict: Metrics for each target.
    """
    # Get the base estimator without fitting
    if model_name == "Logistic Regression":
        base_estimator = LogisticRegression(max_iter=1000)
    elif model_name == "Random Forest":
        base_estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "Decision Tree":
        base_estimator = DecisionTreeClassifier(random_state=42)
    elif model_name == "SVM":
        base_estimator = SVC(probability=True, random_state=42)
    elif model_name == "Gradient Boosting":
        base_estimator = GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif model_name == "Neural Network":
        base_estimator = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Handle multi-target
    if isinstance(y_train, pd.DataFrame) and y_train.shape[1] > 1:
        base_estimator = MultiOutputClassifier(base_estimator)

    # Create a pipeline with the preprocessor and estimator
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("estimator", base_estimator)
    ])

    # Update param_grid to include the pipeline's estimator prefix
    updated_param_grid = {f"estimator__{k}": v for k, v in param_grid.items()}

    # Perform grid search
    grid_search = GridSearchCV(pipeline, updated_param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    metrics = {}

    if isinstance(y_test, pd.DataFrame):
        for i, target in enumerate(y_test.columns):
            metrics[target] = {
                "accuracy": accuracy_score(y_test.iloc[:, i], y_pred[:, i] if y_pred.ndim > 1 else y_pred),
                "precision": precision_score(y_test.iloc[:, i], y_pred[:, i] if y_pred.ndim > 1 else y_pred, pos_label="CA" if target == "CA_Status" else "Y", zero_division=0),
                "recall": recall_score(y_test.iloc[:, i], y_pred[:, i] if y_pred.ndim > 1 else y_pred, pos_label="CA" if target == "CA_Status" else "Y", zero_division=0),
                "f1": f1_score(y_test.iloc[:, i], y_pred[:, i] if y_pred.ndim > 1 else y_pred, pos_label="CA" if target == "CA_Status" else "Y", zero_division=0),
            }
    else:
        metrics["target"] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, pos_label="CA", zero_division=0),
            "recall": recall_score(y_test, y_pred, pos_label="CA", zero_division=0),
            "f1": f1_score(y_test, y_pred, pos_label="CA", zero_division=0),
        }

    return best_model, metrics

def get_model_explanation(model_name, X_sample, model):
    """
    Placeholder for model explanation.
    """
    return f"Explanation for {model_name} predictions (feature importance or SHAP values can be added here)."

def plot_confusion_matrix(y_true, y_pred):
    """
    Plot confusion matrix using Plotly.
    """
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=["Non-CA", "CA"])
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=["Non-CA", "CA"], y=["Non-CA", "CA"],
        colorscale="Blues", text=cm, texttemplate="%{text}"
    ))
    fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
    return fig

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance for tree-based models.
    """
    if hasattr(model.named_steps["estimator"], "feature_importances_"):
        importances = model.named_steps["estimator"].feature_importances_
        fig = go.Figure(data=[
            go.Bar(x=feature_names, y=importances)
        ])
        fig.update_layout(title="Feature Importance", xaxis_title="Features", yaxis_title="Importance")
        return fig
    return None
