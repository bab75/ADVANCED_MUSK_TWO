import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.multioutput import MultiOutputClassifier
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Try to import shap, set flag if not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

def get_model(model_name):
    """Return the base model for a given model name."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Neural Network": MLPClassifier(max_iter=1000, random_state=42)
    }
    return MultiOutputClassifier(models.get(model_name))

def train_model(model_name, X_train, y_train, X_test, y_test):
    """Train a model and return metrics for multi-target learning."""
    model = get_model(model_name)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    metrics = {}
    for i, target in enumerate(y_train.columns):
        metrics[target] = {
            "accuracy": accuracy_score(y_test.iloc[:, i], y_pred[:, i]),
            "precision": precision_score(y_test.iloc[:, i], y_pred[:, i], pos_label=y_test.iloc[:, i].unique()[0], zero_division=0),
            "recall": recall_score(y_test.iloc[:, i], y_pred[:, i], pos_label=y_test.iloc[:, i].unique()[0], zero_division=0),
            "f1": f1_score(y_test.iloc[:, i], y_pred[:, i], pos_label=y_test.iloc[:, i].unique()[0], zero_division=0),
            "roc_auc": roc_auc_score(y_test.iloc[:, i], model.predict_proba(X_test)[i][:, 1]),
            "y_pred": y_pred[:, i]
        }
    return model, metrics

def tune_model(model_name, X_train, y_train, X_test, y_test, custom_params=None):
    """Tune a model with GridSearchCV for multi-target learning."""
    try:
        model = get_model(model_name)
        param_grid = custom_params or {}
        # Define a custom scorer for multi-target classification
        def multi_target_accuracy(estimator, X, y):
            y_pred = estimator.predict(X)
            scores = [accuracy_score(y.iloc[:, i], y_pred[:, i]) for i in range(y.shape[1])]
            return np.mean(scores)
        
        # Use n_jobs=1 to avoid multiprocessing issues
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=3,
            scoring=multi_target_accuracy,
            n_jobs=1
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        metrics = {}
        for i, target in enumerate(y_train.columns):
            # Get probabilities for each target separately
            probas = best_model.predict_proba(X_test)[i][:, 1]
            metrics[target] = {
                "accuracy": accuracy_score(y_test.iloc[:, i], y_pred[:, i]),
                "precision": precision_score(y_test.iloc[:, i], y_pred[:, i], pos_label=y_test.iloc[:, i].unique()[0], zero_division=0),
                "recall": recall_score(y_test.iloc[:, i], y_pred[:, i], pos_label=y_test.iloc[:, i].unique()[0], zero_division=0),
                "f1": f1_score(y_test.iloc[:, i], y_pred[:, i], pos_label=y_test.iloc[:, i].unique()[0], zero_division=0),
                "roc_auc": roc_auc_score(y_test.iloc[:, i], probas),
                "y_pred": y_pred[:, i]
            }
        return best_model, metrics, grid_search.best_params_
    except Exception as e:
        raise ValueError(f"Error tuning {model_name}: {str(e)}")

def get_model_explanation(model_name, X_test, model):
    """Generate model explanation using SHAP if available."""
    if SHAP_AVAILABLE:
        try:
            explainer = shap.KernelExplainer(model.predict_proba, X_test)
            shap_values = explainer.shap_values(X_test)
            return f"SHAP explanation for {model_name}: {shap_values}"
        except Exception as e:
            return f"Error generating SHAP explanation for {model_name}: {str(e)}"
    else:
        return f"SHAP explanations are not available because the 'shap' library is not installed."
