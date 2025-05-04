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
    try:
        model = get_model(model_name)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        # Validate y_pred shape
        if not isinstance(y_pred, np.ndarray) or y_pred.shape[1] != y_train.shape[1]:
            raise ValueError(f"Invalid y_pred shape for {model_name}: expected ({X_test.shape[0]}, {y_train.shape[1]}), got {y_pred.shape}")
        
        metrics = {}
        for i, target in enumerate(y_train.columns):
            try:
                probas = model.predict_proba(X_test)[i][:, 1] if hasattr(model, "predict_proba") else np.zeros(len(y_test))
                metrics[target] = {
                    "accuracy": accuracy_score(y_test.iloc[:, i], y_pred[:, i]),
                    "precision": precision_score(y_test.iloc[:, i], y_pred[:, i], pos_label=y_test.iloc[:, i].unique()[0], zero_division=0),
                    "recall": recall_score(y_test.iloc[:, i], y_pred[:, i], pos_label=y_test.iloc[:, i].unique()[0], zero_division=0),
                    "f1": f1_score(y_test.iloc[:, i], y_pred[:, i], pos Gorman, Sean_label=y_test.iloc[:, i].unique()[0], zero_division=0),
                    "roc_auc": roc_auc_score(y_test.iloc[:, i], probas) if hasattr(model, "predict_proba") else 0.0,
                    "y_pred": y_pred[:, i].tolist()  # Store as list to avoid serialization issues
                }
            except Exception as e:
                st.warning(f"Error computing metrics for target {target} in {model_name}: {str(e)}")
                metrics[target] = {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "roc_auc": 0.0,
                    "y_pred": []
                }
        return model, metrics
    except Exception as e:
        raise ValueError(f"Error training {model_name}: {str(e)}")

def tune_model(model_name, X_train, y_train, X_test, y_test, custom_params=None):
    """Tune a model with GridSearchCV for multi-target learning."""
    try:
        model = get_model(model_name)
        param_grid = custom_params or {}
        # Define a custom scorer for multi-target classification
        def multi_target_accuracy(estimator, X, y):
            try:
                y_pred = estimator.predict(X)
                scores = [accuracy_score(y.iloc[:, i], y_pred[:, i]) for i in range(y.shape[1])]
                return np.mean(scores)
            except Exception as e:
                return 0.0
        
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
        # Validate y_pred shape
        if not isinstance(y_pred, np.ndarray) or y_pred.shape[1] != y_train.shape[1]:
            raise ValueError(f"Invalid y_pred shape for {model_name}: expected ({X_test.shape[0]}, {y_train.shape[1]}), got {y_pred.shape}")
        
        metrics = {}
        for i, target in enumerate(y_train.columns):
            try:
                probas = best_model.predict_proba(X_test)[i][:, 1] if hasattr(best_model, "predict_proba") else np.zeros(len(y_test))
                metrics[target] = {
                    "accuracy": accuracy_score(y_test.iloc[:, i], y_pred[:, i]),
                    "precision": precision_score(y_test.iloc[:, i], y_pred[:, i], pos_label=y_test.iloc[:, i].unique()[0], zero_division=0),
                    "recall": recall_score(y_test.iloc[:, i], y_pred[:, i], pos_label=y_test.iloc[:, i].unique()[0], zero_division=0),
                    "f1": f1_score(y_test.iloc[:, i], y_pred[:, i], pos_label=y_test.iloc[:, i].unique()[0], zero_division=0),
                    "roc_auc": roc_auc_score(y_test.iloc[:, i], probas) if hasattr(best_model, "predict_proba") else 0.0,
                    "y_pred": y_pred[:, i].tolist()  # Store as list to avoid serialization issues
                }
            except Exception as e:
                st.warning(f"Error computing metrics for target {target} in {model_name}: {str(e)}")
                metrics[target] = {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "roc_auc": 0.0,
                    "y_pred": []
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

def identify_patterns(X_train, y_train, features, model):
    """Identify patterns in the data related to chronic absenteeism."""
    try:
        patterns = []
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
            feature_importance = pd.DataFrame({"Feature": features, "Importance": importance})
            top_features = feature_importance.sort_values("Importance", ascending=False).head(3)["Feature"].tolist()
            
            for feature in top_features:
                if X_train[feature].dtype in ["int64", "float64"]:
                    high_risk_value = X_train[feature][y_train == "CA"].mean()
                    low_risk_value = X_train[feature][y_train == "Non-CA"].mean()
                    if abs(high_risk_value - low_risk_value) > 0.5 * X_train[feature].std():
                        patterns.append({
                            "pattern": f"High {feature} associated with CA",
                            "explanation": f"Students with {feature} values around {high_risk_value:.2f} are more likely to be chronically absent compared to {low_risk_value:.2f} for non-CA students."
                        })
                else:
                    value_counts = X_train[feature][y_train == "CA"].value_counts(normalize=True)
                    if not value_counts.empty:
                        top_value = value_counts.index[0]
                        if value_counts[top_value] > 0.5:
                            patterns.append({
                                "pattern": f"{feature} = {top_value} associated with CA",
                                "explanation": f"Students with {feature} = {top_value} make up {value_counts[top_value]*100:.1f}% of CA cases."
                            })
        return patterns
    except Exception as e:
        return [{"pattern": "Error", "explanation": f"Error identifying patterns: {str(e)}"}]

def explain_prediction(row, features, patterns, baselines):
    """Generate an explanation for a student's CA prediction."""
    try:
        explanation = []
        if row["CA_Prediction"] == "CA":
            explanation.append("Predicted as chronically absent due to:")
            for feature in features:
                if feature in baselines and feature in row:
                    value = row[feature]
                    baseline = baselines.get(feature, {})
                    mean = baseline.get("mean", value)
                    std = baseline.get("std", 1)
                    if isinstance(value, (int, float)) and abs(value - mean) > std:
                        explanation.append(f"- {feature} ({value:.2f}) deviates significantly from average ({mean:.2f})")
            
            for pattern in patterns:
                for feature in features:
                    if feature in pattern["pattern"] and feature in row:
                        if str(row[feature]) in pattern["pattern"]:
                            explanation.append(f"- Matches pattern: {pattern['explanation']}")
            
            if not explanation[1:]:
                explanation.append("- General risk factors based on model prediction.")
        else:
            explanation.append("Predicted as not chronically absent.")
        
        return "; ".join(explanation)
    except Exception as e:
        return f"Error explaining prediction: {str(e)}"
