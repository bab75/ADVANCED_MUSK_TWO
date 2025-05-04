from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import plotly.graph_objects as go
import numpy as np

def train_model(model_name, X_train, y_train, X_test, y_test):
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_name == "SVM":
        model = SVC(probability=True, random_state=42)
    elif model_name == "Gradient Boosting":
        model = GradientBoostingClassifier(random_state=42)
    elif model_name == "Neural Network":
        model = MLPClassifier(max_iter=1000, random_state=42, solver='adam', early_stopping=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label="CA"),
        "recall": recall_score(y_test, y_pred, pos_label="CA"),
        "f1": f1_score(y_test, y_pred, pos_label="CA"),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "y_pred": y_pred
    }
    
    return model, metrics

def tune_model(model_name, X_train, y_train, X_test, y_test, custom_params=None):
    param_grid = custom_params if custom_params else {}
    
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
        if not param_grid:
            param_grid = {
                "C": [0.1, 1, 10],
                "solver": ["lbfgs", "liblinear"]
            }
    elif model_name == "Random Forest":
        model = RandomForestClassifier(random_state=42)
        if not param_grid:
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5]
            }
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
        if not param_grid:
            param_grid = {
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5]
            }
    elif model_name == "SVM":
        model = SVC(probability=True, random_state=42)
        if not param_grid:
            param_grid = {
                "C": [0.1, 1, 10],
                "kernel": ["rbf", "linear"]
            }
    elif model_name == "Gradient Boosting":
        model = GradientBoostingClassifier(random_state=42)
        if not param_grid:
            param_grid = {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5]
            }
    elif model_name == "Neural Network":
        model = MLPClassifier(max_iter=1000, random_state=42, solver='adam', early_stopping=True)
        if not param_grid:
            param_grid = {
                "hidden_layer_sizes": [(50,), (100,), (50, 50)],
                "alpha": [0.0001, 0.001],
                "learning_rate": ["constant", "adaptive"]
            }
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring="f1", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label="CA"),
        "recall": recall_score(y_test, y_pred, pos_label="CA"),
        "f1": f1_score(y_test, y_pred, pos_label="CA"),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "y_pred": y_pred
    }
    
    return best_model, metrics, grid_search.best_params_

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=["NO-CA", "CA"])
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["NO-CA", "CA"],
        y=["NO-CA", "CA"],
        colorscale="Blues",
        showscale=True
    ))
    fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
    return fig

def plot_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[feature_names[i] for i in indices],
            y=importances[indices],
            marker_color="#3498db"
        ))
        fig.update_layout(title="Feature Importance", xaxis_title="Feature", yaxis_title="Importance")
        return fig
    return None

def get_model_explanation(model_name, X_sample=None, model=None):
    explanations = {
        "Logistic Regression": "Logistic Regression models the probability of chronic absenteeism using a logistic function. It assumes a linear relationship between features and the log-odds of the outcome.",
        "Random Forest": "Random Forest builds multiple decision trees and combines their predictions to improve accuracy and reduce overfitting. It captures complex interactions between features.",
        "Decision Tree": "Decision Tree splits the data into branches based on feature values, making decisions to classify students as CA or NO-CA. It’s interpretable but can overfit.",
        "SVM": "SVM finds the optimal hyperplane that separates CA and NO-CA students, maximizing the margin between classes. It uses kernel tricks for non-linear boundaries.",
        "Gradient Boosting": "Gradient Boosting builds trees sequentially, each correcting the errors of the previous ones. It’s powerful for capturing patterns in absenteeism data.",
        "Neural Network": "Neural Network uses layers of interconnected nodes to model complex relationships. It’s effective for large datasets but requires careful tuning."
    }
    
    explanation = explanations.get(model_name, "No explanation available.")
    
    if X_sample is not None and model is not None and model_name in ["Logistic Regression", "Random Forest", "Decision Tree", "Gradient Boosting"]:
        try:
            pred = model.predict(X_sample)[0]
            prob = model.predict_proba(X_sample)[0][1]
            explanation += f"\n\n**Example Prediction**: For a sample student, the model predicts '{pred}' with a CA probability of {prob:.2f}."
        except:
            pass
    
    return explanation
