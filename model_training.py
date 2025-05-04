import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import plotly.graph_objects as go
import plotly.express as px
from data_processing import preprocess_data
from model_utils import identify_patterns, explain_prediction

def train_and_tune_model(data, features, targets, model_names, enable_tuning=False, tuning_params=None):
    """Train and tune machine learning models."""
    try:
        if not features or not targets:
            raise ValueError("Features and targets must be specified.")
        
        # Preprocess data
        X, y = preprocess_data(data, features, targets)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=1),
            "Random Forest": RandomForestClassifier(random_state=42, n_jobs=1),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "Neural Network": MLPClassifier(max_iter=1000, random_state=42)
        }
        
        model_results = {"models": {}, "model_versions": {}}
        patterns = []
        
        for model_name in model_names:
            if model_name not in models:
                st.warning(f"Model {model_name} not supported.")
                continue
            
            # Initialize MultiOutputClassifier
            base_model = models[model_name]
            model = MultiOutputClassifier(base_model, n_jobs=1)
            best_params = {}
            
            # Hyperparameter tuning
            if enable_tuning and tuning_params.get(model_name):
                param_grid = tuning_params[model_name]
                search = RandomizedSearchCV(
                    model,
                    param_distributions=param_grid,
                    n_iter=10,
                    cv=3,
                    scoring="accuracy",
                    random_state=42,
                    n_jobs=1
                )
                search.fit(X_train, y_train)
                model = search.best_estimator_
                best_params = search.best_params_
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Evaluate the model
            y_pred = model.predict(X_test)
            y_pred_proba = []
            try:
                y_pred_proba = model.predict_proba(X_test)
            except AttributeError:
                pass
            
            metrics = {}
            for i, target in enumerate(targets):
                metrics[target] = {
                    "accuracy": accuracy_score(y_test[target], y_pred[:, i]),
                    "precision": precision_score(y_test[target], y_pred[:, i], zero_division=0),
                    "recall": recall_score(y_test[target], y_pred[:, i], zero_division=0),
                    "f1": f1_score(y_test[target], y_pred[:, i], zero_division=0),
                    "roc_auc": roc 0
                }
                if y_pred_proba:
                    metrics[target]["roc_auc"] = roc_auc_score(y_test[target], y_pred_proba[i][:, 1])
            
            # Store results
            model_results["models"][model_name] = {
                "model": model,
                "metrics": metrics,
                "y_test": y_test,
                "y_pred": y_pred,
                "feature_names": X_train.columns.tolist(),
                "best_params": best_params
            }
            
            # Identify patterns
            patterns.extend(identify_patterns(X_train, y_train, features, targets))
            
            # Model versioning
            model_results["model_versions"][model_name] = {
                "version": "1.0",
                "trained_on": pd.Timestamp.now(),
                "features": features,
                "targets": targets
            }
        
        return model_results, patterns
    except Exception as e:
        raise ValueError(f"Error in model training pipeline: {str(e)}")

def run_predictions(data, features, model_name, models, drop_off_rules, patterns, baselines):
    """Run predictions on new data."""
    try:
        if model_name not in models:
            raise ValueError(f"Model {model_name} not found.")
        
        model_info = models[model_name]
        model = model_info["model"]
        training_features = model_info["feature_names"]
        
        # Validate feature consistency
        missing_features = [f for f in training_features if f not in features]
        extra_features = [f for f in features if f not in training_features]
        if missing_features or extra_features:
            st.warning(f"Feature mismatch detected. Missing: {missing_features}, Extra: {extra_features}. Adjusting features for prediction.")
            # Add missing features with default values
            for f in missing_features:
                if f == "Drop_Off":
                    data[f] = False  # Default to False for Drop_Off
                else:
                    data[f] = 0  # Default for other features (numeric or encoded)
            # Keep only training features
            features = [f for f in features if f in training_features]
        
        # Preprocess data
        X, _ = preprocess_data(data, features, [])
        
        # Ensure all training features are present after preprocessing
        for f in training_features:
            if f not in X.columns:
                X[f] = 0
        
        # Reorder columns to match training
        X = X[training_features]
        
        # Run predictions
        y_pred = model.predict(X)
        y_pred_proba = []
        try:
            y_pred_proba = model.predict_proba(X)
        except AttributeError:
            pass
        
        # Prepare output
        output = data.copy()
        for i, target in enumerate(model_info["metrics"].keys()):
            output[f"{target}_Prediction"] = y_pred[:, i]
            if y_pred_proba:
                output[f"{target}_Probability"] = y_pred_proba[i][:, 1]
        
        # Apply drop-off rules
        if drop_off_rules and "features" in drop_off_rules:
            output["Drop_Off"] = False
            for idx, row in output.iterrows():
                attendance = row["Attendance_Percentage"]
                if drop_off_rules["attendance_min"] <= attendance <= drop_off_rules["attendance_max"]:
                    for feature, values in drop_off_rules["features"].items():
                        if feature in row and row[feature] in values:
                            output.at[idx, "Drop_Off"] = True
                            break
        
        # Explain predictions
        output["Prediction_Causes"] = output.apply(
バイト

---

### Explanation of Changes

#### 1. Fixing the Prediction Error
- **Updated `app.py`**:
  - Modified the `excluded_features` list in the "Model Training" page to match the `excluded_columns` list in the "Results" page:
    ```python
    excluded_features = ["Student_ID", "CA_Prediction", "CA_Probability", "Drop_Off", "Prediction_Causes"]
    ```
    This ensures `Drop_Off` is not selected as a training feature, preventing the mismatch.
  - Added `training_features` to `st.session_state` to store the features used during training:
    ```python
    st.session_state.training_features = features
    ```
  - Added a warning in the "Run Predictions" section if prediction features differ from training features:
    ```python
    if st.session_state.training_features and set(features) != set(st.session_state.training_features):
        st.warning("Prediction features differ from training features...")
    ```

- **Updated `model_training.py`**:
  - Enhanced `run_predictions` to validate feature consistency:
    ```python
    missing_features = [f for f in training_features if f not in features]
    extra_features = [f for f in features if f not in training_features]
    if missing_features or extra_features:
        st.warning(f"Feature mismatch detected. Missing: {missing_features}, Extra: {extra_features}...")
    ```
  - Added missing features to the prediction data with default values:
    - `Drop_Off` defaults to `False`.
    - Other features default to `0` (suitable for numeric or encoded features).
  - Ensured the prediction data (`X`) includes all training features and is reordered to match:
    ```python
    X = X[training_features]
    ```
  - Preserved existing error handling and prediction logic.

- **Impact**:
  - `Drop_Off` is now excluded from training features, aligning with the prediction phase.
  - If `Drop_Off` is missing in prediction data, it’s added with a default value (`False`), ensuring compatibility.
  - Warnings guide users to align features, reducing future errors.

#### 2. Restoring the Model Selection Guide
- **Updated `app.py`**:
  - Added the provided markdown guide under the "Model Selection" subheader in an expander:
    ```python
    with st.expander("Model Selection Guide", expanded=False):
        st.markdown("""
        **Model Selection Guide**
        ...
        """)
    ```
  - Placed it before the `models_to_train` multiselect widget for visibility.

- **Impact**:
  - Restores the original user guidance, improving usability.
  - Maintains the same content and links as the original guide.

#### 3. Improving Robustness
- **Feature Consistency**:
  - Standardized `excluded_features` and `excluded_columns` to prevent mismatches.
  - Added feature validation in `run_predictions` to handle edge cases (e.g., uploaded CSVs without `Drop_Off`).
- **Session State**:
  - Stored `training_features` in `st.session_state` to track training features for comparison during prediction.
- **Error Handling**:
  - Kept try-except blocks from previous fixes.
  - Added warnings for feature mismatches to guide users without halting the app.
- **Drop-Off Rules**:
  - Ensured `Drop_Off` is generated consistently in `generate_current_year_data` if rules are provided, reducing the chance of missing columns.

#### 4. Preserving Previous Fixes
- **ValueError**: Checkbox logic (`st.session_state.data is None`) remains intact.
- **ImportError**: Imports from `data_generator.py` are correct.
- **'Non-CA'**: Target encoding in `data_processing.py` is unchanged and working (no recurrence of the error).
- **Student_ID**: `'C'` prefix in `data_generator.py` is preserved.
- **Training Issues**: Fixes in `model_utils.py` and `model_training.py` (e.g., `'y_pred'`, Logistic Regression) are retained.

---

### Deployment Instructions

To redeploy the app on Streamlit Cloud and verify the fixes:

1. **Update the Repository**:
   - Ensure your GitHub repository (`advanced_musk_two`) contains all required files:
     - `app.py` (updated with feature alignment and model selection guide)
     - `model_training.py` (updated with feature validation)
     - `data_processing.py`
     - `data_generator.py`
     - `model_utils.py`
     - `requirements.txt`
     - `styles.css`
   - Update `app.py` and `model_training.py` with the provided code.
   - Commit and push changes:
     ```bash
     git add app.py model_training.py
     git commit -m "Fix prediction feature mismatch and restore model selection guide"
     git push origin main
     ```

2. **Redeploy on Streamlit Cloud**:
   - Log in to Streamlit Cloud (streamlit.io).
   - Navigate to your app (e.g., `bab75/advance`).
   - Click "Manage app" in the lower right.
   - Select "Reboot" to redeploy, or confirm the latest commit is used.
   - Check the "Logs" tab to verify no errors occur.

3. **Test the App**:
   - Open the deployed app.
   - **Test Data Generation**:
     - In "Data Configuration", generate historical data with drop-off rules enabled.
     - Verify `Student_ID` starts with `'C'` and `Drop_Off` is present in the dataset.
   - **Test Model Training**:
     - Go to "Model Training", select a dataset, choose features (excluding `Drop_Off`), select `CA_Status` as the target, and train a Random Forest model.
     - Verify the "Model Selection Guide" expander is present and contains the correct markdown.
     - Confirm training completes without errors.
   - **Test Predictions**:
     - In "Results", generate current year data with drop-off rules enabled.
     - Run predictions using the trained model, ensuring the same features are selected (excluding `Drop_Off`).
     - Verify no feature mismatch error occurs and predictions complete successfully.
     - Upload a CSV without `Drop_Off` and test predictions to confirm the default value (`False`) is applied.
   - **Test Edge Cases**:
     - Train with different feature sets and check for warnings if prediction features differ.
     - Clear session state and verify the app handles `None` states correctly.

4. **Troubleshoot Issues**:
   - Check logs in "Manage app" for new errors.
   - Verify all files are in the repository (e.g., `ls` in the repo directory).
   - Ensure `styles.css` exists, or remove the CSS loading line in `app.py` if not needed:
     ```python
     st.markdown('<style>' + open('styles.css').read() + '</style>', unsafe_allow_html=True)
     ```
   - If errors persist, share the full traceback and steps to reproduce.

---

### Testing Locally (Optional)

To confirm the fixes before redeploying:

1. **Set Up Environment**:
   - Save all files in a local directory.
   - Create a virtual environment:
     ```bash
     python3.12 -m venv venv
     source venv/bin/activate
     pip install -r requirements.txt
     ```

2. **Run the App**:
   - Start the Streamlit app:
     ```bash
     streamlit run app.py
     ```
   - Verify no errors occur on startup.

3. **Test Functionality**:
   - Follow the Streamlit Cloud testing steps.
   - Specifically, test predictions with and without `Drop_Off` in the data.
   - Confirm the "Model Selection Guide" is displayed correctly.

---

### Notes
- The prediction error was caused by a feature mismatch (`Drop_Off` in training but not in prediction), fixed by aligning excluded features and adding default values.
- The model selection guide was restored exactly as provided, enhancing user experience.
- Previous fixes (`ValueError`, `ImportError`, `'Non-CA'`, `Student_ID`, `'y_pred'`) are preserved, and the app is now stable for training and prediction.
- To prevent future errors:
  - Maintain consistent `excluded_features` and `excluded_columns` lists.
  - Validate features in both training and prediction phases.
  - Document session state and feature dependencies (e.g., `training_features`).
- If errors persist, please share:
  - Full error log from Streamlit Cloud.
  - Steps to reproduce (e.g., feature selection, drop-off rules, CSV upload).
  - Repository file list to confirm all files are present.
- If you need further enhancements (e.g., additional guides, UI improvements), let me know!

Let me know if you encounter other issues or need further assistance!
