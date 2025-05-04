import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import uuid
import os
import tempfile
from data_generator import generate_historical_data, generate_current_year_data
from model_utils import train_model, tune_model, get_model_explanation, plot_confusion_matrix, plot_feature_importance, recommend_models

# Set page config
st.set_page_config(page_title="Chronic Absenteeism Prediction", layout="wide")

# Load CSS for styling
st.markdown('<style>' + open('styles.css').read() + '</style>', unsafe_allow_html=True)

# Function to clear session state
def clear_session_state():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.data = None
    st.session_state.datasets = {}
    st.session_state.models = {}
    st.session_state.current_data = None
    st.session_state.custom_fields = []
    st.session_state.current_custom_fields = []
    st.session_state.selected_student_id = None
    st.session_state.model_versions = {}
    st.session_state.patterns = []
    st.session_state.page = "📝 Data Configuration"
    st.session_state.compare_models = []

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'custom_fields' not in st.session_state:
    st.session_state.custom_fields = []
if 'current_custom_fields' not in st.session_state:
    st.session_state.current_custom_fields = []
if 'selected_student_id' not in st.session_state:
    st.session_state.selected_student_id = None
if 'model_versions' not in st.session_state:
    st.session_state.model_versions = {}
if 'patterns' not in st.session_state:
    st.session_state.patterns = []
if 'page' not in st.session_state:
    st.session_state.page = "📝 Data Configuration"
if 'compare_models' not in st.session_state:
    st.session_state.compare_models = []

# Sidebar navigation with radio buttons
st.sidebar.title("Navigation")
page_options = [
    "📝 Data Configuration",
    "🤖 Model Training",
    "📊 Results",
    "📚 Documentation"
]
default_index = page_options.index(st.session_state.page) if st.session_state.page in page_options else 0
page = st.sidebar.radio("Go to", page_options, index=default_index, label_visibility="collapsed")
st.session_state.page = page

# Clear All Data Button
if st.sidebar.button("Clear All Data"):
    clear_session_state()
    st.rerun()

# Page 1: Data Configuration
if st.session_state.page == "📝 Data Configuration":
    st.markdown("""
    <h1 class="section-header">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#3498db" class="header-icon">
            <path d="M3 3h18v18H3V3zm2 2v14h14V5H5zm2 2h10v2H7V7zm0 4h10v2H7v-2zm0 4h7v2H7v-2z"/>
        </svg>
        Data Configuration
    </h1>
    """, unsafe_allow_html=True)
    
    # Dataset Settings
    st.subheader("Generate Historical Data")
    with st.container():
        num_students = st.slider("Number of Students", 100, 5000, 1000)
        year_start, year_end = st.slider("Academic Years", 2010, 2025, (2015, 2020), step=1)
        school_prefix = st.text_input("School Prefix (e.g., 10U)", "10U")
        num_schools = st.number_input("Number of Schools", 1, 10, 3)
    
    # Student Demographics
    st.subheader("Student Demographics")
    with st.container():
        grades = st.multiselect("Grades", list(range(1, 13)), default=[1, 2, 3, 4, 5])
        col1, col2, col3 = st.columns(3)
        with col1:
            male_dist = st.slider("Male (%)", 0, 100, 40, step=5)
        with col2:
            female_dist = st.slider("Female (%)", 0, 100, 40, step=5)
        with col3:
            other_dist = st.slider("Other (%)", 0, 100, 20, step=5)
        
        total_dist = male_dist + female_dist + other_dist
        if total_dist != 100:
            st.error(f"Gender distribution must sum to 100%. Current total: {total_dist}%")
            gender_dist = None
        else:
            gender_dist = [male_dist, female_dist, other_dist]
    
    # School Settings
    st.subheader("School Settings")
    with st.container():
        meal_codes = st.multiselect("Meal Codes", ["Free", "Reduced", "Paid"], default=["Free", "Reduced", "Paid"])
        academic_perf = st.slider("Academic Performance Range (%)", 1, 100, (40, 90))
        transportation = st.multiselect("Transportation Options", ["Bus", "Walk", "Car"], default=["Bus", "Walk"])
        suspensions_range = st.slider("Suspensions Range (per year)", 0, 10, (0, 3))
    
    # Attendance Settings
    st.subheader("Attendance Settings")
    with st.container():
        total_days = 180
        st.write(f"Total School Days: {total_days}")
        present_days_range = st.slider("Present Days Range", 0, total_days, (100, total_days))
        
        # Dynamically constrain absent_days_range based on present_days_range
        max_absent_days = total_days - present_days_range[0]
        absent_days_range = st.slider(
            "Absent Days Range",
            0,
            max_absent_days,
            (0, min(80, max_absent_days)),
            help=f"Maximum absent days cannot exceed {max_absent_days} (total days - minimum present days)."
        )
        
        # Enhanced validation for attendance ranges
        attendance_valid = True
        if present_days_range[0] + absent_days_range[1] > total_days:
            st.error(f"Error: Minimum present days ({present_days_range[0]}) plus maximum absent days ({absent_days_range[1]}) exceeds total days ({total_days}). Reduce absent days range.")
            attendance_valid = False
        elif present_days_range[1] + absent_days_range[0] < total_days:
            st.error(f"Error: Maximum present days ({present_days_range[1]}) plus minimum absent days ({absent_days_range[0]}) is less than total days ({total_days}). Increase present or absent days range.")
            attendance_valid = False
        elif present_days_range[1] < present_days_range[0] or absent_days_range[1] < absent_days_range[0]:
            st.error("Error: Range maximum must be greater than or equal to minimum.")
            attendance_valid = False
        elif total_days - present_days_range[0] < absent_days_range[0]:
            st.error(f"Error: Maximum possible absent days ({total_days - present_days_range[0]}) is less than minimum absent days ({absent_days_range[0]}). Reduce minimum absent days or lower minimum present days.")
            attendance_valid = False
        elif total_days - present_days_range[1] < absent_days_range[0]:
            st.error(f"Error: Minimum absent days ({absent_days_range[0]}) cannot be achieved with maximum present days ({present_days_range[1]}). Increase maximum absent days or reduce maximum present days.")
            attendance_valid = False
    
    # Custom Fields
    st.subheader("Custom Fields")
    with st.container():
        if st.button("Add Custom Field"):
            st.session_state.custom_fields.append({"name": "", "values": ""})
        
        for i, field in enumerate(st.session_state.custom_fields):
            col1, col2, col3 = st.columns([3, 3, 1])
            with col1:
                field["name"] = st.text_input(f"Custom Field {i+1} Name", key=f"name_{i}")
            with col2:
                field["values"] = st.text_input(f"Custom Field {i+1} Values (comma-separated)", key=f"values_{i}")
            with col3:
                if st.button("Remove", key=f"remove_{i}"):
                    st.session_state.custom_fields.pop(i)
                    st.rerun()
    
    if st.button("Generate Historical Data") and gender_dist is not None and attendance_valid:
        try:
            st.session_state.patterns = []
            custom_fields = [(f["name"], f["values"]) for f in st.session_state.custom_fields if f["name"] and f["values"]]
            data = generate_historical_data(
                num_students, year_start, year_end, school_prefix, num_schools,
                grades, gender_dist, meal_codes, academic_perf, transportation,
                suspensions_range, present_days_range, absent_days_range, total_days, custom_fields
            )
            st.session_state.data = data
            dataset_name = f"Dataset_{datetime.now().strftime('%Y-%m-%d_%H:%M')}"
            st.session_state.datasets[dataset_name] = data
            st.success(f"Data generated successfully! Saved as {dataset_name}")
            
            st.subheader("Data Preview")
            st.dataframe(data.head(10))
            
            csv = data.to_csv(index=False)
            st.download_button("Download Historical Data", csv, "historical_data.csv", "text/csv")
        except Exception as e:
            st.error(f"Error generating data: {str(e)}")

# Page 2: Model Training
elif st.session_state.page == "🤖 Model Training":
    st.markdown("""
    <h1 class="section-header">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#3498db" class="header-icon">
            <path d="M12 2a10 10 0 00-8 4v2h2v2H4v2h2v2H4v2h2v2h2a10 10 0 008-4 10 10 0 008 4h2v-2h-2v-2h2v-2h-2v-2h2V8h-2V6a10 10 0 00-8-4zm0 2a8 8 0 016.32 3H17v2h-2v2h2v2h-2v2h2v2h-1.32A8 8 0 0112 20a8 8 0 01-6.32-3H7v-2H5v-2h2v-2H5V9h2V7h1.32A8 8 0 0112 4z"/>
        </svg>
        Model Training
    </h1>
    """, unsafe_allow_html=True)
    
    with st.expander("Data Source", expanded=True):
        data_source = st.radio("Data Source", ["Use Generated Data", "Upload CSV"])
        
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload Historical Data (CSV)", type=["csv"])
            if uploaded_file:
                try:
                    data = pd.read_csv(uploaded_file)
                    dataset_name = f"Uploaded_{datetime.now().strftime('%Y-%m-%d_%H:%M')}"
                    st.session_state.datasets[dataset_name] = data
                    st.session_state.data = data
                    st.session_state.patterns = []
                    st.success(f"Data uploaded successfully! Saved as {dataset_name}")
                except Exception as e:
                    st.error(f"Error uploading data: {str(e)}")
    
    with st.expander("Dataset & Target Selection"):
        if st.session_state.datasets:
            selected_dataset = st.selectbox("Select Historical Dataset", list(st.session_state.datasets.keys()))
            st.session_state.data = st.session_state.datasets[selected_dataset]
            
            target_options = [col for col in st.session_state.data.columns if col not in ["Student_ID"]]
            target = st.selectbox("Select Target Variable", target_options, index=target_options.index("CA_Status") if "CA_Status" in target_options else 0)
            
            st.subheader("Model Recommendations")
            recommendations = recommend_models(st.session_state.data[target])
            st.write("**Recommended Models**:")
            for model, reason in recommendations.items():
                available = model in st.session_state.models
                status = "✅ Available" if available else "⚠️ Not Trained"
                st.write(f"- **{model}**: {reason} [{status}]")
        else:
            st.warning("No datasets available. Generate or upload data in Data Configuration.")
            target = "CA_Status"
    
    if st.session_state.data is not None:
        with st.expander("Feature Selection"):
            include_student_id = st.checkbox("Include Historical Student ID", value=True)
            excluded_features = [target]
            if not include_student_id:
                excluded_features.append("Student_ID")
            available_features = [col for col in st.session_state.data.columns if col not in excluded_features]
            feature_toggles = {}
            st.write("Toggle Features:")
            for feature in available_features:
                feature_toggles[feature] = st.checkbox(feature, value=True, key=f"feature_{feature}")
            features = [f for f, enabled in feature_toggles.items() if enabled]
            
            categorical_cols = [col for col in features if st.session_state.data[col].dtype == "object"]
            numerical_cols = [col for col in features if col not in categorical_cols]
        
        with st.expander("Model Selection"):
            st.subheader("Model Selection Guide")
            st.markdown("""
            **Model Selection Guide**

            Choose machine learning models to predict chronic absenteeism. Each model has unique strengths:

            - **Logistic Regression**: Models the probability of absenteeism using a linear relationship. Best for interpretable results.
              - [Learn More](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
            - **Random Forest**: Combines multiple decision trees to improve accuracy and handle complex patterns. Robust to overfitting.
              - [Learn More](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)
            - **Decision Tree**: Splits data into branches based on feature values. Easy to interpret but may overfit.
              - [Learn More](https://scikit-learn.org/stable/modules/tree.html)
            - **SVM**: Finds the optimal boundary to separate classes. Effective for non-linear data.
              - [Learn More](https://scikit-learn.org/stable/modules/svm.html)
            - **Gradient Boosting**: Builds trees sequentially to correct errors. Powerful for predictive accuracy.
              - [Learn More](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)
            - **Neural Network**: Models complex relationships with layered nodes. Suitable for large datasets but requires tuning.
              - [Learn More](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)

            Select multiple models to compare their performance. Use hyperparameter tuning for optimized results.
            For a deeper dive, read [this guide on machine learning models](https://towardsdatascience.com/the-7-most-common-machine-learning-models-8e8d6c0e1c5c).
            """)
            
            models_to_train = st.multiselect("Select Models", [
                "Logistic Regression", "Random Forest", "Decision Tree",
                "SVM", "Gradient Boosting", "Neural Network"
            ], default=["Logistic Regression", "Random Forest"])
            
            enable_tuning = st.checkbox("Enable Hyperparameter Tuning", value=False)
            tuning_params = {}
            if enable_tuning:
                st.subheader("Customize Hyperparameter Tuning")
                for model_name in models_to_train:
                    st.write(f"**{model_name} Parameters**")
                    if model_name == "Logistic Regression":
                        tuning_params[model_name] = {
                            "C": st.multiselect(f"C values ({model_name})", [0.1, 1, 10], default=[0.1, 1, 10]),
                            "solver": st.multiselect(f"Solver ({model_name})", ["lbfgs", "liblinear"], default=["lbfgs", "liblinear"])
                        }
                    elif model_name == "Random Forest":
                        tuning_params[model_name] = {
                            "n_estimators": st.multiselect(f"N Estimators ({model_name})", [100, 200], default=[100, 200]),
                            "max_depth": st.multiselect(f"Max Depth ({model_name})", [None, 10, 20], default=[None, 10, 20]),
                            "min_samples_split": st.multiselect(f"Min Samples Split ({model_name})", [2, 5], default=[2, 5])
                        }
                    elif model_name == "Decision Tree":
                        tuning_params[model_name] = {
                            "max_depth": st.multiselect(f"Max Depth ({model_name})", [None, 10, 20], default=[None, 10, 20]),
                            "min_samples_split": st.multiselect(f"Min Samples Split ({model_name})", [2, 5], default=[2, 5])
                        }
                    elif model_name == "SVM":
                        tuning_params[model_name] = {
                            "C": st.multiselect(f"C values ({model_name})", [0.1, 1, 10], default=[0.1, 1, 10]),
                            "kernel": st.multiselect(f"Kernel ({model_name})", ["rbf", "linear"], default=["rbf", "linear"])
                        }
                    elif model_name == "Gradient Boosting":
                        tuning_params[model_name] = {
                            "n_estimators": st.multiselect(f"N Estimators ({model_name})", [100, 200], default=[100, 200]),
                            "learning_rate": st.multiselect(f"Learning Rate ({model_name})", [0.01, 0.1], default=[0.01, 0.1]),
                            "max_depth": st.multiselect(f"Max Depth ({model_name})", [3, 5], default=[3, 5])
                        }
                    elif model_name == "Neural Network":
                        tuning_params[model_name] = {
                            "hidden_layer_sizes": st.multiselect(f"Hidden Layer Sizes ({model_name})", [(50,), (100,), (50, 50)], default=[(50,), (100,)]),
                            "alpha": st.multiselect(f"Alpha ({model_name})", [0.0001, 0.001], default=[0.0001, 0.001]),
                            "learning_rate": st.multiselect(f"Learning Rate ({model_name})", ["constant", "adaptive"], default=["constant", "adaptive"])
                        }
        
        if st.button("Train Models"):
            try:
                with st.spinner("Training models..."):
                    X = st.session_state.data[features]
                    y = st.session_state.data[target]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    preprocessor = ColumnTransformer(
                        transformers=[
                            ("num", StandardScaler(), numerical_cols),
                            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_cols)
                        ])
                    
                    st.session_state.preprocessor = preprocessor
                    X_train_processed = preprocessor.fit_transform(X_train)
                    X_test_processed = preprocessor.transform(X_test)
                    
                    feature_names = numerical_cols + list(preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols))
                    
                    st.subheader("Training Status")
                    status_container = st.empty()
                    
                    st.session_state.models = {}
                    comparison_data = []
                    st.session_state.patterns = []
                    for model_name in models_to_train:
                        try:
                            status_container.write(f"Training {model_name}...")
                            if enable_tuning and model_name in tuning_params:
                                model, metrics, best_params = tune_model(model_name, X_train_processed, y_train, X_test_processed, y_test)
                            else:
                                model, metrics = train_model(model_name, X_train_processed, y_train, X_test_processed, y_test)
                                best_params = None
                            
                            # Store y_test, y_pred, and X_test_processed
                            y_pred = model.predict(X_test_processed)
                            metrics['y_pred'] = y_pred
                            
                            version_id = str(uuid.uuid4())
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            st.session_state.model_versions.setdefault(model_name, []).append({
                                "version_id": version_id,
                                "timestamp": timestamp,
                                "metrics": metrics,
                                "best_params": best_params,
                                "model": model,
                                "preprocessor": preprocessor,
                                "feature_names": feature_names,
                                "y_test": y_test,
                                "y_pred": y_pred,
                                "X_test_processed": X_test_processed
                            })
                            
                            st.session_state.models[model_name] = {
                                "model": model,
                                "metrics": metrics,
                                "preprocessor": preprocessor,
                                "feature_names": feature_names,
                                "best_params": best_params,
                                "y_test": y_test,
                                "y_pred": y_pred,
                                "X_test_processed": X_test_processed
                            }
                            
                            comparison_data.append({
                                "Model": model_name,
                                "Version": version_id,
                                "Timestamp": timestamp,
                                "Accuracy": metrics['accuracy'],
                                "Precision": metrics['precision'],
                                "Recall": metrics['recall'],
                                "F1 Score": metrics['f1'],
                                "ROC AUC": metrics['roc_auc']
                            })
                            
                            status_container.success(f"{model_name} trained successfully!")
                        except Exception as e:
                            status_container.error(f"Error training {model_name}: {str(e)}")
                    
                    # Generate patterns after training
                    high_risk = st.session_state.data[st.session_state.data[target] == "CA"]
                    if not high_risk.empty:
                        patterns = []
                        low_attendance = f"Average Attendance: {high_risk['Attendance_Percentage'].mean():.2f}%"
                        common_grades = f"Common Grades: {', '.join(map(str, high_risk['Grade'].mode().tolist()))}"
                        common_meal_codes = f"Common Meal Codes: {', '.join(high_risk['Meal_Code'].mode().tolist())}"
                        common_transport = f"Common Transportation: {', '.join(high_risk['Transportation'].mode().tolist())}"
                        
                        existing_patterns = [p["pattern"] for p in st.session_state.patterns]
                        for pattern in [low_attendance, common_grades, common_meal_codes, scommon_transport]:
                            if pattern not in existing_patterns:
                                patterns.append({"pattern": pattern, "explanation": "Identified in high-risk students"})
                        
                        st.session_state.patterns.extend(patterns)
                    
                    st.success("All models trained successfully!")
                    st.balloons()
            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
        
        with st.expander("Model Results"):
            for model_name in models_to_train:
                if model_name in st.session_state.models:
                    model_info = st.session_state.models[model_name]
                    metrics = model_info["metrics"]
                    
                    st.subheader(f"{model_name} Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        <div class="model-icon">
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#3498db"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z"/></svg>
                            {model_name}
                        </div>
                        """, unsafe_allow_html=True)
                        st.write(f"Accuracy: {metrics['accuracy']:.2f}")
                        st.write(f"Precision: {metrics['precision']:.2f}")
                        st.write(f"Recall: {metrics['recall']:.2f}")
                        st.write(f"F1 Score: {metrics['f1']:.2f}")
                        st.write(f"ROC AUC: {metrics['roc_auc']:.2f}")
                        if model_info["best_params"]:
                            st.write("Best Parameters:")
                            st.json(model_info["best_params"])
                    with col2:
                        fig = plot_confusion_matrix(model_info["y_test"], model_info["y_pred"])
                        fig.update_traces(hovertemplate="Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}")
                        fig.update_layout(width=600, height=400, margin=dict(l=50, r=50, t=50, b=50))
                        st.plotly_chart(fig, use_container_width=False)
                    
                    if model_name in ["Random Forest", "Decision Tree", "Gradient Boosting"]:
                        fig = plot_feature_importance(model_info["model"], model_info["feature_names"])
                        if fig:
                            fig.update_traces(hovertemplate="Feature: %{x}<br>Importance: %{y:.4f}")
                            fig.update_layout(width=600, height=400, margin=dict(l=50, r=50, t=50, b=50))
                            st.plotly_chart(fig, use_container_width=False)
                    
                    st.write(get_model_explanation(model_name, model_info["X_test_processed"][:1], model_info["model"]))
        
        with st.expander("Model Comparison"):
            st.subheader("Model Selection & Versioning Dashboard")
            if st.session_state.model_versions:
                version_data = []
                for model_name, versions in st.session_state.model_versions.items():
                    for version in versions:
                        version_data.append({
                            "Model": model_name,
                            "Version ID": version["version_id"],
                            "Timestamp": version["timestamp"],
                            "Accuracy": version["metrics"]["accuracy"],
                            "Precision": version["metrics"]["precision"],
                            "Recall": version["metrics"]["recall"],
                            "F1 Score": version["metrics"]["f1"],
                            "ROC AUC": version["metrics"]["roc_auc"]
                        })
                version_df = pd.DataFrame(version_data)
                st.dataframe(version_df)
                
                with st.form("compare_form"):
                    st.write("Compare Model Versions")
                    compare_models = st.multiselect(
                        "Select Models to Compare",
                        list(st.session_state.model_versions.keys()),
                        default=st.session_state.compare_models,
                        key="compare_models_select"
                    )
                    if st.form_submit_button("Compare"):
                        st.session_state.compare_models = compare_models
                        if compare_models:
                            compare_data = []
                            for model_name in compare_models:
                                for version in st.session_state.model_versions[model_name]:
                                    compare_data.append({
                                        "Model": f"{model_name} ({version['version_id'][:8]})",
                                        "Accuracy": version["metrics"]["accuracy"],
                                        "Precision": version["metrics"]["precision"],
                                        "Recall": version["metrics"]["recall"],
                                        "F1 Score": version["metrics"]["f1"],
                                        "ROC AUC": version["metrics"]["roc_auc"]
                                    })
                            if compare_data:
                                compare_df = pd.DataFrame(compare_data)
                                fig = go.Figure()
                                for metric in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]:
                                    fig.add_trace(go.Bar(
                                        x=compare_df["Model"],
                                        y=compare_df[metric],
                                        name=metric,
                                        hovertemplate=f"{metric}: %{{y:.2f}}"
                                    ))
                                fig.update_layout(
                                    title="Model Version Comparison",
                                    barmode="group",
                                    xaxis_title="Model (Version)",
                                    yaxis_title="Score",
                                    width=600,
                                    height=400,
                                    margin=dict(l=50, r=50, t=50, b=50)
                                )
                                st.markdown("""
                                **About Model Comparison Chart**

                                This bar chart compares the performance of selected model versions across key metrics:
                                - **Accuracy**: Proportion of correct predictions.
                                - **Precision**: Proportion of positive predictions that were correct.
                                - **Recall**: Proportion of actual positives correctly identified.
                                - **F1 Score**: Harmonic mean of precision and recall.
                                - **ROC AUC**: Area under the receiver operating characteristic curve, measuring model discrimination.

                                Use this to identify the best-performing models for chronic absenteeism prediction.
                                Hover over bars to see exact metric values.
                                """)
                                st.plotly_chart(fig, use_container_width=False)
                        else:
                            st.warning("Please select at least one model to compare.")
        
        with st.expander("Pattern Discovery"):
            if st.session_state.models:
                st.write(f"Number of patterns learned: {len(st.session_state.patterns)}")
                if st.session_state.patterns:
                    for pattern in st.session_state.patterns:
                        st.write(f"- {pattern['pattern']}: {pattern['explanation']}")
                else:
                    st.info("No patterns identified yet.")
            else:
                st.info("No patterns discovered yet. Train models to identify patterns.")

# Page 3: Results
elif st.session_state.page == "📊 Results":
    st.markdown("""
    <h1 class="section-header">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#3498db" class="header-icon">
            <path d="M3 3h18v18H3V3zm2 2v14h14V5H5zm2 2h2v6H7V7zm4 0h2v10h-2V7zm4 0h2v4h-2V7z"/>
        </svg>
        Results
    </h1>
    """, unsafe_allow_html=True)
    
    with st.expander("Generate Current Year Data", expanded=True):
        data_source = st.radio("Data Source", ["Generate Data", "Upload CSV"])
        
        if data_source == "Generate Data":
            num_students = st.slider("Number of Students", 100, 5000, 1000)
            school_prefix = st.text_input("School Prefix (e.g., CU)", "CU")
            num_schools = st.number_input("Number of Schools", 1, 10, 3)
            
            grades = st.multiselect("Grades", list(range(1, 13)), default=[1, 2, 3, 4, 5], key="current_grades")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                male_dist = st.slider("Male (%)", 0, 100, 40, step=5, key="current_male")
            with col2:
                female_dist = st.slider("Female (%)", 0, 100, 40, step=5, key="current_female")
            with col3:
                other_dist = st.slider("Other (%)", 0, 100, 20, step=5, key="current_other")
            
            total_dist = male_dist + female_dist + other_dist
            if total_dist != 100:
                st.error(f"Gender distribution must sum to 100%. Current total: {total_dist}%")
                gender_dist = None
            else:
                gender_dist = [male_dist, female_dist, other_dist]
            
            meal_codes = st.multiselect("Meal Codes", ["Free", "Reduced", "Paid"], default=["Free", "Reduced", "Paid"], key="current_meal")
            academic_perf = st.slider("Academic Performance Range (%)", 1, 100, (40, 90), key="current_academic")
            
            transportation = st.multiselect("Transportation Options", ["Bus", "Walk", "Car"], default=["Bus", "Walk"], key="current_transport")
            suspensions_range = st.slider("Suspensions Range (per year)", 0, 10, (0, 3), key="current_suspensions")
            
            total_days = 180
            st.write(f"Total School Days: {total_days}")
            present_days_range = st.slider("Present Days Range", 0, total_days, (100, total_days), key="current_present")
            
            # Dynamically constrain absent_days_range based on present_days_range
            max_absent_days = total_days - present_days_range[0]
            absent_days_range = st.slider(
                "Absent Days Range",
                0,
                max_absent_days,
                (0, min(80, max_absent_days)),
                key="current_absent",
                help=f"Maximum absent days cannot exceed {max_absent_days} (total days - minimum present days)."
            )
            
            # Enhanced validation for current year data
            current_attendance_valid = True
            if present_days_range[0] + absent_days_range[1] > total_days:
                st.error(f"Error: Minimum present days ({present_days_range[0]}) plus maximum absent days ({absent_days_range[1]}) exceeds total days ({total_days}). Reduce absent days range.")
                current_attendance_valid = False
            elif present_days_range[1] + absent_days_range[0] < total_days:
                st.error(f"Error: Maximum present days ({present_days_range[1]}) plus minimum absent days ({absent_days_range[0]}) is less than total days ({total_days}). Increase present or absent days range.")
                current_attendance_valid = False
            elif total_days - present_days_range[0] < absent_days_range[0]:
                st.error(f"Error: Maximum possible absent days ({total_days - present_days_range[0]}) is less than minimum absent days ({absent_days_range[0]}). Reduce minimum absent days or lower minimum present days.")
                current_attendance_valid = False
            elif total_days - present_days_range[1] < absent_days_range[0]:
                st.error(f"Error: Minimum absent days ({absent_days_range[0]}) cannot be achieved with maximum present days ({present_days_range[1]}). Increase maximum absent days or reduce maximum present days.")
                current_attendance_valid = False
            
            use_historical_ids = st.checkbox("Use Historical Student IDs", value=False, disabled=st.session_state.data is None)
            
            if st.button("Generate Current Year Data") and gender_dist is not None and current_attendance_valid:
                try:
                    custom_fields = [(f["name"], f["values"]) for f in st.session_state.current_custom_fields if f["name"] and f["values"]]
                    historical_ids = st.session_state.data["Student_ID"].tolist() if use_historical_ids and st.session_state.data is not None else None
                    st.session_state.current_data = generate_current_year_data(
                        num_students, school_prefix, num_schools, grades, gender_dist,
                        meal_codes, academic_perf, transportation, suspensions_range,
                        present_days_range, absent_days_range, total_days, custom_fields,
                        historical_ids=historical_ids
                    )
                    st.success("Current year data generated successfully!")
                    st.dataframe(st.session_state.current_data.head())
                except Exception as e:
                    st.error(f"Error generating data: {str(e)}")
        else:
            uploaded_file = st.file_uploader("Upload Current Year Data (CSV)", type=["csv"])
            if uploaded_file:
                try:
                    st.session_state.current_data = pd.read_csv(uploaded_file)
                    st.success("Data uploaded successfully!")
                except Exception as e:
                    st.error(f"Error uploading data: {str(e)}")
    
    with st.expander("Custom Fields"):
        if st.button("Add Custom Field", key="add_current_custom"):
            st.session_state.current_custom_fields.append({"name": "", "values": ""})
        
        for i, field in enumerate(st.session_state.current_custom_fields):
            col1, col2, col3 = st.columns([3, 3, 1])
            with col1:
                field["name"] = st.text_input(f"Custom Field {i+1} Name", key=f"current_name_{i}")
            with col2:
                field["values"] = st.text_input(f"Custom Field {i+1} Values (comma-separated)", key=f"current_values_{i}")
            with col3:
                if st.button("Remove", key=f"current_remove_{i}"):
                    st.session_state.current_custom_fields.pop(i)
                    st.rerun()
    
    if st.session_state.current_data is not None and st.session_state.models:
        with st.expander("Run Predictions"):
            selected_model = st.selectbox("Select Model", list(st.session_state.models.keys()))
            available_features = [col for col in st.session_state.current_data.columns if col not in ["Student_ID", "CA_Prediction", "CA_Probability"]]
            feature_toggles = {}
            st.write("Toggle Features:")
            for feature in available_features:
                feature_toggles[feature] = st.checkbox(feature, value=True, key=f"predict_feature_{feature}")
            features = [f for f, enabled in feature_toggles.items() if enabled]
            
            if st.button("Predict"):
                try:
                    model_info = st.session_state.models[selected_model]
                    X = st.session_state.current_data[features]
                    X_processed = model_info["preprocessor"].transform(X)
                    predictions = model_info["model"].predict(X_processed)
                    probabilities = model_info["model"].predict_proba(X_processed)[:, 1]
                    
                    st.session_state.current_data["CA_Prediction"] = predictions
                    st.session_state.current_data["CA_Probability"] = probabilities
                    
                    if "Student_ID" in st.session_state.current_data.columns and st.session_state.data is not None:
                        historical_data = st.session_state.data[["Student_ID", "Attendance_Percentage", "Academic_Performance", "Suspensions"]]
                        st.session_state.current_data = st.session_state.current_data.merge(
                            historical_data, on="Student_ID", how="left", suffixes=("", "_Historical")
                        )
                    
                    st.subheader("Prediction Results")
                    st.dataframe(st.session_state.current_data)
                    
                    with st.expander("About CA Probability Heatmap"):
                        st.markdown("""
                        **About CA Probability Heatmap**

                        This heatmap shows the average CA probability by Grade and School for predicted students.
                        - **Red colors** indicate higher probabilities of chronic absenteeism.
                        - **Hover** to see exact probabilities.
                        - Use this to identify schools or grades with higher risk for targeted interventions.
                        """)
                    heatmap_data = st.session_state.current_data.groupby(["Grade", "School"])["CA_Probability"].mean().unstack().fillna(0)
                    fig = px.imshow(
                        heatmap_data,
                        title="CA Probability Heatmap by Grade and School",
                        labels={"color": "CA Probability"},
                        color_continuous_scale="Reds",
                        hover_data={"value": True}
                    )
                    fig.update_traces(hovertemplate="Grade: %{y}<br>School: %{x}<br>CA Probability: %{z:.2f}")
                    fig.update_layout(width=600, height=400, margin=dict(l=50, r=50, t=50, b=50))
                    st.plotly_chart(fig, use_container_width=False)
                    
                    with st.expander("About CA Probability Distribution"):
                        st.markdown("""
                        **About CA Probability Distribution**

                        This histogram shows the distribution of CA probabilities across predicted students, colored by CA prediction.
                        - **CA (Red)**: Students predicted as chronically absent.
                        - **NO-CA (Blue)**: Students predicted as not chronically absent.
                        - **Hover** to see the number of students in each probability bin.
                        - Use this to understand the spread of risk levels.
                        """)
                    fig = px.histogram(
                        st.session_state.current_data,
                        x="CA_Probability",
                        color="CA_Prediction",
                        title="CA Probability Distribution",
                        hover_data=["CA_Probability", "CA_Prediction"]
                    )
                    fig.update_traces(hovertemplate="Probability: %{x:.2f}<br>Prediction: %{customdata[1]}<br>Count: %{y}")
                    fig.update_layout(width=600, height=400, margin=dict(l=50, r=50, t=50, b=50))
                    st.plotly_chart(fig, use_container_width=False)
                    
                    with st.expander("About CA Prediction Heatmap"):
                        st.markdown("""
                        **About CA Prediction Heatmap**

                        This heatmap shows the count of CA predictions (CA or NO-CA) by Grade.
                        - **Darker colors** indicate higher counts.
                        - **Hover** to see the exact number of students.
                        - Use this to identify grades with higher rates of predicted absenteeism.
                        """)
                    heatmap_data = st.session_state.current_data.groupby(["Grade", "CA_Prediction"]).size().unstack().fillna(0)
                    fig = px.imshow(
                        heatmap_data,
                        title="CA Prediction Heatmap by Grade",
                        hover_data={"value": True}
                    )
                    fig.update_traces(hovertemplate="Grade: %{y}<br>Prediction: %{x}<br>Count: %{z}")
                    fig.update_layout(width=600, height=400, margin=dict(l=50, r=50, t=50, b=50))
                    st.plotly_chart(fig, use_container_width=False)
                    
                    csv = st.session_state.current_data.to_csv(index=False)
                    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
                except Exception as e:
                    st.error(f"Error running predictions: {str(e)}")
        
        if "CA_Prediction" in st.session_state.current_data.columns:
            with st.expander("Student Prediction Visualizations"):
                dataset_options = ["Current Data"] + list(st.session_state.datasets.keys())
                selected_datasets = st.multiselect("Filter by Dataset", dataset_options, default=["Current Data"])
                
                filtered_data = st.session_state.current_data.copy()
                if selected_datasets != ["Current Data"]:
                    filtered_data = pd.DataFrame()
                    for ds in selected_datasets:
                        if ds == "Current Data":
                            filtered_data = pd.concat([filtered_data, st.session_state.current_data], ignore_index=True)
                        else:
                            ds_data = st.session_state.datasets[ds].copy()
                            if "CA_Prediction" in ds_data.columns:
                                filtered_data = pd.concat([filtered_data, ds_data], ignore_index=True)
                
                if not filtered_data.empty:
                    risk_scores = (100 - filtered_data["Attendance_Percentage"]) * 0.4 + \
                                  (100 - filtered_data["Academic_Performance"]) * 0.3 + \
                                  filtered_data["Suspensions"] * 10
                    
                    with st.expander("About Student Prediction Scatter Plot"):
                        st.markdown("""
                        **About Student Prediction Scatter Plot**

                        This scatter plot visualizes predicted students:
                        - **X-axis**: Attendance Percentage
                        - **Y-axis**: CA Probability
                        - **Size**: Risk Score (based on attendance, academic performance, and suspensions)
                        - **Color**: CA Prediction (Red for CA, Blue for NO-CA)
                        - **Hover**: Shows Student ID, exact values, and prediction.
                        - Use this to identify high-risk students and their characteristics.
                        """)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=filtered_data["Attendance_Percentage"],
                        y=filtered_data["CA_Probability"],
                        mode="markers",
                        marker=dict(
                            size=risk_scores * 0.5,
                            color=np.where(filtered_data["CA_Prediction"] == "CA", "#e74c3c", "#3498db"),
                            opacity=0.6
                        ),
                        text=filtered_data["Student_ID"],
                        hovertemplate="Student ID: %{text}<br>Attendance: %{x:.2f}%<br>CA Probability: %{y:.2f}<br>Risk Score: %{marker.size:.2f}<br>Prediction: %{customdata}",
                        customdata=filtered_data["CA_Prediction"]
                    ))
                    fig.update_layout(
                        title="Student Prediction Scatter Plot",
                        xaxis_title="Attendance Percentage",
                        yaxis_title="CA Probability",
                        showlegend=False,
                        width=600,
                        height=400,
                        margin=dict(l=50, r=50, t=50, b=50)
                    )
                    st.plotly_chart(fig, use_container_width=False)
                    
                    st.subheader("Filtered Students")
                    st.dataframe(filtered_data)
        
        with st.expander("Single Student Analysis"):
            if "CA_Prediction" in st.session_state.current_data.columns:
                student_ids = st.session_state.current_data["Student_ID"].tolist()
                with st.form("student_search_form"):
                    selected_id = st.selectbox(
                        "Select Student ID",
                        student_ids,
                        index=student_ids.index(st.session_state.selected_student_id) if st.session_state.selected_student_id in student_ids else 0,
                        key="student_select"
                    )
                    if st.form_submit_button("Analyze"):
                        st.session_state.selected_student_id = selected_id
                
                if st.session_state.selected_student_id in student_ids:
                    st.markdown('<div class="student-analysis-container">', unsafe_allow_html=True)
                    student_data = st.session_state.current_data[st.session_state.current_data["Student_ID"] == st.session_state.selected_student_id]
                    if not student_data.empty:
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.subheader("Student Profile")
                            st.dataframe(student_data)
                            
                            ca_prob = student_data["CA_Probability"].iloc[0]
                            ca_pred = student_data["CA_Prediction"].iloc[0]
                            st.write(f"**CA Prediction**: {ca_pred}")
                            st.write(f"**CA Probability**: {ca_prob:.2f}")
                            
                            st.subheader("Preventive Actions")
                            actions = []
                            if ca_pred == "CA" or ca_prob > 0.5:
                                actions.append((
                                    "Increase attendance monitoring",
                                    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" class="action-icon"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-13h2v6h-2zm0 8h2v2h-2z"/></svg>'
                                ))
                                actions.append((
                                    "Provide academic support or tutoring",
                                    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" class="action-icon"><path d="M4 4h16v16H4V4zm2 2v12h12V6H6zm2 2h8v2H8V8zm0 4h8v2H8v-2zm0 4h4v2H8v-2z"/></svg>'
                                ))
                                actions.append((
                                    "Engage with parents to address absence causes",
                                    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" class="action-icon"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-7h2v4h-2zm0-6h2v2h-2z"/></svg>'
                                ))
                                if student_data["Suspensions"].iloc[0] > 0:
                                    actions.append((
                                        "Address behavioral issues through counseling",
                                        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" class="action-icon"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-11h2v6h-2zm0 8h2v2h-2z"/></svg>'
                                    ))
                            
                            for action, svg in actions:
                                st.markdown(f'<div class="action-item">{svg} {action}</div>', unsafe_allow_html=True)
                        
                        with col2:
                            attendance = student_data["Attendance_Percentage"].iloc[0]
                            academic = student_data["Academic_Performance"].iloc[0]
                            suspensions = student_data["Suspensions"].iloc[0]
                            risk_score = (100 - attendance) * 0.4 + (100 - academic) * 0.3 + suspensions * 10
                            
                            with st.expander("About Risk Assessment Gauge"):
                                st.markdown("""
                                **About Risk Assessment Gauge**

                                This gauge shows the student's risk score (0-100) based on:
                                - **Attendance**: Lower attendance increases risk (40% weight).
                                - **Academic Performance**: Lower scores increase risk (30% weight).
                                - **Suspensions**: More suspensions increase risk (10% per suspension).
                                - **Green (0-50)**: Low risk.
                                - **Yellow (50-75)**: Moderate risk.
                                - **Red (75-100)**: High risk.
                                - Use this to prioritize interventions for high-risk students.
                                """)
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=risk_score,
                                title={"text": "Risk Assessment Score"},
                                gauge={
                                    "axis": {"range": [0, 100]},
                                    "bar": {"color": "#3498db"},
                                    "steps": [
                                        {"range": [0, 50], "color": "green"},
                                        {"range": [50, 75], "color": "yellow"},
                                        {"range": [75, 100], "color": "red"}
                                    ],
                                    "threshold": {
                                        "line": {"color": "black", "width": 4},
                                        "thickness": 0.75,
                                        "value": 50
                                    }
                                }
                            ))
                            fig.update_layout(width=600, height=400, margin=dict(l=50, r=50, t=50, b=50))
                            st.plotly_chart(fig, use_container_width=False)
                            
                            if risk_score > 50:
                                st.warning("High risk of chronic absenteeism!")
                            
                            with st.expander("About Attendance Trend"):
                                st.markdown("""
                                **About Attendance Trend**

                                This plot shows the student's attendance percentage over time.
                                - **Hover** to see the exact attendance value.
                                - Use this to track changes in attendance behavior.
                                - Note: Current data shows only the latest year; historical data may be included if available.
                                """)
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=[student_data["Year"].iloc[0]],
                                y=[student_data["Attendance_Percentage"].iloc[0]],
                                mode="lines+markers",
                                name="Attendance (%)",
                                hovertemplate="Year: %{x}<br>Attendance: %{y:.2f}%"
                            ))
                            fig.update_layout(
                                title="Attendance Trend",
                                xaxis_title="Year",
                                yaxis_title="Attendance Percentage",
                                width=600,
                                height=400,
                                margin=dict(l=50, r=50, t=50, b=50)
                            )
                            st.plotly_chart(fig, use_container_width=False)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Run predictions to enable single student analysis.")

# Page 4: Documentation
elif st.session_state.page == "📚 Documentation":
    st.markdown("""
    <h1 class="section-header">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#3498db" class="header-icon">
            <path d="M4 3h16a2 2 0 012 2v14a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2zm1 2v14h14V5H5zm2 2h10v2H7V7zm0 4h10v2H7v-2zm0 4h7v2H7v-2z"/>
        </svg>
        Documentation
    </h1>
    """, unsafe_allow_html=True)
    
    if st.session_state.data is not None:
        with st.expander("Patterns & Correlations"):
            high_risk = st.session_state.data[st.session_state.data["CA_Status"] == "CA"]
            if not high_risk.empty:
                st.subheader("Discovered Patterns")
                if st.session_state.patterns:
                    st.markdown("**View Discovered Patterns**")
                    for i, pattern in enumerate(st.session_state.patterns):
                        col1, col2, col3 = st.columns([4, 1, 1])
                        with col1:
                            st.write(f"- {pattern['pattern']}: {pattern['explanation']}")
                        with col2:
                            if st.button("Edit", key=f"edit_pattern_{i}"):
                                st.session_state.edit_pattern_index = i
                        with col3:
                            if st.button("Delete", key=f"delete_pattern_{i}"):
                                st.session_state.patterns.pop(i)
                                st.rerun()
                
                if 'edit_pattern_index' in st.session_state:
                    st.subheader("Edit Pattern")
                    idx = st.session_state.edit_pattern_index
                    new_pattern = st.text_input("Pattern", value=st.session_state.patterns[idx]["pattern"])
                    new_explanation = st.text_area("Explanation", value=st.session_state.patterns[idx]["explanation"])
                    if st.button("Save Changes"):
                        st.session_state.patterns[idx] = {"pattern": new_pattern, "explanation": new_explanation}
                        del st.session_state.edit_pattern_index
                        st.rerun()
            
            st.subheader("How to Add New Patterns")
            st.markdown("""
                **How to Add New Patterns**

                Patterns summarize trends in high-risk students to inform interventions. Use the Group Analysis section to identify trends in the data, such as low attendance or specific demographics.

                **Examples** (based on dataset columns):
                - "Average Attendance < 80%": Indicates high-risk students have low attendance.
                - "Grade 6 with Bus Transportation": Common among high-risk students in Grade 6 using buses.
                - "High Suspensions (>2)": Students with multiple suspensions are at risk.

                **Steps**:
                1. Analyze data in "Group Analysis & Comparisons" to find trends (e.g., average attendance by grade).
                2. Write a concise pattern summarizing the trend.
                3. Add an explanation linking the pattern to chronic absenteeism risk.
                4. Submit to save the pattern for future reference.
                """)
            
            st.subheader("Add New Pattern")
            new_pattern = st.text_input("New Pattern")
            new_explanation = st.text_area("Pattern Explanation")
            if st.button("Add Pattern"):
                if new_pattern and new_explanation:
                    st.session_state.patterns.append({"pattern": new_pattern, "explanation": new_explanation})
                    st.success("Pattern added successfully!")
                    st.rerun()
            
            st.subheader("Correlation Visualizations")
            with st.expander("About Correlation Heatmap"):
                st.markdown("""
                **About Correlation Heatmap**

                This heatmap shows correlations between numerical features (e.g., Attendance, Academic Performance).
                - **Values range from -1 to 1**: Positive (blue) means features increase together; negative (red) means they move oppositely.
                - **Hover** to see exact correlation values.
                - Use this to identify relationships, like low attendance correlating with high suspensions.
                """)
            numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns
            corr = st.session_state.data[numeric_cols].corr()
            fig = px.imshow(
                corr,
                title="Correlation Heatmap of Features",
                labels={"color": "Correlation"},
                hover_data={"value": True}
            )
            fig.update_traces(hovertemplate="Feature X: %{x}<br>Feature Y: %{y}<br>Correlation: %{z:.2f}")
            fig.update_layout(width=600, height=400, margin=dict(l=50, r=50, t=50, b=50))
            st.plotly_chart(fig, use_container_width=False)
            
            with st.expander("About Attendance Correlation Bar Plot"):
                st.markdown("""
                **About Attendance Correlation Bar Plot**

                This bar plot shows how attendance correlates with other numerical features.
                - **Positive bars (blue)**: Higher feature values linked to higher attendance.
                - **Negative bars (red)**: Higher feature values linked to lower attendance.
                - **Hover** to see exact correlation coefficients.
                - Use this to identify factors influencing attendance, like suspensions.
                """)
            corr_attendance = corr["Attendance_Percentage"].drop("Attendance_Percentage")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=corr_attendance.index,
                y=corr_attendance.values,
                marker_color=np.where(corr_attendance > 0, "#3498db", "#e74c3c"),
                hovertemplate="Feature: %{x}<br>Correlation: %{y:.2f}"
            ))
            fig.update_layout(
                title="Correlation of Attendance with Other Factors",
                xaxis_title="Feature",
                yaxis_title="Correlation Coefficient",
                width=600,
                height=400,
                margin=dict(l=50, r=50, t=50, b=50)
            )
            st.plotly_chart(fig, use_container_width=False)
            
            with st.expander("About Attendance vs. Suspensions Scatter Plot"):
                st.markdown("""
                **About Attendance vs. Suspensions Scatter Plot**

                This scatter plot shows attendance percentage vs. suspensions, colored by CA Status.
                - **Red (CA)**: Chronically absent students.
                - **Blue (NO-CA)**: Non-chronically absent students.
                - **Hover** to see exact values and student counts.
                - Use this to explore how suspensions impact attendance.
                """)
            fig = px.scatter(
                st.session_state.data,
                x="Suspensions",
                y="Attendance_Percentage",
                color="CA_Status",
                title="Attendance vs. Suspensions",
                hover_data=["Suspensions", "Attendance_Percentage", "CA_Status"]
            )
            fig.update_traces(hovertemplate="Suspensions: %{customdata[0]}<br>Attendance: %{customdata[1]}%<br>CA Status: %{customdata[2]}")
            fig.update_layout(width=600, height=400, margin=dict(l=50, r=50, t=50, b=50))
            st.plotly_chart(fig, use_container_width=False)
        
        with st.expander("AI-Powered Pattern Recognition"):
            if st.session_state.models:
                displayed = False
                for model_name in ["Random Forest", "Decision Tree", "Gradient Boosting"]:
                    if model_name in st.session_state.models:
                        model_info = st.session_state.models[model_name]
                        fig = plot_feature_importance(model_info["model"], model_info["feature_names"])
                        if fig:
                            with st.expander(f"About Feature Importance Plot ({model_name})"):
                                st.markdown(f"""
                                **About Feature Importance Plot ({model_name})**

                                This bar plot shows which features most influence {model_name}'s predictions.
                                - **Higher bars**: More important features.
                                - **Hover** to see exact importance scores.
                                - Use this to identify key risk factors, like low attendance or high suspensions.
                                """)
                            st.write(f"Key factors influencing absenteeism (based on {model_name} feature importance):")
                            fig.update_traces(hovertemplate="Feature: %{x}<br>Importance: %{y:.4f}")
                            fig.update_layout(width=600, height=400, margin=dict(l=50, r=50, t=50, b=50))
                            st.plotly_chart(fig, use_container_width=False)
                            displayed = True
                            break
                if not displayed:
                    st.write("No feature importance available. Train a Random Forest, Decision Tree, or Gradient Boosting model to view key factors.")
            else:
                st.info("Train models to enable AI-powered pattern recognition.")
        
        with st.expander("Group Analysis & Comparisons"):
            st.subheader("Group Analysis & Comparisons")
            with st.expander("About Group Analysis & Comparisons"):
                st.markdown("""
                **About Group Analysis & Comparisons**

                This section allows you to filter students by attributes (e.g., Grade, Gender, School) and visualize attendance trends.
                - **Purpose**: Identify at-risk groups, such as grades with low attendance or schools with high absenteeism.
                - **Insights**:
                  - Compare average attendance across groups to spot trends.
                  - Use violin plots to see attendance variability within groups.
                  - Filter to focus on specific demographics or conditions.
                - **Use Case**: Target interventions, like tutoring for low-performing grades or transportation support for specific schools.
                """)
            
            st.write("**Filter Students**")
            col1, col2, col3 = st.columns(3)
            with col1:
                filter_grade = st.multiselect("Filter by Grade", sorted(st.session_state.data["Grade"].unique()), key="filter_grade")
                filter_gender = st.multiselect("Filter by Gender", st.session_state.data["Gender"].unique(), key="filter_gender")
            with col2:
                filter_school = st.multiselect("Filter by School", st.session_state.data["School"].unique(), key="filter_school")
                filter_meal = st.multiselect("Filter by Meal Code", st.session_state.data["Meal_Code"].unique(), key="filter_meal")
            with col3:
                filter_transport = st.multiselect("Filter by Transportation", st.session_state.data["Transportation"].unique(), key="filter_transport")
            
            filtered_data = st.session_state.data
            if filter_grade:
                filtered_data = filtered_data[filtered_data["Grade"].isin(filter_grade)]
            if filter_gender:
                filtered_data = filtered_data[filtered_data["Gender"].isin(filter_gender)]
            if filter_school:
                filtered_data = filtered_data[filtered_data["School"].isin(filter_school)]
            if filter_meal:
                filtered_data = filtered_data[filtered_data["Meal_Code"].isin(filter_meal)]
            if filter_transport:
                filtered_data = filtered_data[filtered_data["Transportation"].isin(filter_transport)]
            
            if not filtered_data.empty:
                group_by = st.selectbox("Group By", ["Grade", "Gender", "School", "Meal_Code", "Transportation"])
                trend_data = filtered_data.groupby(group_by)["Attendance_Percentage"].mean().reset_index()
                
                with st.expander("About Average Attendance Bar Plot"):
                    st.markdown("""
                    **About Average Attendance Bar Plot**

                    This bar plot shows the average attendance percentage for each group (e.g., by Grade).
                    - **Hover** to see exact attendance values.
                    - Use this to identify groups with lower attendance for targeted interventions.
                    """)
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=trend_data[group_by],
                    y=trend_data["Attendance_Percentage"],
                    marker_color="#3498db",
                    hovertemplate=f"{group_by}: %{{x}}<br>Attendance: %{{y:.2f}}%"
                ))
                fig.update_layout(
                    title=f"Average Attendance by {group_by}",
                    xaxis_title=group_by,
                    yaxis_title="Attendance Percentage",
                    width=600,
                    height=400,
                    margin=dict(l=50, r=50, t=50, b=50)
                )
                st.plotly_chart(fig, use_container_width=False)
                
                with st.expander("About Attendance Distribution Violin Plot"):
                    st.markdown("""
                    **About Attendance Distribution Violin Plot**

                    This violin plot shows the distribution of attendance percentages within each group.
                    - **Wider sections**: Higher density of students at that attendance level.
                    - **Hover** to see summary statistics.
                    - Use this to understand attendance variability and identify outliers.
                    """)
                fig = go.Figure()
                for group in filtered_data[group_by].unique():
                    group_data = filtered_data[filtered_data[group_by] == group]["Attendance_Percentage"]
                    fig.add_trace(go.Violin(
                        x=[group] * len(group_data),
                        y=group_data,
                        name=str(group),
                        box_visible=True,
                        meanline_visible=True,
                        hovertemplate=f"{group_by}: {group}<br>Attendance: %{{y:.2f}}%"
                    ))
                fig.update_layout(
                    title=f"Attendance Distribution by {group_by}",
                    xaxis_title=group_by,
                    yaxis_title="Attendance Percentage",
                    width=600,
                    height=400,
                    margin=dict(l=50, r=50, t=50, b=50)
                )
                st.plotly_chart(fig, use_container_width=False)
        
        with st.expander("Dataset Comparison"):
            st.subheader("Dataset Comparison Heatmap")
            st.markdown("""
            **About Dataset Comparison Heatmap**

            This heatmap compares a selected metric (e.g., average attendance) across datasets by group (e.g., Grade, School).
            - **Normalized values (0-1)**: Ensures fair comparison across datasets.
            - **Blue gradient**: Darker blues indicate higher values.
            - **Filters**: Select grouping variables and metrics to customize the view.
            - **Hover**: Shows exact values and dataset details.
            - Use this to identify trends or differences across historical and current datasets.
            """)
            
            if st.session_state.datasets or st.session_state.current_data is not None:
                group_vars = ["Grade", "School", "Gender", "Meal_Code", "Transportation"]
                metric_vars = ["Attendance_Percentage", "Academic_Performance", "Suspensions"]
                
                group_var = st.selectbox("Group By", group_vars)
                metric = st.selectbox("Metric", metric_vars)
                
                heatmap_data = []
                dataset_names = list(st.session_state.datasets.keys())
                if st.session_state.current_data is not None:
                    dataset_names.append("Current Data")
                
                for ds_name in dataset_names:
                    ds_data = st.session_state.datasets.get(ds_name, st.session_state.current_data)
                    if ds_data is not None and group_var in ds_data.columns and metric in ds_data.columns:
                        group_means = ds_data.groupby(group_var)[metric].mean().reset_index()
                        group_means["Dataset"] = ds_name
                        heatmap_data.append(group_means)
                
                if heatmap_data:
                    heatmap_df = pd.concat(heatmap_data, ignore_index=True)
                    heatmap_pivot = heatmap_df.pivot(index=group_var, columns="Dataset", values=metric)
                    # Normalize data for consistency
                    heatmap_pivot = (heatmap_pivot - heatmap_pivot.min().min()) / (heatmap_pivot.max().max() - heatmap_pivot.min().min())
                    heatmap_pivot = heatmap_pivot.fillna(0)
                    
                    fig = px.imshow(
                        heatmap_pivot,
                        title=f"{metric} by {group_var} Across Datasets",
                        labels={"color": f"Normalized {metric}"},
                        color_continuous_scale="Blues",
                        hover_data={"value": True}
                    )
                    fig.update_traces(hovertemplate=f"{group_var}: %{{y}}<br>Dataset: %{{x}}<br>{metric}: %{{z:.2f}}")
                    fig.update_layout(width=600, height=400, margin=dict(l=50, r=50, t=50, b=50))
                    st.plotly_chart(fig, use_container_width=False)
                else:
                    st.warning("No data available for the selected group or metric.")
            else:
                st.info("Generate or upload data to enable dataset comparison.")
