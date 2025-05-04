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
import seaborn as sns
import matplotlib.pyplot as plt
import uuid
import os
import tempfile
from data_generator import generate_historical_data, generate_current_year_data
from model_utils import train_model, tune_model, get_model_explanation, plot_confusion_matrix, plot_feature_importance
import plotly.io as pio

# Set page config
st.set_page_config(page_title="Chronic Absenteeism Prediction", layout="wide")

# Load CSS for styling
st.markdown('<style>' + open('styles.css').read() + '</style>', unsafe_allow_html=True)

# Function to clear session state
def clear_session_state():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.data = None
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
if 'selected_group_by' not in st.session_state:
    st.session_state.selected_group_by = None

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
    st.experimental_rerun()

# Page 1: Data Configuration
if st.session_state.page == "📝 Data Configuration":
    st.markdown("""
    <h1>
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#3498db" style="width: 30px; height: 30px; vertical-align: middle; margin-right: 10px;">
            <path d="M3 3h18v18H3V3zm2 2v14h14V5H5zm2 2h10v2H7V7zm0 4h10v2H7v-2zm0 4h7v2H7v-2z"/>
        </svg>
        📝 Data Configuration
    </h1>
    """, unsafe_allow_html=True)
    
    st.header("Generate Historical Data")
    num_students = st.slider("Number of Students", 100, 5000, 1000)
    year_start, year_end = st.slider("Academic Years", 2020, 2025, (2020, 2024), step=1)
    school_prefix = st.text_input("School Prefix (e.g., 10U)", "10U001")
    num_schools = st.number_input("Number of Schools", 1, 50,5)
    id_length = st.radio("Student ID Length", [5, 7], index=0)
    dropoff_percent = st.slider("Target Drop Off Percentage (%)", 0, 50, 2, step=5, help="Percentage of students with CA Status = CA")
    
    grades = st.multiselect("Grades", list(range(1, 13)), default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    
    st.subheader("Gender Distribution (%)")
    male_dist = st.slider("Male (%)", 0, 100, 40, step=5)
    female_dist = st.slider("Female (%)", 0, 100, 40, step=5)
    other_dist = st.slider("Other (%)", 0, 100, 20, step=5)
    
    total_dist = male_dist + female_dist + other_dist
    if total_dist != 100:
        st.error(f"Gender distribution must sum to 100%. Current total: {total_dist}%")
        gender_dist = None
    else:
        gender_dist = [male_dist, female_dist, other_dist]
    
    meal_codes = st.multiselect("Meal Codes", ["Free", "Reduced", "Paid"], default=["Free", "Reduced", "Paid"])
    academic_perf = st.slider("Academic Performance Range (%)", 1, 100, (40, 90))
    
    transportation = st.multiselect("Transportation Options", ["Bus", "Walk", "Car"], default=["Bus", "Walk"])
    suspensions_range = st.slider("Suspensions Range (per year)", 0, 10, (0, 3))
    
    st.subheader("Attendance Data")
    total_days = 180
    st.write(f"Total School Days: {total_days}")
    present_days_range = st.slider("Present Days Range", 0, total_days, (100, total_days))
    
    max_absent_days = total_days - present_days_range[0]
    if max_absent_days <= 0:
        st.error(f"Error: Minimum present days ({present_days_range[0]}) equals or exceeds total days ({total_days}). Reduce minimum present days.")
        max_absent_days = 1
    absent_days_range = st.slider(
        "Absent Days Range",
        0,
        max_absent_days,
        (0, min(80, max_absent_days)),
        help=f"Maximum absent days cannot exceed {max_absent_days}."
    )
    
    attendance_valid = True
    if max_absent_days <= 0:
        st.error(f"Error: Minimum present days ({present_days_range[0]}) equals or exceeds total days ({total_days}).")
        attendance_valid = False
    elif present_days_range[0] >= present_days_range[1]:
        st.error(f"Error: Minimum present days ({present_days_range[0]}) must be less than maximum present days ({present_days_range[1]}).")
        attendance_valid = False
    elif absent_days_range[0] >= absent_days_range[1]:
        st.error(f"Error: Minimum absent days ({absent_days_range[0]}) must be less than maximum absent days ({absent_days_range[1]}).")
        attendance_valid = False
    elif present_days_range[0] + absent_days_range[1] > total_days:
        st.error(f"Error: Minimum present days ({present_days_range[0]}) plus maximum absent days ({absent_days_range[1]}) exceeds total days ({total_days}).")
        attendance_valid = False
    elif present_days_range[1] + absent_days_range[0] < total_days:
        st.error(f"Error: Maximum present days ({present_days_range[1]}) plus minimum absent days ({absent_days_range[0]}) is less than total days ({total_days}).")
        attendance_valid = False
    
    st.subheader("Custom Fields")
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
                st.experimental_rerun()
    
    if st.button("Generate Historical Data", disabled=not (gender_dist is not None and attendance_valid)):
        try:
            st.session_state.patterns = []
            custom_fields = [(f["name"], f["values"]) for f in st.session_state.custom_fields if f["name"] and f["values"]]
            data = generate_historical_data(
                num_students, year_start, year_end, school_prefix, num_schools,
                grades, gender_dist, meal_codes, academic_perf, transportation,
                suspensions_range, present_days_range, absent_days_range, total_days,
                custom_fields, id_length, dropoff_percent
            )
            st.session_state.data = data
            st.success("Data generated successfully!")
            
            st.subheader("Data Preview")
            st.dataframe(data.head(10))
            
            csv = data.to_csv(index=False)
            st.download_button("Download Historical Data", csv, "historical_data.csv", "text/csv")
        except Exception as e:
            st.error(f"Error generating data: {str(e)}")

# Page 2: Model Training
elif st.session_state.page == "🤖 Model Training":
    st.markdown("""
    <h1>
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#3498db" style="width: 30px; height: 30px; vertical-align: middle; margin-right: 10px;">
            <path d="M12 2a10 10 0 00-8 4v2h2v2H4v2h2v2H4v2h2v2h2a10 10 0 008-4 10 10 0 008 4h2v-2h-2v-2h2v-2h-2v-2h2V8h-2V6a10 10 0 00-8-4zm0 2a8 8 0 016.32 3H17v2h-2v2h2v2h-2v2h2v2h-1.32A8 8 0 0112 20a8 8 0 01-6.32-3H7v-2H5v-2h2v-2H5V9h2V7h1.32A8 8 0 0112 4z"/>
        </svg>
        🤖 Model Training
    </h1>
    """, unsafe_allow_html=True)
    
    st.header("Load Data")
    data_source = st.radio("Data Source", ["Use Generated Data", "Upload CSV"])
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload Historical Data (CSV)", type=["csv"])
        if uploaded_file:
            try:
                st.session_state.data = pd.read_csv(uploaded_file)
                st.session_state.patterns = []
                st.success("Data uploaded successfully!")
            except Exception as e:
                st.error(f"Error uploading data: {str(e)}")
    
    if st.session_state.data is not None:
        st.subheader("Feature Selection")
        excluded_features = ["Student_ID", "CA_Status"]
        available_features = [col for col in st.session_state.data.columns if col not in excluded_features]
        feature_toggles = {}
        st.write("Toggle Features:")
        for feature in available_features:
            feature_toggles[feature] = st.checkbox(feature, value=True, key=f"feature_{feature}")
        features = [f for f, enabled in feature_toggles.items() if enabled]
        target = "CA_Status"
        
        categorical_cols = [col for col in features if st.session_state.data[col].dtype == "object"]
        numerical_cols = [col for col in features if col not in categorical_cols]
        
        st.subheader("Model Selection")
        with st.expander("Model Selection Guide", expanded=False):
            st.markdown("""
            **Model Selection Guide**

            - **Logistic Regression**: Best for interpretable results.
              - [Learn More](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
            - **Random Forest**: Robust to overfitting.
              - [Learn More](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)
            - **Decision Tree**: Easy to interpret but may overfit.
              - [Learn More](https://scikit-learn.org/stable/modules/tree.html)
            - **SVM**: Effective for non-linear data.
              - [Learn More](https://scikit-learn.org/stable/modules/svm.html)
            - **Gradient Boosting**: Powerful for predictive accuracy.
              - [Learn More](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)
            - **Neural Network**: Suitable for large datasets.
              - [Learn More](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)
            """)
        
        models_to_train = st.multiselect("Select Models", [
            "Logistic Regression", "Random Forest", "Decision Tree",
            "SVM", "Gradient Boosting", "Neural Network"
        ], default=["Logistic Regression", "Random Forest"], key="model_select")
        
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
                                model, metrics, best_params = tune_model(model_name, X_train_processed, y_train, X_test_processed, y_test, tuning_params[model_name])
                            else:
                                model, metrics = train_model(model_name, X_train_processed, y_train, X_test_processed, y_test)
                                best_params = None
                            
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
                                "y_pred": metrics["y_pred"],
                                "X_test_processed": X_test_processed
                            })
                            
                            st.session_state.models[model_name] = {
                                "model": model,
                                "metrics": metrics,
                                "preprocessor": preprocessor,
                                "feature_names": feature_names,
                                "best_params": best_params,
                                "y_test": y_test,
                                "y_pred": metrics["y_pred"],
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
                    
                    high_risk = st.session_state.data[st.session_state.data["CA_Status"] == "CA"]
                    if not high_risk.empty:
                        patterns = []
                        low_attendance = f"Average Attendance: {high_risk['Attendance_Percentage'].mean():.2f}%"
                        common_grades = f"Common Grades: {', '.join(map(str, high_risk['Grade'].mode().tolist()))}"
                        common_meal_codes = f"Common Meal Codes: {', '.join(high_risk['Meal_Code'].mode().tolist())}"
                        common_transport = f"Common Transportation: {', '.join(high_risk['Transportation'].mode().tolist())}"
                        
                        existing_patterns = [p["pattern"] for p in st.session_state.patterns]
                        for pattern in [low_attendance, common_grades, common_meal_codes, common_transport]:
                            if pattern not in existing_patterns:
                                patterns.append({"pattern": pattern, "explanation": "Identified in high-risk students (CA Status = CA)"})
                        
                        st.session_state.patterns.extend(patterns)
                    
                    st.success("All models trained successfully!")
                    st.balloons()
            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
        
        if st.session_state.models:
            st.subheader("Model Results")
            for model_name in models_to_train:
                if model_name in st.session_state.models:
                    model_info = st.session_state.models[model_name]
                    metrics = model_info["metrics"]
                    
                    st.write(f"**{model_name} Results**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        <div class="model-icon">
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#3498db"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-13h2v6h-2zm0 8h2v2h-2z"/></svg>
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
                        st.plotly_chart(fig)
                    
                    if model_name in ["Random Forest", "Decision Tree", "Gradient Boosting"]:
                        fig = plot_feature_importance(model_info["model"], model_info["feature_names"])
                        if fig:
                            st.plotly_chart(fig)
                    
                    st.write(get_model_explanation(model_name, model_info["X_test_processed"][:1], model_info["model"]))
        
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
                            st.plotly_chart(fig, use_container_width=False)
                    else:
                        st.warning("Please select at least one model to compare.")
        
        if st.session_state.models:
            st.subheader("Pattern Discovery")
            st.write(f"Number of patterns learned: {len(st.session_state.patterns)}")
            if st.session_state.patterns:
                for pattern in st.session_state.patterns:
                    st.write(f"- {pattern['pattern']}: {pattern['explanation']}")
            else:
                st.info("No patterns identified yet.")
        else:
            st.info("Train models to enable pattern discovery.")

# Page 3: Results
elif st.session_state.page == "📊 Results":
    st.markdown("""
    <h1>
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#3498db" style="width: 30px; height: 30px; vertical-align: middle; margin-right: 10px;">
            <path d="M3 3h18v18H3V3zm2 2v14h14V5H5zm2 2h2v6H7V7zm4 0h2v10h-2V7zm4 0h2v4h-2V7z"/>
        </svg>
        📊 Results
    </h1>
    """, unsafe_allow_html=True)
    
    st.header("Generate Current Year Data")
    data_source = st.radio("Data Source", ["Generate Data", "Upload CSV"])
    
    if data_source == "Generate Data":
        num_students = st.slider("Number of Students", 100, 5000, 1000)
        school_prefix = st.text_input("School Prefix (e.g., CU)", "CU")
        num_schools = st.number_input("Number of Schools", 1, 10, 3)
        id_length = st.radio("Student ID Length", [5, 7], index=0)
        dropoff_percent = st.slider("Target Drop Off Percentage (%)", 5, 50, 20, step=5, help="Percentage of students with CA Status = CA")
        
        grades = st.multiselect("Grades", list(range(1, 13)), default=[1, 2, 3, 4, 5], key="current_grades")
        
        st.subheader("Gender Distribution (%)")
        male_dist = st.slider("Male (%)", 0, 100, 40, step=5, key="current_male")
        female_dist = st.slider("Female (%)", 0, 100, 40, step=5, key="current_female")
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
        
        st.subheader("Attendance Data")
        total_days = 180
        st.write(f"Total School Days: {total_days}")
        present_days_range = st.slider("Present Days Range", 0, total_days, (100, total_days), key="current_present")
        
        max_absent_days = total_days - present_days_range[0]
        if max_absent_days <= 0:
            st.error(f"Error: Minimum present days ({present_days_range[0]}) equals or exceeds total days ({total_days}).")
            max_absent_days = 1
        absent_days_range = st.slider(
            "Absent Days Range",
            0,
            max_absent_days,
            (0, min(80, max_absent_days)),
            key="current_absent",
            help=f"Maximum absent days cannot exceed {max_absent_days}."
        )
        
        attendance_valid = True
        if max_absent_days <= 0:
            st.error(f"Error: Minimum present days ({present_days_range[0]}) equals or exceeds total days ({total_days}).")
            attendance_valid = False
        elif present_days_range[0] >= present_days_range[1]:
            st.error(f"Error: Minimum present days ({present_days_range[0]}) must be less than maximum present days ({present_days_range[1]}).")
            attendance_valid = False
        elif absent_days_range[0] >= absent_days_range[1]:
            st.error(f"Error: Minimum absent days ({absent_days_range[0]}) must be less than maximum absent days ({absent_days_range[1]}).")
            attendance_valid = False
        elif present_days_range[0] + absent_days_range[1] > total_days:
            st.error(f"Error: Minimum present days ({present_days_range[0]}) plus maximum absent days ({absent_days_range[1]}) exceeds total days ({total_days}).")
            attendance_valid = False
        elif present_days_range[1] + absent_days_range[0] < total_days:
            st.error(f"Error: Maximum present days ({present_days_range[1]}) plus minimum absent days ({absent_days_range[0]}) is less than total days ({total_days}).")
            attendance_valid = False
        
        use_historical_ids = st.checkbox("Use Historical Student IDs", value=False, disabled=st.session_state.data is None)
        include_graduates = st.checkbox("Include Graduating Students (Cap at Grade 12)", value=False, disabled=not use_historical_ids)
        
        if use_historical_ids and st.session_state.data is not None:
            unique_historical_students = len(st.session_state.data["Student_ID"].unique())
            if unique_historical_students > num_students:
                st.warning(f"Note: Historical data contains {unique_historical_students} unique students, but only {num_students} will be included in current-year data.")
        
        st.subheader("Custom Fields")
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
                    st.experimental_rerun()
        
        if st.button("Generate Current Year Data", disabled=not (gender_dist is not None and attendance_valid)):
            try:
                custom_fields = [(f["name"], f["values"]) for f in st.session_state.current_custom_fields if f["name"] and f["values"]]
                historical_data = st.session_state.data if use_historical_ids and st.session_state.data is not None else None
                st.session_state.current_data = generate_current_year_data(
                    num_students, school_prefix, num_schools, grades, gender_dist,
                    meal_codes, academic_perf, transportation, suspensions_range,
                    present_days_range, absent_days_range, total_days, custom_fields,
                    historical_ids=historical_data, id_length=id_length, dropoff_percent=dropoff_percent,
                    include_graduates=include_graduates
                )
                st.success("Current year data generated successfully!")
                st.subheader("Data Preview")
                st.dataframe(st.session_state.current_data.head(10))
                csv = st.session_state.current_data.to_csv(index=False)
                st.download_button(
                    label="Download Current Year Data",
                    data=csv,
                    file_name="current_year_data.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error generating data: {str(e)}")
    else:
        uploaded_file = st.file_uploader("Upload Current Year Data (CSV)", type=["csv"])
        if uploaded_file:
            try:
                st.session_state.current_data = pd.read_csv(uploaded_file)
                st.success("Data uploaded successfully!")
                st.subheader("Data Preview")
                st.dataframe(st.session_state.current_data.head(10))
            except Exception as e:
                st.error(f"Error uploading data: {str(e)}")
    
    if st.session_state.current_data is not None and st.session_state.models:
        st.subheader("Run Predictions")
        selected_model = st.selectbox("Select Model", list(st.session_state.models.keys()))
        excluded_columns = ["Student_ID", "CA_Prediction", "CA_Probability", "Drop_Off_Percent"]
        available_features = [col for col in st.session_state.current_data.columns if col not in excluded_columns]
        feature_toggles = {}
        st.write("Toggle Features:")
        for feature in available_features:
            feature_toggles[feature] = st.checkbox(feature, value=True, key=f"predict_feature_{feature}")
        features = [f for f, enabled in feature_toggles.items() if enabled]
        
        group_by_options = [col for col in available_features if st.session_state.current_data[col].dtype == "object" or col == "Grade"]
        group_by_options += [f["name"] for f in st.session_state.current_custom_fields if f["name"] in st.session_state.current_data.columns]
        group_by_feature = st.selectbox(
            "Group Drop Off % By",
            group_by_options,
            index=group_by_options.index(st.session_state.selected_group_by) if st.session_state.selected_group_by in group_by_options else 0,
            help="Select a feature to visualize the percentage of students predicted as chronically absent (CA) for each group. This uses existing predictions and does not require re-running the model."
        )
        st.session_state.selected_group_by = group_by_feature
        
        if st.button("Predict"):
            try:
                model_info = st.session_state.models[selected_model]
                training_features = model_info["feature_names"]
                missing_features = [f for f in features if f not in st.session_state.current_data.columns]
                if missing_features:
                    raise ValueError(f"Selected features not found in current data: {missing_features}")
                
                X = st.session_state.current_data[features]
                X_processed = model_info["preprocessor"].transform(X)
                predictions = model_info["model"].predict(X_processed)
                probabilities = model_info["model"].predict_proba(X_processed)[:, 1]
                
                prediction_data = st.session_state.current_data.copy()
                prediction_data["CA_Prediction"] = predictions
                prediction_data["CA_Probability"] = probabilities
                
                # Calculate Drop Off % for the selected group
                if group_by_feature:
                    dropoff_data = prediction_data.groupby(group_by_feature)["CA_Prediction"].value_counts(normalize=True).unstack().fillna(0)
                    if "CA" in dropoff_data.columns:
                        dropoff_data["Drop_Off_Percent"] = dropoff_data["CA"] * 100
                        dropoff_data = dropoff_data[["Drop_Off_Percent"]].reset_index()
                        prediction_data = prediction_data.merge(
                            dropoff_data[[group_by_feature, "Drop_Off_Percent"]],
                            on=group_by_feature,
                            how="left"
                        )
                    else:
                        prediction_data["Drop_Off_Percent"] = 0.0
                
                if use_historical_ids and "Student_ID" in prediction_data.columns and st.session_state.data is not None:
                    if "Student_ID" not in st.session_state.data.columns:
                        raise ValueError("Student_ID column missing in historical data")
                    historical_data = st.session_state.data[["Student_ID", "Year", "Grade", "Attendance_Percentage", "Academic_Performance", "Suspensions"]]
                    prediction_data = prediction_data.merge(
                        historical_data, on="Student_ID", how="left", suffixes=("", "_Historical")
                    )
                
                st.session_state.current_data = prediction_data
                
                st.subheader("Prediction Results")
                st.dataframe(st.session_state.current_data)
                
                heatmap_data = st.session_state.current_data.groupby(["Grade", "School"])["CA_Probability"].mean().unstack().fillna(0)
                fig = px.imshow(
                    heatmap_data,
                    title="CA Probability Heatmap by Grade and School (High Risk in Red)",
                    labels={"color": "CA Probability"},
                    color_continuous_scale="Reds"
                )
                fig.update_traces(hovertemplate="Grade: %{y}<br>School: %{x}<br>CA Probability: %{z:.2f}")
                st.plotly_chart(fig)
                
                fig = px.histogram(
                    st.session_state.current_data,
                    x="CA_Probability",
                    color="CA_Prediction",
                    title="CA Probability Distribution",
                    hover_data=["CA_Probability", "CA_Prediction"]
                )
                fig.update_traces(hovertemplate="Probability: %{x:.2f}<br>Prediction: %{customdata[1]}<br>Count: %{y}")
                st.plotly_chart(fig)
                
                heatmap_data = st.session_state.current_data.groupby(["Grade", "CA_Prediction"]).size().unstack().fillna(0)
                fig = px.imshow(
                    heatmap_data,
                    title="CA Prediction Heatmap by Grade",
                    labels={"color": "Count"}
                )
                fig.update_traces(hovertemplate="Grade: %{y}<br>Prediction: %{x}<br>Count: %{z}")
                st.plotly_chart(fig)
                
                if group_by_feature:
                    dropoff_data = st.session_state.current_data.groupby(group_by_feature)["CA_Prediction"].value_counts(normalize=True).unstack().fillna(0)
                    if "CA" in dropoff_data.columns:
                        dropoff_data["Drop_Off_Percent"] = dropoff_data["CA"] * 100
                        dropoff_data = dropoff_data[["Drop_Off_Percent"]].reset_index()
                        fig = px.pie(
                            dropoff_data,
                            values="Drop_Off_Percent",
                            names=group_by_feature,
                            title=f"Drop Off Percentage by {group_by_feature}",
                            hole=0.4,  # Make it a donut chart
                            color_discrete_sequence=px.colors.sequential.Reds
                        )
                        fig.update_traces(
                            hovertemplate=f"{group_by_feature}: %{{label}}<br>Drop Off %: %{{value:.2f}}%",
                            textinfo="percent+label"
                        )
                        st.plotly_chart(fig)
                    else:
                        st.warning(f"No students predicted as CA for grouping by {group_by_feature}.")
                
                csv = st.session_state.current_data.to_csv(index=False)
                st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Error running predictions: {str(e)}")
        
        st.subheader("Single Student Analysis")
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
                student_data = st.session_state.current_data[st.session_state.current_data["Student_ID"] == st.session_state.selected_student_id]
                if not student_data.empty:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write("**Student Profile**")
                        st.dataframe(student_data)
                        
                        ca_prob = student_data["CA_Probability"].iloc[0]
                        ca_pred = student_data["CA_Prediction"].iloc[0]
                        st.write(f"**CA Prediction**: {ca_pred}")
                        st.write(f"**CA Probability**: {ca_prob:.2f}")
                        
                        st.write("**Preventive Actions**")
                        actions = []
                        if ca_pred == "CA" or ca_prob > 0.5:
                            actions.append((
                                "Increase attendance monitoring",
                                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" class="action-icon" fill="#3498db"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-13h2v6h-2zm0 8h2v2h-2z"/></svg>'
                            ))
                            actions.append((
                                "Provide academic support or tutoring",
                                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" class="action-icon" fill="#3498db"><path d="M4 4h16v16H4V4zm2 2v12h12V6H6zm2 2h8v2H8V8zm0 4h8v2H8v-2zm0 4h4v2H8v-2z"/></svg>'
                            ))
                            actions.append((
                                "Engage with parents to address absence causes",
                                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" class="action-icon" fill="#3498db"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-7h2v4h-2zm0-6h2v2h-2z"/></svg>'
                            ))
                            if student_data["Suspensions"].iloc[0] > 0:
                                actions.append((
                                    "Address behavioral issues through counseling",
                                    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" class="action-icon" fill="#3498db"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-11h2v6h-2zm0 8h2v2h-2z"/></svg>'
                                ))
                        
                        for action, svg in actions:
                            st.markdown(f'<div class="action-item">{svg} {action}</div>', unsafe_allow_html=True)
                    
                    with col2:
                        attendance = student_data["Attendance_Percentage"].iloc[0]
                        academic = student_data["Academic_Performance"].iloc[0]
                        suspensions = student_data["Suspensions"].iloc[0]
                        risk_score = (100 - attendance) * 0.4 + (100 - academic) * 0.3 + suspensions * 10
                        
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
                        fig.update_layout(height=300)
                        st.plotly_chart(fig)
                        
                        if risk_score > 50:
                            st.warning("High risk of chronic absenteeism!")
                        
                        st.write("**Attendance Trends**")
                        fig = go.Figure()
                        # Include historical attendance data if available
                        historical_attendance = st.session_state.data[
                            st.session_state.data["Student_ID"] == st.session_state.selected_student_id
                        ][["Year", "Attendance_Percentage"]].sort_values("Year") if st.session_state.data is not None else pd.DataFrame()
                        
                        if not historical_attendance.empty:
                            fig.add_trace(go.Scatter(
                                x=historical_attendance["Year"],
                                y=historical_attendance["Attendance_Percentage"],
                                mode="lines+markers",
                                name="Historical Attendance (%)",
                                hovertemplate="Year: %{x}<br>Attendance: %{y:.2f}%"
                            ))
                        
                        fig.add_trace(go.Scatter(
                            x=[student_data["Year"].iloc[0]],
                            y=[student_data["Attendance_Percentage"].iloc[0]],
                            mode="lines+markers",
                            name="Current Year Attendance (%)",
                            hovertemplate="Year: %{x}<br>Attendance: %{y:.2f}%"
                        ))
                        fig.update_layout(
                            title="Attendance Trend",
                            xaxis_title="Year",
                            yaxis_title="Attendance Percentage",
                            width=400,
                            height=300
                        )
                        st.plotly_chart(fig)
        else:
            st.info("Run predictions to enable single student analysis.")

# Page 4: Documentation
elif st.session_state.page == "📚 Documentation":
    st.markdown("""
    <h1>
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#3498db" style="width: 30px; height: 30px; vertical-align: middle; margin-right: 10px;">
            <path d="M4 3h16a2 2 0 012 2v14a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2zm1 2v14h14V5H5zm2 2h10v2H7V7zm0 4h10v2H7v-2zm0 4h7v2H7v-2z"/>
        </svg>
        📚 Documentation
    </h1>
    """, unsafe_allow_html=True)
    
    if st.session_state.data is not None:
        st.header("Patterns & Correlations Visualizer Dashboard")
        high_risk = st.session_state.data[st.session_state.data["CA_Status"] == "CA"]
        if not high_risk.empty:
            st.subheader("Discovered Patterns")
            if st.session_state.patterns:
                with st.expander("View Discovered Patterns", expanded=False):
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
                                st.experimental_rerun()
                
                if 'edit_pattern_index' in st.session_state:
                    st.subheader("Edit Pattern")
                    idx = st.session_state.edit_pattern_index
                    new_pattern = st.text_input("Pattern", value=st.session_state.patterns[idx]["pattern"])
                    new_explanation = st.text_area("Explanation", value=st.session_state.patterns[idx]["explanation"])
                    if st.button("Save Changes"):
                        st.session_state.patterns[idx] = {"pattern": new_pattern, "explanation": new_explanation}
                        del st.session_state.edit_pattern_index
                        st.experimental_rerun()
            
            st.subheader("Add New Pattern")
            new_pattern = st.text_input("New Pattern")
            new_explanation = st.text_area("Pattern Explanation")
            if st.button("Add Pattern"):
                if new_pattern and new_explanation:
                    st.session_state.patterns.append({"pattern": new_pattern, "explanation": new_explanation})
                    st.success("Pattern added successfully!")
                    st.experimental_rerun()
            
            st.subheader("Correlation Visualizations")
            numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns
            corr = st.session_state.data[numeric_cols].corr()
            fig = px.imshow(corr, title="Correlation Heatmap of Features", labels={"color": "Correlation"})
            st.plotly_chart(fig)
            
            st.write("**Correlation with Attendance**")
            corr_attendance = corr["Attendance_Percentage"].drop("Attendance_Percentage")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=corr_attendance.index,
                y=corr_attendance.values,
                marker_color=np.where(corr_attendance > 0, "#3498db", "#e74c3c")
            ))
            fig.update_layout(
                title="Correlation of Attendance with Other Factors",
                xaxis_title="Feature",
                yaxis_title="Correlation Coefficient"
            )
            st.plotly_chart(fig)
            
            st.write("**Attendance vs. Suspensions**")
            fig = px.scatter(st.session_state.data, x="Suspensions", y="Attendance_Percentage", 
                           color="CA_Status", title="Attendance vs. Suspensions")
            st.plotly_chart(fig)
        
        st.header("AI-Powered Pattern Recognition")
        if st.session_state.models:
            displayed = False
            for model_name in ["Random Forest", "Decision Tree", "Gradient Boosting"]:
                if model_name in st.session_state.models:
                    model_info = st.session_state.models[model_name]
                    fig = plot_feature_importance(model_info["model"], model_info["feature_names"])
                    if fig:
                        st.write(f"Key factors influencing absenteeism (based on {model_name} feature importance):")
                        st.plotly_chart(fig)
                        displayed = True
                        break
            if not displayed:
                st.write("No feature importance available. Train a Random Forest, Decision Tree, or Gradient Boosting model.")
        else:
            st.info("Train models to enable AI-powered pattern recognition.")
        
        st.header("Group Analysis & Comparisons")
        st.write("""
        **Purpose of Analysis**: Use the dropdown to visualize Drop Off % for each value of a selected feature (e.g., Grade, Gender, or custom fields like Extracurricular) in the current-year data. Optional filters allow you to subset historical data for deeper analysis of attendance trends.
        """)
        
        # Dropdown for selecting field to visualize Drop Off %
        if st.session_state.current_data is not None and "CA_Prediction" in st.session_state.current_data.columns:
            group_by_options = [col for col in st.session_state.current_data.columns if st.session_state.current_data[col].dtype == "object" or col == "Grade"]
            group_by_options += [f["name"] for f in st.session_state.current_custom_fields if f["name"] in st.session_state.current_data.columns]
            group_by_field = st.selectbox("Visualize Drop Off % By", group_by_options, key="group_analysis_field")
            
            if group_by_field:
                dropoff_data = st.session_state.current_data.groupby(group_by_field)["CA_Prediction"].value_counts(normalize=True).unstack().fillna(0)
                if "CA" in dropoff_data.columns:
                    dropoff_data["Drop_Off_Percent"] = dropoff_data["CA"] * 100
                    dropoff_data = dropoff_data[["Drop_Off_Percent"]].reset_index()
                    fig = px.bar(
                        dropoff_data,
                        x=group_by_field,
                        y="Drop_Off_Percent",
                        title=f"Drop Off Percentage by {group_by_field}",
                        labels={"Drop_Off_Percent": "Drop Off Percentage (%)"},
                        color="Drop_Off_Percent",
                        color_continuous_scale="Reds"
                    )
                    fig.update_traces(hovertemplate=f"{group_by_field}: %{{x}}<br>Drop Off %: %{{y:.2f}}%")
                    st.plotly_chart(fig)
                else:
                    st.warning(f"No students predicted as CA for grouping by {group_by_field}.")
        
        # Optional filters for historical data
        with st.expander("Advanced Filters for Historical Data", expanded=False):
            st.write("Filter historical data to analyze specific student cohorts:")
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
            
            st.write("**Attendance Trends by Group**")
            if not filtered_data.empty:
                group_by = st.selectbox("Group By", ["Grade", "Gender", "School", "Meal_Code", "Transportation"], key="historical_group_by")
                trend_data = filtered_data.groupby(group_by)["Attendance_Percentage"].mean().reset_index()
                fig = px.bar(trend_data, x=group_by, y="Attendance_Percentage", title=f"Average Attendance by {group_by}")
                st.plotly_chart(fig)
                
                fig = px.box(filtered_data, x=group_by, y="Attendance_Percentage", title=f"Attendance Distribution by {group_by}")
                st.plotly_chart(fig)
        
        st.subheader("Advanced Attendance Visualizations")
        if not high_risk.empty:
            with tempfile.TemporaryDirectory() as tmpdirname:
                if st.session_state.current_data is not None:
                    hist_trend = high_risk.groupby("Grade")["Attendance_Percentage"].mean().reset_index()
                    hist_trend["Dataset"] = "Historical"
                    curr_trend = st.session_state.current_data.groupby("Grade")["Attendance_Percentage"].mean().reset_index()
                    curr_trend["Dataset"] = "Current Year"
                    combined_trend = pd.concat([hist_trend, curr_trend], ignore_index=True)
                    
                    plt.figure(figsize=(10, 6))
                    sns.barplot(data=combined_trend, x="Grade", y="Attendance_Percentage", hue="Dataset", palette="Blues")
                    plt.title("Average Attendance by Grade: Historical vs. Current Year")
                    plt.xlabel("Grade")
                    plt.ylabel("Average Attendance (%)")
                    plt.legend(title="Dataset")
                    bar_plot_path = os.path.join(tmpdirname, "attendance_bar.png")
                    plt.savefig(bar_plot_path, bbox_inches="tight")
                    plt.close()
                    st.image(bar_plot_path, caption="Grouped Bar Plot of Average Attendance by Grade")
                
                plt.figure(figsize=(10, 6))
                sns.violinplot(data=high_risk, x="Grade", y="Attendance_Percentage", palette="Blues")
                plt.title("Attendance Distribution by Grade (Historical Data)")
                plt.xlabel("Grade")
                plt.ylabel("Attendance Percentage (%)")
                violin_plot_path = os.path.join(tmpdirname, "attendance_violin.png")
                plt.savefig(violin_plot_path, bbox_inches="tight")
                plt.close()
                st.image(violin_plot_path, caption="Violin Plot of Attendance Distribution by Grade")
        
        st.header("Intervention Recommendations")
        if not high_risk.empty:
            with st.expander("Why Historical Data?", expanded=False):
                st.markdown("""
                **Why Historical Data?**

                Historical data for high-risk students (CA Status = CA) helps identify patterns and risk factors. Key features like attendance, academic performance, suspensions, and transportation reveal characteristics:
                - Low attendance indicates chronic absence.
                - Poor academic performance may show disengagement.
                - Suspensions suggest behavioral challenges.
                - Transportation issues may limit access.

                These insights inform targeted interventions like tutoring, counseling, or transportation support.
                """)
            
            st.write("High-risk students (CA Status = CA):")
            st.dataframe(high_risk[["Student_ID", "Attendance_Percentage", "Academic_Performance", "Suspensions", "Transportation"]])
            st.write("Recommended Actions:")
            st.write("- **Tutoring Programs**: For low academic performance.")
            st.write("- **Counseling Services**: For high absence rates or suspensions.")
            st.write("- **Parental Engagement**: Contact families of high-risk students.")
            st.write("- **Transportation Support**: Assist students with unreliable transportation.")
