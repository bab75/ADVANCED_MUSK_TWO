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
from data_generator import generate_historical_data, generate_current_year_data
from model_utils import train_model, tune_model, get_model_explanation, plot_confusion_matrix, plot_feature_importance

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

# Sidebar navigation with radio buttons
st.sidebar.title("Navigation")
page_options = [
    "📝 Data Configuration",
    "🤖 Model Training",
    "📊 Results",
    "📚 Documentation"
]
# Ensure valid index for radio button
default_index = 0
if st.session_state.page in page_options:
    default_index = page_options.index(st.session_state.page)
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
    
    # Historical Data Generation
    st.header("Generate Historical Data")
    num_students = st.slider("Number of Students", 100, 5000, 1000)
    year_start, year_end = st.slider("Academic Years", 2010, 2025, (2015, 2020), step=1)
    school_prefix = st.text_input("School Prefix (e.g., 10U)", "10U")
    num_schools = st.number_input("Number of Schools", 1, 10, 3)
    
    # Student Attributes
    grades = st.multiselect("Grades", list(range(1, 13)), default=[1, 2, 3, 4, 5])
    
    # Gender Distribution with validation
    st.subheader("Gender Distribution (%)")
    male_dist = st.slider("Male (%)", 0, 100, 40, step=5)
    female_dist = st.slider("Female (%)", 0, 100, 40, step=5)
    other_dist = st.slider("Other (%)", 0, 100, 20, step=5)
    
    # Validate gender distribution sums to 100%
    total_dist = male_dist + female_dist + other_dist
    if total_dist != 100:
        st.error(f"Gender distribution must sum to 100%. Current total: {total_dist}%")
        gender_dist = None
    else:
        gender_dist = [male_dist, female_dist, other_dist]
    
    meal_codes = st.multiselect("Meal Codes", ["Free", "Reduced", "Paid"], default=["Free", "Reduced", "Paid"])
    academic_perf = st.slider("Academic Performance Range (%)", 1, 100, (40, 90))
    
    # Additional Attributes
    transportation = st.multiselect("Transportation Options", ["Bus", "Walk", "Car"], default=["Bus", "Walk"])
    suspensions_range = st.slider("Suspensions Range (per year)", 0, 10, (0, 3))
    
    # Attendance Data
    st.subheader("Attendance Data")
    total_days = 180
    st.write(f"Total School Days: {total_days}")
    present_days_range = st.slider("Present Days Range", 0, total_days, (100, total_days))
    absent_days_range = st.slider("Absent Days Range", 0, total_days, (0, 80))
    
    # Validate attendance ranges
    if present_days_range[0] + absent_days_range[1] > total_days or present_days_range[1] + absent_days_range[0] < total_days:
        st.error(f"Present and Absent days ranges must allow for total days to sum to {total_days}.")
        attendance_valid = False
    else:
        attendance_valid = True
    
    # Multiple Custom Fields
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
    
    # Generate Data
    if st.button("Generate Historical Data") and gender_dist is not None and attendance_valid:
        try:
            # Clear patterns when generating new data
            st.session_state.patterns = []
            custom_fields = [(f["name"], f["values"]) for f in st.session_state.custom_fields if f["name"] and f["values"]]
            data = generate_historical_data(
                num_students, year_start, year_end, school_prefix, num_schools,
                grades, gender_dist, meal_codes, academic_perf, transportation,
                suspensions_range, present_days_range, absent_days_range, total_days, custom_fields
            )
            st.session_state.data = data
            st.success("Data generated successfully!")
            
            # Data Preview
            st.subheader("Data Preview")
            st.dataframe(data.head(10))
            
            # Download option
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
    
    # Data Upload or Use Generated
    st.header("Load Data")
    data_source = st.radio("Data Source", ["Use Generated Data", "Upload CSV"])
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload Historical Data (CSV)", type=["csv"])
        if uploaded_file:
            try:
                st.session_state.data = pd.read_csv(uploaded_file)
                st.session_state.patterns = []  # Clear patterns for new data
                st.success("Data uploaded successfully!")
            except Exception as e:
                st.error(f"Error uploading data: {str(e)}")
    
    if st.session_state.data is not None:
        st.subheader("Feature Selection")
        include_student_id = st.checkbox("Include Historical Student ID", value=True)
        excluded_features = ["CA_Status"]
        if not include_student_id:
            excluded_features.append("Student_ID")
        available_features = [col for col in st.session_state.data.columns if col not in excluded_features]
        feature_toggles = {}
        st.write("Toggle Features:")
        for feature in available_features:
            feature_toggles[feature] = st.checkbox(feature, value=True, key=f"feature_{feature}")
        features = [f for f, enabled in feature_toggles.items() if enabled]
        target = "CA_Status"
        
        # Identify categorical and numerical columns
        categorical_cols = [col for col in features if st.session_state.data[col].dtype == "object"]
        numerical_cols = [col for col in features if col not in categorical_cols]
        
        # Model Selection
        st.subheader("Model Selection")
        models_to_train = st.multiselect("Select Models", [
            "Logistic Regression", "Random Forest", "Decision Tree",
            "SVM", "Gradient Boosting", "Neural Network"
        ], default=["Logistic Regression", "Random Forest"])
        
        # Hyperparameter Tuning
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
                    
                    # Preprocessing pipeline
                    preprocessor = ColumnTransformer(
                        transformers=[
                            ("num", StandardScaler(), numerical_cols),
                            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_cols)
                        ])
                    
                    st.session_state.preprocessor = preprocessor
                    X_train_processed = preprocessor.fit_transform(X_train)
                    X_test_processed = preprocessor.transform(X_test)
                    
                    # Get feature names after encoding
                    feature_names = numerical_cols + list(preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols))
                    
                    # Training Status Dashboard
                    st.subheader("Training Status")
                    status_container = st.empty()
                    
                    st.session_state.models = {}
                    comparison_data = []
                    for model_name in models_to_train:
                        try:
                            status_container.write(f"Training {model_name}...")
                            if enable_tuning and model_name in tuning_params:
                                model, metrics, best_params = tune_model(model_name, X_train_processed, y_train, X_test_processed, y_test, tuning_params[model_name])
                            else:
                                model, metrics = train_model(model_name, X_train_processed, y_train, X_test_processed, y_test)
                                best_params = None
                            
                            # Store model version
                            version_id = str(uuid.uuid4())
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            st.session_state.model_versions.setdefault(model_name, []).append({
                                "version_id": version_id,
                                "timestamp": timestamp,
                                "metrics": metrics,
                                "best_params": best_params,
                                "model": model,
                                "preprocessor": preprocessor,
                                "feature_names": feature_names
                            })
                            
                            st.session_state.models[model_name] = {
                                "model": model,
                                "metrics": metrics,
                                "preprocessor": preprocessor,
                                "feature_names": feature_names,
                                "best_params": best_params
                            }
                            
                            # Collect metrics for comparison
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
                    
                    # Model Selection & Versioning Dashboard
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
                        
                        # Model Comparison
                        st.write("Compare Model Versions")
                        compare_models = st.multiselect("Select Models to Compare", list(st.session_state.model_versions.keys()))
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
                                        name=metric
                                    ))
                                fig.update_layout(
                                    title="Model Version Comparison",
                                    barmode="group",
                                    xaxis_title="Model (Version)",
                                    yaxis_title="Score"
                                )
                                st.plotly_chart(fig)
                    
                    # Display Individual Model Results
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
                                fig = plot_confusion_matrix(y_test, metrics['y_pred'])
                                st.plotly_chart(fig)
                            
                            # Feature Importance (if applicable)
                            if model_name in ["Random Forest", "Decision Tree", "Gradient Boosting"]:
                                fig = plot_feature_importance(model_info["model"], model_info["feature_names"])
                                st.plotly_chart(fig)
                            
                            # Model Explanation with Example
                            st.write(get_model_explanation(model_name, X_test_processed[:1], model_info["model"]))
            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
        
        # Pattern Discovery Module
        if st.session_state.data is not None:
            st.subheader("Pattern Discovery")
            high_risk = st.session_state.data[st.session_state.data["CA_Status"] == "CA"]
            if not high_risk.empty:
                patterns = []
                low_attendance = f"Average Attendance: {high_risk['Attendance_Percentage'].mean():.2f}%"
                common_grades = f"Common Grades: {', '.join(map(str, high_risk['Grade'].mode().tolist()))}"
                common_meal_codes = f"Common Meal Codes: {', '.join(high_risk['Meal_Code'].mode().tolist())}"
                common_transport = f"Common Transportation: {', '.join(high_risk['Transportation'].mode().tolist())}"
                
                # Check for duplicates before adding
                existing_patterns = [p["pattern"] for p in st.session_state.patterns]
                for pattern in [low_attendance, common_grades, common_meal_codes, common_transport]:
                    if pattern not in existing_patterns:
                        patterns.append({"pattern": pattern, "explanation": "Identified in high-risk students (CA Status = CA)"})
                
                if patterns:
                    st.session_state.patterns.extend(patterns)
                    st.write(f"Number of patterns learned: {len(st.session_state.patterns)}")
                    for pattern in patterns:
                        st.write(f"- {pattern['pattern']}: {pattern['explanation']}")

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
    
    # Current Year Data
    st.header("Generate Current Year Data")
    data_source = st.radio("Data Source", ["Generate Data", "Upload CSV"])
    
    if data_source == "Generate Data":
        num_students = st.slider("Number of Students", 100, 5000, 1000)
        school_prefix = st.text_input("School Prefix (e.g., CU)", "CU")
        num_schools = st.number_input("Number of Schools", 1, 10, 3)
        
        # Student Attributes
        grades = st.multiselect("Grades", list(range(1, 13)), default=[1, 2, 3, 4, 5], key="current_grades")
        
        # Gender Distribution with validation
        st.subheader("Gender Distribution (%)")
        male_dist = st.slider("Male (%)", 0, 100, 40, step=5, key="current_male")
        female_dist = st.slider("Female (%)", 0, 100, 40, step=5, key="current_female")
        other_dist = st.slider("Other (%)", 0, 100, 20, step=5, key="current_other")
        
        # Validate gender distribution sums to 100%
        total_dist = male_dist + female_dist + other_dist
        if total_dist != 100:
            st.error(f"Gender distribution must sum to 100%. Current total: {total_dist}%")
            gender_dist = None
        else:
            gender_dist = [male_dist, female_dist, other_dist]
        
        meal_codes = st.multiselect("Meal Codes", ["Free", "Reduced", "Paid"], default=["Free", "Reduced", "Paid"], key="current_meal")
        academic_perf = st.slider("Academic Performance Range (%)", 1, 100, (40, 90), key="current_academic")
        
        # Additional Attributes
        transportation = st.multiselect("Transportation Options", ["Bus", "Walk", "Car"], default=["Bus", "Walk"], key="current_transport")
        suspensions_range = st.slider("Suspensions Range (per year)", 0, 10, (0, 3), key="current_suspensions")
        
        # Attendance Data
        st.subheader("Attendance Data")
        total_days = 180
        st.write(f"Total School Days: {total_days}")
        present_days_range = st.slider("Present Days Range", 0, total_days, (100, total_days), key="current_present")
        absent_days_range = st.slider("Absent Days Range", 0, total_days, (0, 80), key="current_absent")
        
        # Validate attendance ranges
        if present_days_range[0] + absent_days_range[1] > total_days or present_days_range[1] + absent_days_range[0] < total_days:
            st.error(f"Present and Absent days ranges must allow for total days to sum to {total_days}.")
            attendance_valid = False
        else:
            attendance_valid = True
        
        # Multiple Custom Fields
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
        
        if st.button("Generate Current Year Data") and gender_dist is not None and attendance_valid:
            try:
                custom_fields = [(f["name"], f["values"]) for f in st.session_state.current_custom_fields if f["name"] and f["values"]]
                st.session_state.current_data = generate_current_year_data(
                    num_students, school_prefix, num_schools, grades, gender_dist,
                    meal_codes, academic_perf, transportation, suspensions_range,
                    present_days_range, absent_days_range, total_days, custom_fields
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
    
    # Run Predictions
    if st.session_state.current_data is not None and st.session_state.models:
        st.subheader("Run Predictions")
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
                
                # Results
                st.session_state.current_data["CA_Prediction"] = predictions
                st.session_state.current_data["CA_Probability"] = probabilities
                
                st.subheader("Prediction Results")
                st.dataframe(st.session_state.current_data)
                
                # Predictive Heatmap
                heatmap_data = st.session_state.current_data.groupby(["Grade", "School"])["CA_Probability"].mean().unstack().fillna(0)
                fig = px.imshow(heatmap_data, title="CA Probability Heatmap by Grade and School (High Risk in Red)", 
                               labels={"color": "CA Probability"}, color_continuous_scale="Reds")
                st.plotly_chart(fig)
                
                # Visualizations
                fig = px.histogram(st.session_state.current_data, x="CA_Probability", 
                                 color="CA_Prediction", title="CA Probability Distribution")
                st.plotly_chart(fig)
                
                # Heatmap by Grade and Prediction
                heatmap_data = st.session_state.current_data.groupby(["Grade", "CA_Prediction"]).size().unstack().fillna(0)
                fig = px.imshow(heatmap_data, title="CA Prediction Heatmap by Grade")
                st.plotly_chart(fig)
                
                # Download Results
                csv = st.session_state.current_data.to_csv(index=False)
                st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Error running predictions: {str(e)}")
        
        # Single Student Analysis
        st.subheader("Single Student Analysis")
        if "CA_Prediction" in st.session_state.current_data.columns:
            student_ids = st.session_state.current_data["Student_ID"].tolist()
            search_query = st.text_input("Search Student ID", value=st.session_state.selected_student_id or "", key="student_search")
            
            # Autocomplete suggestions
            if search_query:
                suggestions = [sid for sid in student_ids if search_query.lower() in sid.lower()][:10]
                if suggestions:
                    selected_suggestion = st.selectbox("Suggestions", suggestions, key="student_suggestion")
                    if selected_suggestion:
                        st.session_state.selected_student_id = selected_suggestion
            
            if st.session_state.selected_student_id in student_ids:
                student_data = st.session_state.current_data[st.session_state.current_data["Student_ID"] == st.session_state.selected_student_id]
                if not student_data.empty:
                    st.write("**Student Profile**")
                    st.dataframe(student_data)
                    
                    # Risk Assessment Score
                    attendance = student_data["Attendance_Percentage"].iloc[0]
                    academic = student_data["Academic_Performance"].iloc[0]
                    suspensions = student_data["Suspensions"].iloc[0]
                    risk_score = (100 - attendance) * 0.4 + (100 - academic) * 0.3 + suspensions * 10
                    
                    # Gauge Plot for Risk Score
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
                    
                    # Trends Visualization
                    st.write("**Attendance Trends**")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=[student_data["Year"].iloc[0]],
                        y=[student_data["Attendance_Percentage"].iloc[0]],
                        mode="lines+markers",
                        name="Attendance (%)"
                    ))
                    fig.update_layout(
                        title="Attendance Trend",
                        xaxis_title="Year",
                        yaxis_title="Attendance Percentage"
                    )
                    st.plotly_chart(fig)
                    
                    # CA Prediction and Preventive Actions
                    ca_prob = student_data["CA_Probability"].iloc[0]
                    ca_pred = student_data["CA_Prediction"].iloc[0]
                    st.write(f"**CA Prediction**: {ca_pred}")
                    st.write(f"**CA Probability**: {ca_prob:.2f}")
                    st.write("**Preventive Actions**:")
                    if ca_pred == "CA" or risk_score > 50:
                        st.write("- Increase attendance monitoring.")
                        st.write("- Provide academic support or tutoring.")
                        st.write("- Engage with parents to address absence causes.")
                        if suspensions > 0:
                            st.write("- Address behavioral issues through counseling.")
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
                
                # Edit Pattern
                if 'edit_pattern_index' in st.session_state:
                    st.subheader("Edit Pattern")
                    idx = st.session_state.edit_pattern_index
                    new_pattern = st.text_input("Pattern", value=st.session_state.patterns[idx]["pattern"])
                    new_explanation = st.text_area("Explanation", value=st.session_state.patterns[idx]["explanation"])
                    if st.button("Save Changes"):
                        st.session_state.patterns[idx] = {"pattern": new_pattern, "explanation": new_explanation}
                        del st.session_state.edit_pattern_index
                        st.experimental_rerun()
            
            # Add New Pattern
            st.subheader("Add New Pattern")
            new_pattern = st.text_input("New Pattern")
            new_explanation = st.text_area("Pattern Explanation")
            if st.button("Add Pattern"):
                if new_pattern and new_explanation:
                    st.session_state.patterns.append({"pattern": new_pattern, "explanation": new_explanation})
                    st.success("Pattern added successfully!")
                    st.experimental_rerun()
            
            # Visualizations
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
            model_name = list(st.session_state.models.keys())[0]
            model_info = st.session_state.models[model_name]
            if model_name in ["Random Forest", "Decision Tree", "Gradient Boosting"]:
                fig = plot_feature_importance(model_info["model"], model_info["feature_names"])
                if fig:
                    st.write("Key factors influencing absenteeism (based on feature importance):")
                    st.plotly_chart(fig)
        
        st.header("Group Analysis & Comparisons")
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
        
        st.write("**Attendance Trends by Group**")
        if not filtered_data.empty:
            group_by = st.selectbox("Group By", ["Grade", "Gender", "School", "Meal_Code", "Transportation"])
            trend_data = filtered_data.groupby(group_by)["Attendance_Percentage"].mean().reset_index()
            fig = px.bar(trend_data, x=group_by, y="Attendance_Percentage", title=f"Average Attendance by {group_by}")
            st.plotly_chart(fig)
            
            # Box Plot for Distribution
            fig = px.box(filtered_data, x=group_by, y="Attendance_Percentage", title=f"Attendance Distribution by {group_by}")
            st.plotly_chart(fig)
        
        st.header("Intervention Recommendations")
        if not high_risk.empty:
            st.write("High-risk students (CA Status = CA):")
            st.dataframe(high_risk[["Student_ID", "Attendance_Percentage", "Academic_Performance", "Suspensions", "Transportation"]])
            st.write("Recommended Actions:")
            st.write("- **Tutoring Programs**: For students with low academic performance.")
            st.write("- **Counseling Services**: For students with high absence rates or suspensions.")
            st.write("- **Parental Engagement**: Contact families of high-risk students.")
            st.write("- **Transportation Support**: Assist students with unreliable transportation.")
