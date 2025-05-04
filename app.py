import streamlit as st
import pandas as pd
import uuid
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from data_processing import (
    load_uploaded_data, compute_high_risk_baselines, preprocess_data, combine_datasets
)
from data_generator import (
    generate_historical_data, generate_current_year_data
)
from model_training import (
    train_and_tune_model, run_predictions, plot_confusion_matrix,
    plot_feature_importance, get_model_explanation
)

# Set page config
st.set_page_config(page_title="Chronic Absenteeism Prediction", layout="wide")

# Load CSS for styling
st.markdown('<style>' + open('styles.css').read() + '</style>', unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    defaults = {
        'data': None,
        'datasets': {},
        'models': {},
        'current_data': None,
        'custom_fields': [],
        'current_custom_fields': [],
        'selected_student_id': None,
        'model_versions': {},
        'patterns': [],
        'page': "📝 Data Configuration",
        'compare_models': [],
        'drop_off_rules': {},
        'high_risk_baselines': None,
        'selected_group_by': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Clear session state
def clear_session_state():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_session_state()

initialize_session_state()

# Sidebar navigation
st.sidebar.title("Navigation")
page_options = ["📝 Data Configuration", "🤖 Model Training", "📊 Results", "📚 Documentation"]
default_index = page_options.index(st.session_state.page) if st.session_state.page in page_options else 0
page = st.sidebar.radio("Go to", page_options, index=default_index, label_visibility="collapsed")
st.session_state.page = page

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
    school_prefix = st.text_input("School Prefix (e.g., 10U)", "10U")
    num_schools = st.number_input("Number of Schools", 1, 10, 3)
    id_length = st.radio("Student ID Length", [5, 7], index=0)
    dropoff_percent = st.slider("Target CA Percentage (%)", 5, 50, 20, step=5)
    
    grades = st.multiselect("Grades", list(range(1, 13)), default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    
    st.subheader("Gender Distribution (%)")
    male_dist = st.slider("Male (%)", 0, 100, 40, step=5)
    female_dist = st.slider("Female (%)", 0, 100, 40, step=5)
    other_dist = st.slider("Other (%)", 0, 100, 20, step=5)
    
    total_dist = male_dist + female_dist + other_dist
    gender_dist = [male_dist, female_dist, other_dist] if total_dist == 100 else None
    if total_dist != 100:
        st.error(f"Gender distribution must sum to 100%. Current total: {total_dist}%")
    
    meal_codes = st.multiselect("Meal Codes", ["Free", "Reduced", "Paid"], default=["Free", "Reduced", "Paid"])
    academic_perf = st.slider("Academic Performance Range (%)", 1, 100, (40, 90))
    academic_perf_valid = academic_perf[0] < academic_perf[1]
    if not academic_perf_valid:
        st.error("Academic Performance Range: Minimum must be less than maximum.")
    
    transportation = st.multiselect("Transportation Options", ["Bus", "Walk", "Car"], default=["Bus", "Walk"])
    suspensions_range = st.slider("Suspensions Range (per year)", 0, 10, (0, 3))
    suspensions_valid = suspensions_range[0] < suspensions_range[1]
    if not suspensions_valid:
        st.error("Suspensions Range: Minimum must be less than maximum.")
    
    st.subheader("Attendance Data")
    total_days = 180
    st.write(f"Total School Days: {total_days}")
    present_days_range = st.slider("Present Days Range", 0, total_days - 1, (100, 179))
    present_days_valid = present_days_range[0] < present_days_range[1]
    if not present_days_valid:
        st.error("Present Days Range: Minimum must be less than maximum.")
    
    max_absent_days = total_days - present_days_range[0]
    if max_absent_days <= 0:
        st.error(f"Minimum present days ({present_days_range[0]}) equals or exceeds total days ({total_days}).")
        max_absent_days = 1
    
    absent_days_range = st.slider(
        "Absent Days Range",
        0,
        max_absent_days,
        (0, min(80, max_absent_days - 1)),
        help=f"Maximum absent days cannot exceed {max_absent_days}."
    )
    absent_days_valid = absent_days_range[0] < absent_days_range[1]
    if not absent_days_valid:
        st.error("Absent Days Range: Minimum must be less than maximum.")
    
    attendance_valid = (
        max_absent_days > 0 and
        present_days_valid and
        absent_days_valid and
        present_days_range[0] + absent_days_range[1] <= total_days and
        present_days_range[1] + absent_days_range[0] >= total_days - max_absent_days
    )
    if not attendance_valid:
        st.error("Invalid attendance configuration. Ensure present and absent days are consistent with total days.")
    
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
    
    st.subheader("Drop Off Rules")
    with st.expander("Define Drop Off Rules (Optional)", expanded=False):
        attendance_min = st.slider("Attendance Percentage Min (%)", 0, 100, 0, step=5)
        attendance_max = st.slider("Attendance Percentage Max (%)", 0, 100, 80, step=5)
        drop_off_rules_valid = attendance_min <= attendance_max
        if not drop_off_rules_valid:
            st.error("Attendance Percentage Min must be less than or equal to Max.")
        
        drop_off_features = st.multiselect(
            "Select Features for Drop Off Rules",
            ["Grade", "Transportation", "Meal_Code", "Gender", "School"] + [f["name"] for f in st.session_state.custom_fields if f["name"]],
            key="drop_off_features"
        )
        drop_off_rules = {"attendance_min": attendance_min, "attendance_max": attendance_max, "features": {}}
        
        for feature in drop_off_features:
            if feature == "Grade":
                values = st.multiselect(f"Select {feature} Values", grades, default=grades, key=f"drop_off_{feature}")
            elif feature == "Transportation":
                values = st.multiselect(f"Select {feature} Values", transportation, default=transportation, key=f"drop_off_{feature}")
            elif feature == "Meal_Code":
                values = st.multiselect(f"Select {feature} Values", meal_codes, default=meal_codes, key=f"drop_off_{feature}")
            elif feature == "Gender":
                values = st.multiselect(f"Select {feature} Values", ["Male", "Female", "Other"], default=["Male", "Female", "Other"], key=f"drop_off_{feature}")
            elif feature == "School":
                schools = [f"{school_prefix}{i:03d}" for i in range(1, num_schools + 1)]
                values = st.multiselect(f"Select {feature} Values", schools, default=schools, key=f"drop_off_{feature}")
            else:
                custom_field = next((f for f in st.session_state.custom_fields if f["name"] == feature), None)
                if custom_field:
                    values_list = [v.strip() for v in custom_field["values"].split(",")]
                    values = st.multiselect(f"Select {feature} Values", values_list, default=values_list, key=f"drop_off_{feature}")
                else:
                    values = []
            if values:
                drop_off_rules["features"][feature] = values
    
    generate_disabled = not (gender_dist and attendance_valid and academic_perf_valid and suspensions_valid and drop_off_rules_valid)
    if st.button("Generate Historical Data", disabled=generate_disabled):
        try:
            custom_fields = [(f["name"], f["values"]) for f in st.session_state.custom_fields if f["name"] and f["values"]]
            data = generate_historical_data(
                num_students, year_start, year_end, school_prefix, num_schools,
                grades, gender_dist, meal_codes, academic_perf, transportation,
                suspensions_range, present_days_range, absent_days_range, total_days,
                custom_fields, id_length, dropoff_percent, drop_off_rules
            )
            dataset_id = str(uuid.uuid4())
            st.session_state.datasets[dataset_id] = data
            st.session_state.data = data
            st.session_state.drop_off_rules = drop_off_rules
            st.session_state.high_risk_baselines = compute_high_risk_baselines(data)
            st.success(f"Data generated successfully! Dataset ID: {dataset_id}")
            
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
    uploaded_files = st.file_uploader("Upload Historical Data (CSV)", type=["csv"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                data = load_uploaded_data(uploaded_file)
                dataset_id = str(uuid.uuid4())
                st.session_state.datasets[dataset_id] = data
                st.success(f"Dataset uploaded successfully! ID: {dataset_id}")
            except Exception as e:
                st.error(f"Error uploading {uploaded_file.name}: {str(e)}")
    
    if st.session_state.datasets:
        st.subheader("Select Datasets")
        selected_datasets = []
        for dataset_id, data in st.session_state.datasets.items():
            label = f"Dataset {dataset_id[:8]} (Rows: {len(data)}, Columns: {len(data.columns)})"
            if st.checkbox(label, key=f"dataset_{dataset_id}"):
                selected_datasets.append(dataset_id)
        
        if not selected_datasets:
            st.warning("Please select at least one dataset to proceed.")
        else:
            st.subheader("Dataset Summary")
            for ds_id in selected_datasets:
                data = st.session_state.datasets[ds_id]
                st.write(f"**Dataset {ds_id[:8]}**: {len(data)} rows, Columns: {', '.join(data.columns)}")
            
            st.subheader("Feature Selection")
            combined_data = combine_datasets([st.session_state.datasets[ds] for ds in selected_datasets])
            excluded_features = ["Student_ID"]
            available_features = [col for col in combined_data.columns if col not in excluded_features]
            feature_toggles = {f: st.checkbox(f, value=True, key=f"feature_{f}") for f in available_features}
            features = [f for f, enabled in feature_toggles.items() if enabled]
            
            st.subheader("Target Selection")
            # Identify potential target columns (categorical or binary with valid values)
            potential_targets = []
            for col in combined_data.columns:
                if col in ["Student_ID", "Attendance_Percentage", "Academic_Performance", "Present_Days", "Absent_Days", "Suspensions"]:
                    continue
                if combined_data[col].dtype in ["object", "category"] or len(combined_data[col].unique()) <= 2:
                    # Ensure column has no missing values and at least two non-null values
                    if not combined_data[col].isna().any() and len(combined_data[col].dropna().unique()) >= 2:
                        potential_targets.append(col)
            if not potential_targets:
                st.error("No suitable target columns found in the selected datasets. Ensure targets are categorical or binary with no missing values.")
            else:
                targets = st.multiselect("Select Targets", potential_targets, default=["CA_Status"] if "CA_Status" in potential_targets else potential_targets[:1])
                if not targets:
                    st.error("Please select at least one target.")
            
            st.subheader("Model Selection")
            models_to_train = st.multiselect(
                "Select Models",
                ["Logistic Regression", "Random Forest", "Decision Tree", "SVM", "Gradient Boosting", "Neural Network"],
                default=["Logistic Regression", "Random Forest"]
            )
            
            enable_tuning = st.checkbox("Enable Hyperparameter Tuning", value=False)
            tuning_params = {}
            if enable_tuning:
                for model_name in models_to_train:
                    st.write(f"**{model_name} Parameters**")
                    if model_name == "Logistic Regression":
                        tuning_params[model_name] = {
                            "estimator__C": st.multiselect(f"C values ({model_name})", [0.1, 1, 10], default=[0.1, 1, 10]),
                            "estimator__solver": st.multiselect(f"Solver ({model_name})", ["lbfgs", "liblinear"], default=["lbfgs", "liblinear"])
                        }
                    elif model_name == "Random Forest":
                        tuning_params[model_name] = {
                            "estimator__n_estimators": st.multiselect(f"Number of Trees ({model_name})", [50, 100, 200], default=[100]),
                            "estimator__max_depth": st.multiselect(f"Max Depth ({model_name})", [None, 10, 20], default=[None]),
                            "estimator__min_samples_split": st.multiselect(f"Min Samples Split ({model_name})", [2, 5], default=[2])
                        }
                    elif model_name == "Decision Tree":
                        tuning_params[model_name] = {
                            "estimator__max_depth": st.multiselect(f"Max Depth ({model_name})", [None, 10, 20], default=[None]),
                            "estimator__min_samples_split": st.multiselect(f"Min Samples Split ({model_name})", [2, 5], default=[2]),
                            "estimator__min_samples_leaf": st.multiselect(f"Min Samples Leaf ({model_name})", [1, 2], default=[1])
                        }
                    elif model_name == "SVM":
                        tuning_params[model_name] = {
                            "estimator__C": st.multiselect(f"C values ({model_name})", [0.1, 1, 10], default=[1]),
                            "estimator__gamma": st.multiselect(f"Gamma ({model_name})", ["scale", "auto", 0.1], default=["scale"]),
                            "estimator__kernel": st.multiselect(f"Kernel ({model_name})", ["rbf", "linear"], default=["rbf"])
                        }
                    elif model_name == "Gradient Boosting":
                        tuning_params[model_name] = {
                            "estimator__n_estimators": st.multiselect(f"Number of Trees ({model_name})", [50, 100, 200], default=[100]),
                            "estimator__learning_rate": st.multiselect(f"Learning Rate ({model_name})", [0.01, 0.1, 0.2], default=[0.1]),
                            "estimator__max_depth": st.multiselect(f"Max Depth ({model_name})", [3, 5], default=[3])
                        }
                    elif model_name == "Neural Network":
                        tuning_params[model_name] = {
                            "estimator__hidden_layer_sizes": st.multiselect(f"Hidden Layers ({model_name})", [(50,), (100,), (50, 50)], default=[(100,)]),
                            "estimator__alpha": st.multiselect(f"Alpha ({model_name})", [0.0001, 0.001], default=[0.0001])
                        }
            
            if st.button("Train Models") and features and targets:
                try:
                    with st.spinner("Training models..."):
                        model_results, patterns = train_and_tune_model(
                            combined_data, features, targets, models_to_train, enable_tuning, tuning_params
                        )
                        st.session_state.models.update(model_results["models"])
                        st.session_state.model_versions.update(model_results["model_versions"])
                        st.session_state.patterns.extend(patterns)
                        st.success("Models trained successfully!")
                        st.balloons()
                except Exception as e:
                    st.error(f"Error training models: {str(e)}")
            
            if st.session_state.models:
                st.subheader("Model Results")
                for model_name, model_info in st.session_state.models.items():
                    metrics = model_info["metrics"]
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**{model_name} Results**")
                        for target in targets:
                            st.write(f"**Target: {target}**")
                            st.write(f"Accuracy: {metrics[target]['accuracy']:.2f}")
                            st.write(f"Precision: {metrics[target]['precision']:.2f}")
                            st.write(f"Recall: {metrics[target]['recall']:.2f}")
                            st.write(f"F1 Score: {metrics[target]['f1']:.2f}")
                            st.write(f"ROC AUC: {metrics[target]['roc_auc']:.2f}")
                        if model_info["best_params"]:
                            st.write("Best Parameters:")
                            st.json(model_info["best_params"])
                    with col2:
                        for target in targets:
                            fig = plot_confusion_matrix(model_info["y_test"][target], model_info["y_pred"][target])
                            st.plotly_chart(fig)
                
                if model_name in ["Random Forest", "Decision Tree", "Gradient Boosting"]:
                    fig = plot_feature_importance(model_info["model"], model_info["feature_names"])
                    if fig:
                        st.plotly_chart(fig)

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
        dropoff_percent = st.slider("Target CA Percentage (%)", 5, 50, 20, step=5)
        
        grades = st.multiselect("Grades", list(range(1, 13)), default=[1, 2, 3, 4, 5], key="current_grades")
        
        st.subheader("Gender Distribution (%)")
        male_dist = st.slider("Male (%)", 0, 100, 40, step=5, key="current_male")
        female_dist = st.slider("Female (%)", 0, 100, 40, step=5, key="current_female")
        other_dist = st.slider("Other (%)", 0, 100, 20, step=5, key="current_other")
        
        total_dist = male_dist + female_dist + other_dist
        gender_dist = [male_dist, female_dist, other_dist] if total_dist == 100 else None
        if total_dist != 100:
            st.error(f"Gender distribution must sum to 100%. Current total: {total_dist}%")
        
        meal_codes = st.multiselect("Meal Codes", ["Free", "Reduced", "Paid"], default=["Free", "Reduced", "Paid"], key="current_meal")
        academic_perf = st.slider("Academic Performance Range (%)", 1, 100, (40, 90), key="current_academic")
        academic_perf_valid = academic_perf[0] < academic_perf[1]
        if not academic_perf_valid:
            st.error("Academic Performance Range: Minimum must be less than maximum.")
        
        transportation = st.multiselect("Transportation Options", ["Bus", "Walk", "Car"], default=["Bus", "Walk"], key="current_transport")
        suspensions_range = st.slider("Suspensions Range (per year)", 0, 10, (0, 3), key="current_suspensions")
        suspensions_valid = suspensions_range[0] < suspensions_range[1]
        if not suspensions_valid:
            st.error("Suspensions Range: Minimum must be less than maximum.")
        
        st.subheader("Attendance Data")
        total_days = 180
        st.write(f"Total School Days: {total_days}")
        present_days_range = st.slider("Present Days Range", 0, total_days - 1, (100, 179), key="current_present")
        present_days_valid = present_days_range[0] < present_days_range[1]
        if not present_days_valid:
            st.error("Present Days Range: Minimum must be less than maximum.")
        
        max_absent_days = total_days - present_days_range[0]
        if max_absent_days <= 0:
            st.error(f"Minimum present days ({present_days_range[0]}) equals or exceeds total days ({total_days}).")
            max_absent_days = 1
        
        absent_days_range = st.slider(
            "Absent Days Range",
            0,
            max_absent_days,
            (0, min(80, max_absent_days - 1)),
            key="current_absent",
            help=f"Maximum absent days cannot exceed {max_absent_days}."
        )
        absent_days_valid = absent_days_range[0] < absent_days_range[1]
        if not absent_days_valid:
            st.error("Absent Days Range: Minimum must be less than maximum.")
        
        attendance_valid = (
            max_absent_days > 0 and
            present_days_valid and
            absent_days_valid and
            present_days_range[0] + absent_days_range[1] <= total_days and
            present_days_range[1] + absent_days_range[0] >= total_days - max_absent_days
        )
        if not attendance_valid:
            st.error("Invalid attendance configuration.")
        
        # Check if historical data exists to enable/disable the checkbox
        use_historical_ids = st.checkbox("Use Historical Student IDs", value=False, disabled=st.session_state.data is None)
        include_graduates = st.checkbox("Include Graduating Students (Cap at Grade 12)", value=False, disabled=not use_historical_ids)
        
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
        
        st.subheader("Drop Off Rules")
        with st.expander("Define Drop Off Rules (Optional)", expanded=False):
            attendance_min = st.slider("Attendance Percentage Min (%)", 0, 100, 0, step=5, key="current_attendance_min")
            attendance_max = st.slider("Attendance Percentage Max (%)", 0, 100, 80, step=5, key="current_attendance_max")
            drop_off_rules_valid = attendance_min <= attendance_max
            if not drop_off_rules_valid:
                st.error("Attendance Percentage Min must be less than or equal to Max.")
            
            drop_off_features = st.multiselect(
                "Select Features for Drop Off Rules",
                ["Grade", "Transportation", "Meal_Code", "Gender", "School"] + [f["name"] for f in st.session_state.current_custom_fields if f["name"]],
                key="current_drop_off_features"
            )
            drop_off_rules = {"attendance_min": attendance_min, "attendance_max": attendance_max, "features": {}}
            
            for feature in drop_off_features:
                if feature == "Grade":
                    values = st.multiselect(f"Select {feature} Values", grades, default=grades, key=f"current_drop_off_{feature}")
                elif feature == "Transportation":
                    values = st.multiselect(f"Select {feature} Values", transportation, default=transportation, key=f"current_drop_off_{feature}")
                elif feature == "Meal_Code":
                    values = st.multiselect(f"Select {feature} Values", meal_codes, default=meal_codes, key=f"current_drop_off_{feature}")
                elif feature == "Gender":
                    values = st.multiselect(f"Select {feature} Values", ["Male", "Female", "Other"], default=["Male", "Female", "Other"], key=f"current_drop_off_{feature}")
                elif feature == "School":
                    schools = [f"{school_prefix}{i:03d}" for i in range(1, num_schools + 1)]
                    values = st.multiselect(f"Select {feature} Values", schools, default=schools, key=f"current_drop_off_{feature}")
                else:
                    custom_field = next((f for f in st.session_state.current_custom_fields if f["name"] == feature), None)
                    if custom_field:
                        values_list = [v.strip() for v in custom_field["values"].split(",")]
                        values = st.multiselect(f"Select {feature} Values", values_list, default=values_list, key=f"current_drop_off_{feature}")
                    else:
                        values = []
                if values:
                    drop_off_rules["features"][feature] = values
        
        generate_disabled = not (gender_dist and attendance_valid and academic_perf_valid and suspensions_valid and drop_off_rules_valid)
        if st.button("Generate Current Year Data", disabled=generate_disabled):
            try:
                custom_fields = [(f["name"], f["values"]) for f in st.session_state.current_custom_fields if f["name"] and f["values"]]
                historical_data = st.session_state.data if use_historical_ids else None
                st.session_state.current_data = generate_current_year_data(
                    num_students, school_prefix, num_schools, grades, gender_dist,
                    meal_codes, academic_perf, transportation, suspensions_range,
                    present_days_range, absent_days_range, total_days, custom_fields,
                    historical_ids=historical_data, id_length=id_length, dropoff_percent=dropoff_percent,
                    include_graduates=include_graduates, drop_off_rules=drop_off_rules
                )
                st.session_state.drop_off_rules = drop_off_rules
                dataset_id = str(uuid.uuid4())
                st.session_state.datasets[dataset_id] = st.session_state.current_data
                st.success(f"Current year data generated successfully! Dataset ID: {dataset_id}")
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
                data = load_uploaded_data(uploaded_file)
                dataset_id = str(uuid.uuid4())
                st.session_state.datasets[dataset_id] = data
                st.session_state.current_data = data
                st.success(f"Data uploaded successfully! ID: {dataset_id}")
                st.subheader("Data Preview")
                st.dataframe(st.session_state.current_data.head(10))
            except Exception as e:
                st.error(f"Error uploading data: {str(e)}")
    
    if st.session_state.current_data is not None and st.session_state.models:
        st.subheader("Run Predictions")
        selected_model = st.selectbox("Select Model", list(st.session_state.models.keys()))
        excluded_columns = ["Student_ID", "CA_Prediction", "CA_Probability", "Drop_Off", "Prediction_Causes"]
        available_features = [col for col in st.session_state.current_data.columns if col not in excluded_columns]
        feature_toggles = {f: st.checkbox(f, value=True, key=f"predict_feature_{f}") for f in available_features}
        features = [f for f, enabled in feature_toggles.items() if enabled]
        
        group_by_options = [col for col in available_features if st.session_state.current_data[col].dtype == "object" or col == "Grade"]
        group_by_options += [f["name"] for f in st.session_state.current_custom_fields if f["name"] in st.session_state.current_data.columns]
        group_by_feature = st.selectbox(
            "Group Drop Off % By",
            group_by_options,
            index=group_by_options.index(st.session_state.selected_group_by) if st.session_state.selected_group_by in group_by_options else 0
        )
        st.session_state.selected_group_by = group_by_feature
        
        if st.button("Predict"):
            try:
                prediction_data = run_predictions(
                    st.session_state.current_data, features, selected_model,
                    st.session_state.models, st.session_state.drop_off_rules,
                    st.session_state.patterns, st.session_state.high_risk_baselines
                )
                st.session_state.current_data = prediction_data
                
                st.subheader("Prediction Results")
                st.dataframe(prediction_data)
                
                heatmap_data = prediction_data.groupby(["Grade", "School"])["CA_Probability"].mean().unstack().fillna(0)
                fig = px.imshow(
                    heatmap_data,
                    title="CA Probability Heatmap by Grade and School (High Risk in Red)",
                    labels={"color": "CA Probability"},
                    color_continuous_scale="Reds"
                )
                st.plotly_chart(fig)
                
                csv = prediction_data.to_csv(index=False)
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
                    index=student_ids.index(st.session_state.selected_student_id) if st.session_state.selected_student_id in student_ids else 0
                )
                if st.form_submit_button("Analyze"):
                    st.session_state.selected_student_id = selected_id
            
            if st.session_state.selected_student_id in student_ids:
                student_data = st.session_state.current_data[st.session_state.current_data["Student_ID"] == st.session_state.selected_student_id]
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write("**Student Profile**")
                    st.dataframe(student_data)
                    
                    ca_prob = student_data["CA_Probability"].iloc[0]
                    ca_pred = student_data["CA_Prediction"].iloc[0]
                    drop_off = student_data["Drop_Off"].iloc[0]
                    causes = student_data["Prediction_Causes"].iloc[0]
                    st.write(f"**CA Prediction**: {ca_pred}")
                    st.write(f"**CA Probability**: {ca_prob:.2f}")
                    st.write(f"**Drop Off**: {drop_off}")
                    st.write(f"**Prediction Causes**: {causes}")
                
                with col2:
                    attendance = student_data["Attendance_Percentage"].iloc[0]
                    academic = student_data["Academic_Performance"].iloc[0]
                    suspensions = student_data["Suspensions"].iloc[0]
                    risk_score = (100 - attendance) * 0.4 + (100 - academic) * 0.3 + suspensions * 10
                    
                    fig_usd = go.Figure(go.Indicator(
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
                    fig_usd.update_layout(height=300)
                    st.plotly_chart(fig_usd)
                    
                    if risk_score > 50:
                        st.warning("High risk of chronic absenteeism!")

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
    
    st.header("Patterns & Correlations")
    if st.session_state.patterns:
        st.write("**Discovered Patterns**")
        for pattern in st.session_state.patterns:
            st.write(f"- {pattern['pattern']}: {pattern['explanation']}")
    
    if st.session_state.models:
        st.header("Feature Importance")
        for model_name in ["Random Forest", "Decision Tree", "Gradient Boosting"]:
            if model_name in st.session_state.models:
                fig = plot_feature_importance(st.session_state.models[model_name]["model"], st.session_state.models[model_name]["feature_names"])
                if fig:
                    st.plotly_chart(fig)
                    break
