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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import uuid

# Set page config
st.set_page_config(page_title="Chronic Absenteeism Prediction", layout="wide")

# Embed CSS styling
st.markdown("""
<style>
h1 { color: #2c3e50; font-size: 2.5em; }
h2 { color: #34495e; }
.stButton>button { background-color: #3498db; color: white; border-radius: 5px; }
.stButton>button:hover { background-color: #2980b9; }
.action-item { display: flex; align-items: center; margin: 10px 0; }
.action-icon { width: 24px; height: 24px; margin-right: 10px; }
.warning { background-color: #f8d7da; padding: 10px; border-radius: 5px; }
.info { background-color: #d1ecf1; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# Data generation function
def generate_historical_data(
    num_students, year_start, year_end, school_prefix, num_schools, grades, gender_dist,
    meal_codes, academic_perf, transportation, suspensions_range, present_days_range,
    absent_days_range, total_days, custom_fields, id_length, dropoff_percent, drop_off_rules
):
    years = list(range(year_start, year_end + 1))
    schools = [f"{school_prefix}{i:03d}" for i in range(1, num_schools + 1)]
    student_ids = [f"S{str(i).zfill(id_length)}" for i in range(1, num_students * len(years) + 1)]
    
    data = []
    for year in years:
        for _ in range(num_students):
            grade = np.random.choice(grades)
            school = np.random.choice(schools)
            gender = np.random.choice(["Male", "Female", "Other"], p=[g/100 for g in gender_dist])
            meal_code = np.random.choice(meal_codes)
            transport = np.random.choice(transportation)
            academic = np.random.uniform(academic_perf[0], academic_perf[1])
            suspensions = np.random.randint(suspensions_range[0], suspensions_range[1] + 1)
            present_days = np.random.randint(present_days_range[0], present_days_range[1] + 1)
            max_absent = total_days - present_days
            absent_days = np.random.randint(absent_days_range[0], min(absent_days_range[1] + 1, max_absent + 1))
            attendance_pct = (present_days / total_days) * 100
            
            ca_status = "CA" if attendance_pct < (100 - dropoff_percent) else "Non-CA"
            drop_off = "Y" if ca_status == "CA" else "N"
            if ca_status == "CA" and drop_off_rules:
                if not (drop_off_rules["attendance_min"] <= attendance_pct <= drop_off_rules["attendance_max"]):
                    drop_off = "N"
                for feature, values in drop_off_rules["features"].items():
                    if feature == "Grade" and grade not in values:
                        drop_off = "N"
                    elif feature == "Transportation" and transport not in values:
                        drop_off = "N"
                    elif feature == "Meal_Code" and meal_code not in values:
                        drop_off = "N"
                    elif feature == "Gender" and gender not in values:
                        drop_off = "N"
                    elif feature == "School" and school not in values:
                        drop_off = "N"
                    elif feature in [f["name"] for f in custom_fields]:
                        custom_field = next((f for f in custom_fields if f["name"] == feature), None)
                        if custom_field:
                            custom_values = [v.strip() for v in custom_field["values"].split(",")]
                            if custom_field["name"] not in values:
                                drop_off = "N"
            
            record = {
                "Student_ID": np.random.choice(student_ids, replace=False),
                "Year": year,
                "Grade": grade,
                "School": school,
                "Gender": gender,
                "Meal_Code": meal_code,
                "Transportation": transport,
                "Academic_Performance": academic,
                "Suspensions": suspensions,
                "Present_Days": present_days,
                "Absent_Days": absent_days,
                "Attendance_Percentage": attendance_pct,
                "CA_Status": ca_status,
                "Drop_Off": drop_off
            }
            
            for name, values in custom_fields:
                record[name] = np.random.choice([v.strip() for v in values.split(",")])
            data.append(record)
            student_ids.remove(record["Student_ID"])  # Ensure unique IDs
    
    df = pd.DataFrame(data)
    df = df.drop_duplicates(subset=["Student_ID", "Year"], keep="first")
    return df

def generate_current_year_data(
    num_students, school_prefix, num_schools, grades, gender_dist, meal_codes,
    academic_perf, transportation, suspensions_range, present_days_range,
    absent_days_range, total_days, custom_fields, historical_ids, id_length,
    dropoff_percent, include_graduates, drop_off_rules
):
    year = datetime.now().year
    schools = [f"{school_prefix}{i:03d}" for i in range(1, num_schools + 1)]
    if historical_ids is not None and not include_graduates:
        student_ids = historical_ids["Student_ID"].unique()
        num_students = min(num_students, len(student_ids))
        student_ids = np.random.choice(student_ids, num_students, replace=False)
    else:
        student_ids = [f"S{str(i).zfill(id_length)}" for i in range(1, num_students + 1)]
    
    data = []
    for student_id in student_ids:
        grade = np.random.choice(grades)
        school = np.random.choice(schools)
        gender = np.random.choice(["Male", "Female", "Other"], p=[g/100 for g in gender_dist])
        meal_code = np.random.choice(meal_codes)
        transport = np.random.choice(transportation)
        academic = np.random.uniform(academic_perf[0], academic_perf[1])
        suspensions = np.random.randint(suspensions_range[0], suspensions_range[1] + 1)
        present_days = np.random.randint(present_days_range[0], present_days_range[1] + 1)
        max_absent = total_days - present_days
        absent_days = np.random.randint(absent_days_range[0], min(absent_days_range[1] + 1, max_absent + 1))
        attendance_pct = (present_days / total_days) * 100
        
        ca_status = "CA" if attendance_pct < (100 - dropoff_percent) else "Non-CA"
        drop_off = "Y" if ca_status == "CA" else "N"
        if ca_status == "CA" and drop_off_rules:
            if not (drop_off_rules["attendance_min"] <= attendance_pct <= drop_off_rules["attendance_max"]):
                drop_off = "N"
            for feature, values in drop_off_rules["features"].items():
                if feature == "Grade" and grade not in values:
                    drop_off = "N"
                elif feature == "Transportation" and transport not in values:
                    drop_off = "N"
                elif feature == "Meal_Code" and meal_code not in values:
                    drop_off = "N"
                elif feature == "Gender" and gender not in values:
                    drop_off = "N"
                elif feature == "School" and school not in values:
                    drop_off = "N"
                elif feature in [f["name"] for f in custom_fields]:
                    custom_field = next((f for f in custom_fields if f["name"] == feature), None)
                    if custom_field:
                        custom_values = [v.strip() for v in custom_field["values"].split(",")]
                        if custom_field["name"] not in values:
                            drop_off = "N"
        
        record = {
            "Student_ID": student_id,
            "Year": year,
            "Grade": grade,
            "School": school,
            "Gender": gender,
            "Meal_Code": meal_code,
            "Transportation": transport,
            "Academic_Performance": academic,
            "Suspensions": suspensions,
            "Present_Days": present_days,
            "Absent_Days": absent_days,
            "Attendance_Percentage": attendance_pct,
            "CA_Status": ca_status,
            "Drop_Off": drop_off
        }
        
        for name, values in custom_fields:
            record[name] = np.random.choice([v.strip() for v in values.split(",")])
        data.append(record)
    
    return pd.DataFrame(data)

# Model utility functions
def train_model(model_name, X_train, y_train, X_test, y_test, preprocessor):
    if model_name == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    elif model_name == "Logistic Regression":
        model = LogisticRegression(random_state=42)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("estimator", model)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label="CA", zero_division=0),
        "recall": recall_score(y_test, y_pred, pos_label="CA", zero_division=0),
        "f1": f1_score(y_test, y_pred, pos_label="CA", zero_division=0)
    }
    
    return pipeline, {"CA_Status": metrics}

def plot_feature_importance(model, feature_names):
    if hasattr(model.named_steps["estimator"], "feature_importances_"):
        importance = model.named_steps["estimator"].feature_importances_
        df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
        df = df.sort_values("Importance", ascending=True)
        fig = px.bar(df, x="Importance", y="Feature", title="Feature Importance", orientation="h")
        return fig
    return None

# Session state initialization
def clear_session_state():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.datasets = {}
    st.session_state.current_data = None
    st.session_state.models = {}
    st.session_state.custom_fields = []
    st.session_state.current_custom_fields = []
    st.session_state.selected_student_id = None
    st.session_state.model_versions = {}
    st.session_state.patterns = {}
    st.session_state.page = "📝 Data Configuration"
    st.session_state.drop_off_rules = {}
    st.session_state.high_risk_baselines = None

if 'datasets' not in st.session_state:
    clear_session_state()

# Sidebar navigation
st.sidebar.title("Navigation")
page_options = ["📝 Data Configuration", "🤖 Model Training", "📊 Results", "📚 Documentation"]
page = st.sidebar.radio("Go to", page_options, index=page_options.index(st.session_state.page), label_visibility="collapsed")
st.session_state.page = page

if st.sidebar.button("Clear All Data"):
    clear_session_state()
    st.experimental_rerun()

# Compute high-risk baselines
def compute_high_risk_baselines(data):
    high_risk = data[data["CA_Status"] == "CA"]
    if not high_risk.empty:
        return {
            "Attendance_Percentage": high_risk["Attendance_Percentage"].mean(),
            "Academic_Performance": high_risk["Academic_Performance"].mean(),
            "Suspensions": high_risk["Suspensions"].mean(),
            "Transportation": high_risk["Transportation"].mode().iloc[0] if not high_risk["Transportation"].empty else "Unknown"
        }
    return None

# Generate patterns
def generate_patterns(data, dataset_name):
    patterns = []
    high_risk = data[data["CA_Status"] == "CA"]
    if not high_risk.empty:
        patterns.append({
            "pattern": f"Average Attendance: {high_risk['Attendance_Percentage'].mean():.2f}%",
            "explanation": "Low attendance is a strong indicator of chronic absenteeism."
        })
        if not high_risk["Grade"].mode().empty:
            patterns.append({
                "pattern": f"Common Grades: {', '.join(map(str, high_risk['Grade'].mode().tolist()))}",
                "explanation": "Certain grades have higher CA rates."
            })
    return patterns

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
    
    st.header("Manage Historical Datasets")
    data_source = st.radio("Data Source", ["Generate Data", "Upload CSV"])
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload Historical Data (CSV)", type=["csv"])
        dataset_name = st.text_input("Dataset Name", "Uploaded_Dataset")
        if uploaded_file and dataset_name:
            try:
                data = pd.read_csv(uploaded_file)
                data = data.drop_duplicates(subset=["Student_ID", "Year"], keep="first")
                st.session_state.datasets[dataset_name] = data
                st.session_state.patterns[dataset_name] = generate_patterns(data, dataset_name)
                st.success(f"Dataset '{dataset_name}' uploaded successfully!")
            except Exception as e:
                st.error(f"Error uploading data: {str(e)}")
    
    st.subheader("Generate Historical Data")
    num_students = st.slider("Number of Students", 100, 1000, 500)
    year_start, year_end = st.slider("Academic Years", 2020, 2025, (2020, 2024))
    school_prefix = st.text_input("School Prefix", "10U")
    num_schools = st.number_input("Number of Schools", 1, 5, 3)
    id_length = st.radio("Student ID Length", [5, 7], index=0)
    dropoff_percent = st.slider("Target CA Percentage (%)", 5, 50, 20, step=5)
    dataset_name = st.text_input("Generated Dataset Name", f"Historical_{year_start}_{year_end}")
    
    grades = st.multiselect("Grades", list(range(1, 13)), default=[1, 2, 3, 4, 5])
    male_dist = st.slider("Male (%)", 0, 100, 40, step=5)
    female_dist = st.slider("Female (%)", 0, 100, 40, step=5)
    other_dist = st.slider("Other (%)", 0, 100, 20, step=5)
    
    total_dist = male_dist + female_dist + other_dist
    gender_dist_valid = total_dist == 100
    if not gender_dist_valid:
        st.warning(f"Gender distribution must sum to 100%. Current total: {total_dist}%")
    gender_dist = [male_dist, female_dist, other_dist] if gender_dist_valid else None
    
    meal_codes = st.multiselect("Meal Codes", ["Free", "Reduced", "Paid"], default=["Free", "Reduced", "Paid"])
    academic_perf = st.slider("Academic Performance Range (%)", 1, 100, (40, 90))
    academic_perf_valid = academic_perf[0] < academic_perf[1]
    transportation = st.multiselect("Transportation Options", ["Bus", "Walk", "Car"], default=["Bus", "Walk"])
    suspensions_range = st.slider("Suspensions Range", 0, 5, (0, 2))
    suspensions_valid = suspensions_range[0] <= suspensions_range[1]
    
    total_days = 180
    present_days_range = st.slider("Present Days Range", 1, total_days - 1, (120, 179))
    present_days_valid = present_days_range[0] < present_days_range[1] and present_days_range[0] > 0 and present_days_range[1] < total_days
    if not present_days_valid:
        st.warning("Present Days Range must have min > 0 and max < 180, with min < max.")
    
    max_absent_days = total_days - present_days_range[0]
    if max_absent_days <= 0:
        st.error("Maximum absent days must be greater than 0. Adjust Present Days Range.")
        max_absent_days = 1
    
    absent_days_range = st.slider("Absent Days Range", 0, max_absent_days - 1, (0, min(60, max_absent_days - 1)))
    absent_days_valid = absent_days_range[0] <= absent_days_range[1] and absent_days_range[1] < max_absent_days
    if not absent_days_valid:
        st.warning("Absent Days Range must have min <= max and max < max absent days.")
    
    attendance_valid = present_days_valid and absent_days_valid and max_absent_days > 0
    
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
    
    custom_fields_valid = all(f["name"] and f["values"] for f in st.session_state.custom_fields)
    
    st.subheader("Drop Off Rules")
    with st.expander("Define Drop Off Rules"):
        attendance_min = st.slider("Attendance Percentage Min (%)", 0, 100, 0, step=5)
        attendance_max = st.slider("Attendance Percentage Max (%)", 0, 100, 80, step=5)
        drop_off_rules_valid = attendance_min <= attendance_max
        if not drop_off_rules_valid:
            st.warning("Attendance min must be <= max.")
        
        drop_off_features = st.multiselect(
            "Select Features for Drop Off Rules",
            ["Grade", "Transportation", "Meal_Code", "Gender", "School"] + [f["name"] for f in st.session_state.custom_fields if f["name"]]
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
                values_list = [v.strip() for v in custom_field["values"].split(",")] if custom_field else []
                values = st.multiselect(f"Select {feature} Values", values_list, default=values_list, key=f"drop_off_{feature}")
            if values:
                drop_off_rules["features"][feature] = values
    
    generate_disabled = not (gender_dist_valid and attendance_valid and academic_perf_valid and suspensions_valid and drop_off_rules_valid and custom_fields_valid)
    if generate_disabled:
        st.warning("Resolve input issues to enable data generation.")
    
    if st.button("Generate Historical Data", disabled=generate_disabled):
        try:
            custom_fields = [(f["name"], f["values"]) for f in st.session_state.custom_fields if f["name"] and f["values"]]
            st.session_state.drop_off_rules = drop_off_rules
            data = generate_historical_data(
                num_students, year_start, year_end, school_prefix, num_schools, grades,
                gender_dist, meal_codes, academic_perf, transportation, suspensions_range,
                present_days_range, absent_days_range, total_days, custom_fields, id_length,
                dropoff_percent, drop_off_rules
            )
            st.session_state.datasets[dataset_name] = data
            st.session_state.patterns[dataset_name] = generate_patterns(data, dataset_name)
            st.session_state.high_risk_baselines = compute_high_risk_baselines(data)
            st.success(f"Dataset '{dataset_name}' generated successfully!")
            st.dataframe(data.head())
            csv = data.to_csv(index=False)
            st.download_button("Download Historical Data", csv, f"{dataset_name}.csv", "text/csv")
        except Exception as e:
            st.error(f"Error generating data: {str(e)}")
    
    if st.session_state.datasets:
        st.subheader("Available Datasets")
        st.write(list(st.session_state.datasets.keys()))

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
    
    if st.session_state.datasets:
        selected_dataset = st.selectbox("Select Historical Dataset", list(st.session_state.datasets.keys()))
        data = st.session_state.datasets[selected_dataset]
        
        st.subheader("Feature Selection")
        features = st.multiselect(
            "Select Features",
            [col for col in data.columns if col not in ["Student_ID", "CA_Status", "Drop_Off", "Year", "Present_Days", "Absent_Days"]],
            default=["Attendance_Percentage", "Academic_Performance", "Suspensions", "Grade", "Transportation", "Meal_Code"]
        )
        
        categorical_cols = [col for col in features if data[col].dtype == "object" or col == "Grade"]
        numerical_cols = [col for col in features if col not in categorical_cols]
        
        model_options = ["Random Forest", "Logistic Regression"]
        models_to_train = st.multiselect("Select Models", model_options, default=["Random Forest"])
        
        if st.button("Train Models", disabled=not features or not models_to_train):
            try:
                X = data[features]
                y = data["CA_Status"]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                preprocessor = ColumnTransformer(
                    transformers=[
                        ("num", StandardScaler(), numerical_cols),
                        ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_cols)
                    ])
                
                st.session_state.models = {}
                for model_name in models_to_train:
                    with st.spinner(f"Training {model_name}..."):
                        model, metrics = train_model(model_name, X_train, y_train, X_test, y_test, preprocessor)
                        feature_names = numerical_cols + list(preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols))
                        st.session_state.models[model_name] = {
                            "model": model,
                            "metrics": metrics,
                            "feature_names": feature_names,
                            "features": features
                        }
                        st.write(f"**{model_name}** Accuracy: {metrics['CA_Status']['accuracy']:.2f}")
                st.success("All models trained successfully!")
            except Exception as e:
                st.error(f"Error training models: {str(e)}")

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
        num_students = st.slider("Number of Students", 100, 1000, 500)
        school_prefix = st.text_input("School Prefix", "CU")
        num_schools = st.number_input("Number of Schools", 1, 5, 3)
        id_length = st.radio("Student ID Length", [5, 7], index=0)
        dropoff_percent = st.slider("Target CA Percentage (%)", 5, 50, 20, step=5)
        
        grades = st.multiselect("Grades", list(range(1, 13)), default=[1, 2, 3, 4, 5])
        male_dist = st.slider("Male (%)", 0, 100, 40, step=5, key="current_male")
        female_dist = st.slider("Female (%)", 0, 100, 40, step=5, key="current_female")
        other_dist = st.slider("Other (%)", 0, 100, 20, step=5, key="current_other")
        
        gender_dist = [male_dist, female_dist, other_dist] if male_dist + female_dist + other_dist == 100 else None
        meal_codes = st.multiselect("Meal Codes", ["Free", "Reduced", "Paid"], default=["Free", "Reduced", "Paid"], key="current_meal")
        academic_perf = st.slider("Academic Performance Range (%)", 1, 100, (40, 90), key="current_academic")
        academic_perf_valid = academic_perf[0] < academic_perf[1]
        transportation = st.multiselect("Transportation Options", ["Bus", "Walk", "Car"], default=["Bus", "Walk"], key="current_transport")
        suspensions_range = st.slider("Suspensions Range", 0, 5, (0, 2), key="current_suspensions")
        suspensions_valid = suspensions_range[0] <= suspensions_range[1]
        
        total_days = 180
        present_days_range = st.slider("Present Days Range", 1, total_days - 1, (120, 179), key="current_present")
        present_days_valid = present_days_range[0] < present_days_range[1] and present_days_range[0] > 0 and present_days_range[1] < total_days
        
        max_absent_days = total_days - present_days_range[0]
        if max_absent_days <= 0:
            max_absent_days = 1
        absent_days_range = st.slider("Absent Days Range", 0, max_absent_days - 1, (0, min(60, max_absent_days - 1)), key="current_absent")
        absent_days_valid = absent_days_range[0] <= absent_days_range[1] and absent_days_range[1] < max_absent_days
        
        attendance_valid = present_days_valid and absent_days_valid and max_absent_days > 0
        use_historical_ids = st.checkbox("Use Historical Student IDs", value=False, disabled=not st.session_state.datasets)
        include_graduates = st.checkbox("Include Graduating Students", value=False, disabled=not use_historical_ids)
        
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
        
        custom_fields_valid = all(f["name"] and f["values"] for f in st.session_state.current_custom_fields)
        
        st.subheader("Drop Off Rules")
        with st.expander("Define Drop Off Rules"):
            attendance_min = st.slider("Attendance Percentage Min (%)", 0, 100, 0, step=5, key="current_attendance_min")
            attendance_max = st.slider("Attendance Percentage Max (%)", 0, 100, 80, step=5, key="current_attendance_max")
            drop_off_rules_valid = attendance_min <= attendance_max
            
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
                    values_list = [v.strip() for v in custom_field["values"].split(",")] if custom_field else []
                    values = st.multiselect(f"Select {feature} Values", values_list, default=values_list, key=f"current_drop_off_{feature}")
                if values:
                    drop_off_rules["features"][feature] = values
        
        generate_disabled = not (gender_dist and attendance_valid and academic_perf_valid and suspensions_valid and drop_off_rules_valid and custom_fields_valid)
        if st.button("Generate Current Year Data", disabled=generate_disabled):
            try:
                custom_fields = [(f["name"], f["values"]) for f in st.session_state.current_custom_fields if f["name"] and f["values"]]
                historical_data = list(st.session_state.datasets.values())[0] if use_historical_ids and st.session_state.datasets else None
                st.session_state.current_data = generate_current_year_data(
                    num_students, school_prefix, num_schools, grades, gender_dist, meal_codes,
                    academic_perf, transportation, suspensions_range, present_days_range,
                    absent_days_range, total_days, custom_fields, historical_data, id_length,
                    dropoff_percent, include_graduates, drop_off_rules
                )
                st.session_state.drop_off_rules = drop_off_rules
                st.success("Current year data generated successfully!")
                st.dataframe(st.session_state.current_data.head())
                csv = st.session_state.current_data.to_csv(index=False)
                st.download_button("Download Current Year Data", csv, "current_year_data.csv", "text/csv")
            except Exception as e:
                st.error(f"Error generating data: {str(e)}")
    else:
        uploaded_file = st.file_uploader("Upload Current Year Data (CSV)", type=["csv"])
        if uploaded_file:
            try:
                st.session_state.current_data = pd.read_csv(uploaded_file)
                st.success("Data uploaded successfully!")
                st.dataframe(st.session_state.current_data.head())
            except Exception as e:
                st.error(f"Error uploading data: {str(e)}")
    
    if st.session_state.current_data is not None and st.session_state.models:
        st.subheader("Run Predictions")
        selected_model = st.selectbox("Select Model", list(st.session_state.models.keys()))
        model_info = st.session_state.models[selected_model]
        features = model_info["features"]
        
        if st.button("Predict"):
            try:
                X = st.session_state.current_data[features]
                predictions = model_info["model"].predict(X)
                probabilities = model_info["model"].predict_proba(X)
                
                prediction_data = st.session_state.current_data.copy()
                prediction_data["CA_Prediction"] = predictions
                prediction_data["CA_Probability"] = probabilities[:, 1]
                
                def apply_prediction_drop_off_rules(row):
                    if row["CA_Prediction"] != "CA":
                        return "N"
                    attendance = row["Attendance_Percentage"]
                    rules = st.session_state.drop_off_rules
                    if not (rules.get("attendance_min", 0) <= attendance <= rules.get("attendance_max", 100)):
                        return "N"
                    for feature, values in rules.get("features", {}).items():
                        if feature in row and row[feature] not in values:
                            return "N"
                    return "Y"
                
                prediction_data["Drop_Off"] = prediction_data.apply(apply_prediction_drop_off_rules, axis=1)
                
                def identify_causes(row):
                    causes = []
                    if row["CA_Prediction"] == "CA":
                        if row["Attendance_Percentage"] < (st.session_state.high_risk_baselines["Attendance_Percentage"] if st.session_state.high_risk_baselines else 80):
                            causes.append("Low attendance")
                        if row["Suspensions"] > (st.session_state.high_risk_baselines["Suspensions"] if st.session_state.high_risk_baselines else 0):
                            causes.append("High suspensions")
                        if row["Academic_Performance"] < (st.session_state.high_risk_baselines["Academic_Performance"] if st.session_state.high_risk_baselines else 50):
                            causes.append("Low academic performance")
                    return ", ".join(causes) if causes else "None"
                
                prediction_data["Prediction_Causes"] = prediction_data.apply(identify_causes, axis=1)
                st.session_state.current_data = prediction_data
                st.subheader("Prediction Results")
                st.dataframe(prediction_data.head())
                csv = prediction_data.to_csv(index=False)
                st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Error running predictions: {str(e)}")
    
    st.subheader("Single Student Analysis")
    if "CA_Prediction" in st.session_state.current_data.columns:
        student_ids = st.session_state.current_data["Student_ID"].unique()
        selected_id = st.selectbox("Select Student ID", student_ids, index=student_ids.tolist().index(st.session_state.selected_student_id) if st.session_state.selected_student_id in student_ids else 0)
        st.session_state.selected_student_id = selected_id
        
        student_data = st.session_state.current_data[st.session_state.current_data["Student_ID"] == selected_id]
        if not student_data.empty:
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
            
            st.write("**Interventions**")
            interventions = []
            if ca_pred == "CA":
                if "Low attendance" in causes:
                    interventions.append(("Increase attendance monitoring", "📅"))
                if "High suspensions" in causes:
                    interventions.append(("Provide behavioral counseling", "🛡️"))
                if "Low academic performance" in causes:
                    interventions.append(("Offer tutoring support", "📚"))
                interventions.append(("Engage with parents", "📞"))
            
            if interventions:
                for action, icon in interventions:
                    st.markdown(f'<div class="action-item">{icon} {action}</div>', unsafe_allow_html=True)
            else:
                st.info("No specific interventions recommended.")
            
            if st.session_state.high_risk_baselines:
                st.write("**Lagging Areas**")
                lagging = []
                if student_data["Attendance_Percentage"].iloc[0] < st.session_state.high_risk_baselines["Attendance_Percentage"]:
                    lagging.append({
                        "Metric": "Attendance",
                        "Student Value": f"{student_data['Attendance_Percentage'].iloc[0]:.2f}%",
                        "High-Risk Avg": f"{st.session_state.high_risk_baselines['Attendance_Percentage']:.2f}%"
                    })
                if student_data["Academic_Performance"].iloc[0] < st.session_state.high_risk_baselines["Academic_Performance"]:
                    lagging.append({
                        "Metric": "Academic Performance",
                        "Student Value": f"{student_data['Academic_Performance'].iloc[0]:.2f}%",
                        "High-Risk Avg": f"{st.session_state.high_risk_baselines['Academic_Performance']:.2f}%"
                    })
                if student_data["Suspensions"].iloc[0] > st.session_state.high_risk_baselines["Suspensions"]:
                    lagging.append({
                        "Metric": "Suspensions",
                        "Student Value": student_data["Suspensions"].iloc[0],
                        "High-Risk Avg": f"{st.session_state.high_risk_baselines['Suspensions']:.2f}"
                    })
                if lagging:
                    st.dataframe(pd.DataFrame(lagging))
                else:
                    st.info("No lagging areas detected.")

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
    
    st.write("This application predicts chronic absenteeism using machine learning models.")
    if st.session_state.patterns:
        st.subheader("Discovered Patterns")
        for ds, patterns in st.session_state.patterns.items():
            st.write(f"**{ds}**")
            for p in patterns:
                st.write(f"- {p['pattern']}: {p['explanation']}")
    
    if st.session_state.models:
        st.subheader("Model Performance")
        for model_name, info in st.session_state.models.items():
            st.write(f"**{model_name}**")
            metrics = info["metrics"]["CA_Status"]
            st.write(f"Accuracy: {metrics['accuracy']:.2f}, Precision: {metrics['precision']:.2f}, Recall: {metrics['recall']:.2f}, F1: {metrics['f1']:.2f}")
