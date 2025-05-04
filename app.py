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

# Sidebar navigation with SVG icons
st.sidebar.title("Navigation")
#st.sidebar.markdown('<div class="nav-icon"><svg>...</svg>Data Preparation</div>', unsafe_allow_html=True)
#st.sidebar.markdown('<div class="nav-icon"><svg>...</svg>Model Training</div>', unsafe_allow_html=True)
#st.sidebar.markdown('<div class="nav-icon"><svg>...</svg>Predictions</div>', unsafe_allow_html=True)
#st.sidebar.markdown('<div class="nav-icon"><svg>...</svg>Advanced Analysis</div>', unsafe_allow_html=True)
page = st.sidebar.radio("Go to", ["Data Preparation", "Model Training", "Predictions", "Advanced Analysis"], label_visibility="collapsed")

# Page 1: Data Preparation
if page == "Data Preparation":
    st.title("Data Preparation")
    
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
            custom_fields = [(f["name"], f["values"]) for f in st.session_state.custom_fields if f["name"] and f["values"]]
            data = generate_historical_data(
                num_students, year_start, year_end, school_prefix, num_schools,
                grades, gender_dist, meal_codes, academic_perf,
                present_days_range, absent_days_range, total_days, custom_fields
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
elif page == "Model Training":
    st.title("Model Training")
    
    # Data Upload or Use Generated
    st.header("Load Data")
    data_source = st.radio("Data Source", ["Use Generated Data", "Upload CSV"])
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload Historical Data (CSV)", type=["csv"])
        if uploaded_file:
            try:
                st.session_state.data = pd.read_csv(uploaded_file)
                st.success("Data uploaded successfully!")
            except Exception as e:
                st.error(f"Error uploading data: {str(e)}")
    
    if st.session_state.data is not None:
        st.subheader("Feature Selection")
        # Exclude Student_ID from features
        available_features = [col for col in st.session_state.data.columns if col not in ["Student_ID", "CA_Status"]]
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
                            if enable_tuning:
                                model, metrics, best_params = tune_model(model_name, X_train_processed, y_train, X_test_processed, y_test)
                            else:
                                model, metrics = train_model(model_name, X_train_processed, y_train, X_test_processed, y_test)
                                best_params = None
                            
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
                                "Accuracy": metrics['accuracy'],
                                "Precision": metrics['precision'],
                                "Recall": metrics['recall'],
                                "F1 Score": metrics['f1'],
                                "ROC AUC": metrics['roc_auc']
                            })
                            
                            status_container.success(f"{model_name} trained successfully!")
                        except Exception as e:
                            status_container.error(f"Error training {model_name}: {str(e)}")
                    
                    # Side-by-Side Model Comparison
                    if comparison_data:
                        st.subheader("Model Comparison")
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df)
                        
                        # Comparison Plot
                        fig = go.Figure()
                        for metric in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]:
                            fig.add_trace(go.Bar(
                                x=comparison_df["Model"],
                                y=comparison_df[metric],
                                name=metric
                            ))
                        fig.update_layout(
                            title="Model Performance Comparison",
                            barmode="group",
                            xaxis_title="Model",
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
                                fig = plot_feature_importance(model_info["model"], feature_names)
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
                # Example patterns
                low_attendance = high_risk["Attendance_Percentage"].mean()
                common_grades = high_risk["Grade"].mode().tolist()
                common_meal_codes = high_risk["Meal_Code"].mode().tolist()
                patterns.append(f"Average Attendance: {low_attendance:.2f}%")
                patterns.append(f"Common Grades: {', '.join(map(str, common_grades))}")
                patterns.append(f"Common Meal Codes: {', '.join(common_meal_codes)}")
                st.write(f"Number of patterns learned: {len(patterns)}")
                for pattern in patterns:
                    st.write(f"- {pattern}")

# Page 3: Predictions
elif page == "Predictions":
    st.title("Chronic Absenteeism Predictions")
    
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
                    meal_codes, academic_perf, present_days_range, absent_days_range,
                    total_days, custom_fields
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
        # Dynamic Feature Toggling
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
                fig = px.imshow(heatmap_data, title="CA Probability Heatmap by Grade and School", labels={"color": "CA Probability"})
                st.plotly_chart(fig)
                
                # Single Student Analysis
                st.subheader("Single Student Analysis")
                student_id = st.selectbox("Select Student ID", st.session_state.current_data["Student_ID"])
                student_data = st.session_state.current_data[st.session_state.current_data["Student_ID"] == student_id]
                if not student_data.empty:
                    st.write("Student Details:")
                    st.dataframe(student_data)
                    ca_prob = student_data["CA_Probability"].iloc[0]
                    ca_pred = student_data["CA_Prediction"].iloc[0]
                    st.write(f"CA Prediction: {ca_pred}")
                    st.write(f"CA Probability: {ca_prob:.2f}")
                    st.write("Preventive Actions:")
                    if ca_pred == "CA":
                        st.write("- Increase attendance monitoring.")
                        st.write("- Provide academic support or tutoring.")
                        st.write("- Engage with parents to address absence causes.")
                
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

# Page 4: Advanced Analysis
elif page == "Advanced Analysis":
    st.title("Advanced Analysis")
    
    if st.session_state.data is not None:
        st.header("Correlation Analysis")
        numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns
        corr = st.session_state.data[numeric_cols].corr()
        fig = px.imshow(corr, title="Feature Correlation Heatmap")
        st.plotly_chart(fig)
        
        st.header("Attendance vs. Academic Performance")
        fig = px.scatter(st.session_state.data, x="Attendance_Percentage", 
                        y="Academic_Performance", color="CA_Status", 
                        title="Attendance vs. Academic Performance")
        st.plotly_chart(fig)
        
        # Intervention Recommendations
        st.header("Intervention Recommendations")
        high_risk = st.session_state.data[st.session_state.data["CA_Status"] == "CA"]
        if not high_risk.empty:
            st.write("High-risk students (CA Status = CA):")
            st.dataframe(high_risk[["Student_ID", "Attendance_Percentage", "Academic_Performance"]])
            st.write("Recommended Actions:")
            st.write("- **Tutoring Programs**: For students with low academic performance.")
            st.write("- **Counseling Services**: For students with high absence rates.")
            st.write("- **Parental Engagement**: Contact families of high-risk students.")
