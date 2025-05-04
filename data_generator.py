import pandas as pd
import numpy as np
from datetime import datetime

def generate_historical_data(
    num_students, year_start, year_end, school_prefix, num_schools,
    grades, gender_dist, meal_codes, academic_perf, transportation,
    suspensions_range, present_days_range, absent_days_range, total_days,
    custom_fields, id_length, dropoff_percent, drop_off_rules
):
    """Generate historical student data."""
    try:
        np.random.seed(42)
        data = []
        
        for year in range(year_start, year_end + 1):
            for school_num in range(1, num_schools + 1):
                school_students = num_students // num_schools
                for student_num in range(1, school_students + 1):
                    # Generate Student_ID with 'C' prefix
                    student_id = f"C{student_num:0{id_length}d}"
                    
                    grade = np.random.choice(grades)
                    gender = np.random.choice(["Male", "Female", "Other"], p=[g/100 for g in gender_dist])
                    meal_code = np.random.choice(meal_codes)
                    academic_performance = np.random.uniform(academic_perf[0], academic_perf[1])
                    transport = np.random.choice(transportation)
                    suspensions = np.random.randint(suspensions_range[0], suspensions_range[1] + 1)
                    
                    present_days = np.random.randint(present_days_range[0], present_days_range[1] + 1)
                    max_absent_days = total_days - present_days
                    absent_days = np.random.randint(
                        absent_days_range[0],
                        min(absent_days_range[1] + 1, max_absent_days + 1)
                    )
                    attendance_percentage = (present_days / total_days) * 100
                    
                    student_data = {
                        "Student_ID": student_id,
                        "Year": year,
                        "School": f"{school_prefix}{school_num:03d}",
                        "Grade": grade,
                        "Gender": gender,
                        "Meal_Code": meal_code,
                        "Academic_Performance": academic_performance,
                        "Transportation": transport,
                        "Suspensions": suspensions,
                        "Present_Days": present_days,
                        "Absent_Days": absent_days,
                        "Attendance_Percentage": attendance_percentage,
                        "CA_Status": "CA" if attendance_percentage < (1 - dropoff_percent / 100) * 100 else "Non-CA"
                    }
                    
                    # Add custom fields
                    for field_name, field_values in custom_fields:
                        if field_values:
                            values = [v.strip() for v in field_values.split(",")]
                            student_data[field_name] = np.random.choice(values)
                    
                    data.append(student_data)
        
        df = pd.DataFrame(data)
        
        # Apply drop-off rules
        if drop_off_rules and "features" in drop_off_rules:
            df["Drop_Off"] = False
            for idx, row in df.iterrows():
                attendance = row["Attendance_Percentage"]
                if drop_off_rules["attendance_min"] <= attendance <= drop_off_rules["attendance_max"]:
                    for feature, values in drop_off_rules["features"].items():
                        if feature in row and row[feature] in values:
                            df.at[idx, "Drop_Off"] = True
                            break
        
        return df
    except Exception as e:
        raise ValueError(f"Error generating historical data: {str(e)}")

def generate_current_year_data(
    num_students, school_prefix, num_schools, grades, gender_dist,
    meal_codes, academic_perf, transportation, suspensions_range,
    present_days_range, absent_days_range, total_days, custom_fields,
    historical_ids=None, id_length=5, dropoff_percent=20, include_graduates=False,
    drop_off_rules=None
):
    """Generate current year student data."""
    try:
        np.random.seed(42)
        data = []
        existing_ids = set(historical_ids["Student_ID"].astype(str)) if historical_ids is not None else set()
        
        for school_num in range(1, num_schools + 1):
            school_students = num_students // num_schools
            for student_num in range(1, school_students + 1):
                if historical_ids is not None and existing_ids:
                    student_id = np.random.choice(list(existing_ids))
                    existing_ids.remove(student_id)
                    student_data = historical_ids[historical_ids["Student_ID"] == student_id].iloc[0].to_dict()
                    grade = min(student_data["Grade"] + 1, 12) if include_graduates else student_data["Grade"]
                    if grade > 12:
                        continue
                else:
                    # Generate Student_ID with 'C' prefix
                    student_id = f"C{student_num:0{id_length}d}"
                    grade = np.random.choice(grades)
                
                gender = np.random.choice(["Male", "Female", "Other"], p=[g/100 for g in gender_dist])
                meal_code = np.random.choice(meal_codes)
                academic_performance = np.random.uniform(academic_perf[0], academic_perf[1])
                transport = np.random.choice(transportation)
                suspensions = np.random.randint(suspensions_range[0], suspensions_range[1] + 1)
                
                present_days = np.random.randint(present_days_range[0], present_days_range[1] + 1)
                max_absent_days = total_days - present_days
                absent_days = np.random.randint(
                    absent_days_range[0],
                    min(absent_days_range[1] + 1, max_absent_days + 1)
                )
                attendance_percentage = (present_days / total_days) * 100
                
                student_data = {
                    "Student_ID": student_id,
                    "Year": datetime.now().year,
                    "School": f"{school_prefix}{school_num:03d}",
                    "Grade": grade,
                    "Gender": gender,
                    "Meal_Code": meal_code,
                    "Academic_Performance": academic_performance,
                    "Transportation": transport,
                    "Suspensions": suspensions,
                    "Present_Days": present_days,
                    "Absent_Days": absent_days,
                    "Attendance_Percentage": attendance_percentage,
                    "CA_Status": "CA" if attendance_percentage < (1 - dropoff_percent / 100) * 100 else "Non-CA"
                }
                
                # Add custom fields
                for field_name, field_values in custom_fields:
                    if field_values:
                        values = [v.strip() for v in field_values.split(",")]
                        student_data[field_name] = np.random.choice(values)
                
                data.append(student_data)
        
        df = pd.DataFrame(data)
        
        # Apply drop-off rules
        if drop_off_rules and "features" in drop_off_rules:
            df["Drop_Off"] = False
            for idx, row in df.iterrows():
                attendance = row["Attendance_Percentage"]
                if drop_off_rules["attendance_min"] <= attendance <= drop_off_rules["attendance_max"]:
                    for feature, values in drop_off_rules["features"].items():
                        if feature in row and row[feature] in values:
                            df.at[idx, "Drop_Off"] = True
                            break
        
        return df
    except Exception as e:
        raise ValueError(f"Error generating current year data: {str(e)}")
