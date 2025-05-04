import pandas as pd
import numpy as np
import random
import warnings

def safe_randrange(start, stop):
    """
    Safely generate a random integer in [start, stop).
    If start == stop, return start to avoid empty range error.
    """
    if start == stop:
        warnings.warn(f"Empty range detected in randrange({start}, {stop}). Returning {start}.")
        return start
    return random.randrange(start, stop)

def apply_drop_off_rules(student_data, drop_off_rules):
    """
    Apply drop-off rules to determine Drop_Off status.
    Rules include attendance percentage range and feature conditions.
    Returns 'Y' if all conditions are met and CA_Status='CA', else 'N'.
    """
    if not drop_off_rules or not student_data["CA_Status"] == "CA":
        return "Y" if student_data["CA_Status"] == "CA" else "N"
    
    attendance = student_data["Attendance_Percentage"]
    attendance_min = drop_off_rules.get("attendance_min", 0)
    attendance_max = drop_off_rules.get("attendance_max", 100)
    
    if not (attendance_min <= attendance <= attendance_max):
        return "N"
    
    for feature, values in drop_off_rules.get("features", {}).items():
        if feature in student_data and student_data[feature] not in values:
            return "N"
    
    return "Y"

def generate_historical_data(
    num_students, year_start, year_end, school_prefix, num_schools,
    grades, gender_dist, meal_codes, academic_perf, transportation,
    suspensions_range, present_days_range, absent_days_range, total_days,
    custom_fields, id_length, dropoff_percent, drop_off_rules=None
):
    years = list(range(year_start, year_end + 1))
    schools = [f"{school_prefix}{i:03d}" for i in range(1, num_schools + 1)]
    genders = ["Male", "Female", "Other"]
    gender_probs = [p / 100 for p in gender_dist]
    
    data = []
    student_ids = [f"{'S' + str(i).zfill(id_length - 1)}" for i in range(1, num_students + 1)]
    
    for student_id in student_ids:
        for year in years:
            grade = random.choice(grades)
            school = random.choice(schools)
            gender = np.random.choice(genders, p=gender_probs)
            meal_code = random.choice(meal_codes)
            transport = random.choice(transportation)
            academic_performance = safe_randrange(academic_perf[0], academic_perf[1] + 1)
            suspensions = safe_randrange(suspensions_range[0], suspensions_range[1] + 1)
            
            present_days = safe_randrange(present_days_range[0], present_days_range[1] + 1)
            max_absent_days = total_days - present_days
            if max_absent_days < absent_days_range[0]:
                absent_days = absent_days_range[0]
            elif max_absent_days > absent_days_range[1]:
                absent_days = safe_randrange(absent_days_range[0], absent_days_range[1] + 1)
            else:
                absent_days = safe_randrange(absent_days_range[0], max_absent_days + 1)
            
            attendance_percentage = (present_days / total_days) * 100
            
            ca_threshold = (total_days * (100 - dropoff_percent)) / 100
            ca_status = "CA" if present_days < ca_threshold else "Non-CA"
            
            student_data = {
                "Student_ID": student_id,
                "Year": year,
                "Grade": grade,
                "School": school,
                "Gender": gender,
                "Meal_Code": meal_code,
                "Transportation": transport,
                "Academic_Performance": academic_performance,
                "Suspensions": suspensions,
                "Present_Days": present_days,
                "Absent_Days": absent_days,
                "Attendance_Percentage": attendance_percentage,
                "CA_Status": ca_status
            }
            
            for field_name, field_values in custom_fields:
                values = [v.strip() for v in field_values.split(",")]
                student_data[field_name] = random.choice(values)
            
            student_data["Drop_Off"] = apply_drop_off_rules(student_data, drop_off_rules)
            data.append(student_data)
    
    df = pd.DataFrame(data)
    df = df.sort_values(["Student_ID", "Year"]).reset_index(drop=True)
    return df

def generate_current_year_data(
    num_students, school_prefix, num_schools, grades, gender_dist,
    meal_codes, academic_perf, transportation, suspensions_range,
    present_days_range, absent_days_range, total_days, custom_fields,
    historical_ids=None, id_length=5, dropoff_percent=20, include_graduates=False,
    drop_off_rules=None
):
    schools = [f"{school_prefix}{i:03d}" for i in range(1, num_schools + 1)]
    genders = ["Male", "Female", "Other"]
    gender_probs = [p / 100 for p in gender_dist]
    current_year = max(historical_ids["Year"].max() + 1, 2025) if historical_ids is not None else 2025
    
    data = []
    
    if historical_ids is not None and not historical_ids.empty:
        historical_students = historical_ids[["Student_ID", "Grade", "Year"]].drop_duplicates()
        historical_students = historical_students.sort_values("Year").groupby("Student_ID").last().reset_index()
        
        if not include_graduates:
            historical_students = historical_students[historical_students["Grade"] < 12]
        
        student_ids = historical_students["Student_ID"].tolist()
        if len(student_ids) > num_students:
            student_ids = random.sample(student_ids, num_students)
        elif len(student_ids) < num_students:
            additional_ids = [f"{'S' + str(i).zfill(id_length - 1)}" for i in range(len(student_ids) + 1, num_students + 1)]
            student_ids.extend(additional_ids)
    else:
        student_ids = [f"{'S' + str(i).zfill(id_length - 1)}" for i in range(1, num_students + 1)]
    
    for student_id in student_ids:
        grade = random.choice(grades)
        if historical_ids is not None and student_id in historical_students["Student_ID"].values:
            last_grade = historical_students.loc[historical_students["Student_ID"] == student_id, "Grade"].iloc[0]
            grade = min(last_grade + 1, 12) if last_grade < 12 else 12
        
        school = random.choice(schools)
        gender = np.random.choice(genders, p=gender_probs)
        meal_code = random.choice(meal_codes)
        transport = random.choice(transportation)
        academic_performance = safe_randrange(academic_perf[0], academic_perf[1] + 1)
        suspensions = safe_randrange(suspensions_range[0], suspensions_range[1] + 1)
        
        present_days = safe_randrange(present_days_range[0], present_days_range[1] + 1)
        max_absent_days = total_days - present_days
        if max_absent_days < absent_days_range[0]:
            absent_days = absent_days_range[0]
        elif max_absent_days > absent_days_range[1]:
            absent_days = safe_randrange(absent_days_range[0], absent_days_range[1] + 1)
        else:
            absent_days = safe_randrange(absent_days_range[0], max_absent_days + 1)
        
        attendance_percentage = (present_days / total_days) * 100
        
        ca_threshold = (total_days * (100 - dropoff_percent)) / 100
        ca_status = "CA" if present_days < ca_threshold else "Non-CA"
        
        student_data = {
            "Student_ID": student_id,
            "Year": current_year,
            "Grade": grade,
            "School": school,
            "Gender": gender,
            "Meal_Code": meal_code,
            "Transportation": transport,
            "Academic_Performance": academic_performance,
            "Suspensions": suspensions,
            "Present_Days": present_days,
            "Absent_Days": absent_days,
            "Attendance_Percentage": attendance_percentage,
            "CA_Status": ca_status
        }
        
        for field_name, field_values in custom_fields:
            values = [v.strip() for v in field_values.split(",")]
            student_data[field_name] = random.choice(values)
        
        student_data["Drop_Off"] = apply_drop_off_rules(student_data, drop_off_rules)
        data.append(student_data)
    
    df = pd.DataFrame(data)
    df = df.sort_values("Student_ID").reset_index(drop=True)
    return df
