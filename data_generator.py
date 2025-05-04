import pandas as pd
import numpy as np
import uuid

def generate_historical_data(
    num_students, year_start, year_end, school_prefix, num_schools,
    grades, gender_dist, meal_codes, academic_perf, transportation,
    suspensions_range, present_days_range, absent_days_range, total_days,
    custom_fields, id_length, dropoff_percent, drop_off_rules
):
    """Generate historical student data with target columns."""
    np.random.seed(42)
    data = []
    schools = [f"{school_prefix}{i:03d}" for i in range(1, num_schools + 1)]
    genders = ["Male", "Female", "Other"]
    
    # Validate attendance parameters
    if present_days_range[0] + absent_days_range[1] > total_days:
        raise ValueError("Present days and absent days ranges exceed total days.")
    if present_days_range[0] >= total_days:
        raise ValueError("Minimum present days cannot equal or exceed total days.")
    
    for year in range(year_start, year_end + 1):
        for _ in range(num_students):
            student_id = str(uuid.uuid4())[:id_length]
            grade = np.random.choice(grades)
            school = np.random.choice(schools)
            gender = np.random.choice(genders, p=[g/100 for g in gender_dist])
            meal_code = np.random.choice(meal_codes)
            academic_performance = np.random.uniform(academic_perf[0], academic_perf[1])
            transport = np.random.choice(transportation)
            suspensions = np.random.randint(suspensions_range[0], suspensions_range[1] + 1)
            present_days = np.random.randint(present_days_range[0], present_days_range[1] + 1)
            
            # Ensure valid absent days
            max_possible_absent = total_days - present_days
            if max_possible_absent <= 0:
                raise ValueError(f"Present days ({present_days}) exceeds total days ({total_days}).")
            absent_upper_bound = min(absent_days_range[1] + 1, max_possible_absent + 1)
            if absent_upper_bound <= absent_days_range[0]:
                absent_upper_bound = absent_days_range[0] + 1
            absent_days = np.random.randint(absent_days_range[0], absent_upper_bound)
            attendance_percentage = (present_days / total_days) * 100
            
            # Assign CA_Status based on attendance and dropoff_percent
            ca_threshold = 90  # Chronic absenteeism if attendance < 90%
            is_ca = attendance_percentage < ca_threshold and np.random.random() < (dropoff_percent / 100)
            ca_status = "CA" if is_ca else "Non-CA"
            
            # Assign Drop_Off based on rules
            drop_off = "N"
            if ca_status == "CA":
                if (drop_off_rules["attendance_min"] <= attendance_percentage <= drop_off_rules["attendance_max"]):
                    drop_off = "Y"
                    for feature, values in drop_off_rules.get("features", {}).items():
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
            
            row = {
                "Student_ID": student_id,
                "Year": year,
                "Grade": grade,
                "School": school,
                "Gender": gender,
                "Meal_Code": meal_code,
                "Academic_Performance": academic_performance,
                "Transportation": transport,
                "Suspensions": suspensions,
                "Present_Days": present_days,
                "Absent_Days": absent_days,
                "Attendance_Percentage": attendance_percentage,
                "CA_Status": ca_status,
                "Drop_Off": drop_off
            }
            
            # Add custom fields
            for field_name, field_values in custom_fields:
                if field_values:
                    values = [v.strip() for v in field_values.split(",")]
                    row[field_name] = np.random.choice(values)
            
            data.append(row)
    
    return pd.DataFrame(data)

def generate_current_year_data(
    num_students, school_prefix, num_schools, grades, gender_dist,
    meal_codes, academic_perf, transportation, suspensions_range,
    present_days_range, absent_days_range, total_days, custom_fields,
    historical_ids=None, id_length=5, dropoff_percent=20,
    include_graduates=False, drop_off_rules=None
):
    """Generate current year data with target columns."""
    np.random.seed(42)
    data = []
    schools = [f"{school_prefix}{i:03d}" for i in range(1, num_schools + 1)]
    genders = ["Male", "Female", "Other"]
    
    # Validate attendance parameters
    if present_days_range[0] + absent_days_range[1] > total_days:
        raise ValueError("Present days and absent days ranges exceed total days.")
    if present_days_range[0] >= total_days:
        raise ValueError("Minimum present days cannot equal or exceed total days.")
    
    if historical_ids is not None and include_graduates:
        historical_students = historical_ids[["Student_ID", "Grade"]].drop_duplicates()
        historical_students["Grade"] = historical_students["Grade"].apply(lambda g: min(g + 1, 12))
        student_ids = historical_students["Student_ID"].tolist()
        grades_dict = dict(zip(historical_students["Student_ID"], historical_students["Grade"]))
    else:
        student_ids = [str(uuid.uuid4())[:id_length] for _ in range(num_students)]
    
    for student_id in student_ids:
        grade = grades_dict.get(student_id, np.random.choice(grades)) if historical_ids is not None else np.random.choice(grades)
        school = np.random.choice(schools)
        gender = np.random.choice(genders, p=[g/100 for g in gender_dist])
        meal_code = np.random.choice(meal_codes)
        academic_performance = np.random.uniform(academic_perf[0], academic_perf[1])
        transport = np.random.choice(transportation)
        suspensions = np.random.randint(suspensions_range[0], suspensions_range[1] + 1)
        present_days = np.random.randint(present_days_range[0], present_days_range[1] + 1)
        
        # Ensure valid absent days
        max_possible_absent = total_days - present_days
        if max_possible_absent <= 0:
            raise ValueError(f"Present days ({present_days}) exceeds total days ({total_days}).")
        absent_upper_bound = min(absent_days_range[1] + 1, max_possible_absent + 1)
        if absent_upper_bound <= absent_days_range[0]:
            absent_upper_bound = absent_days_range[0] + 1
        absent_days = np.random.randint(absent_days_range[0], absent_upper_bound)
        attendance_percentage = (present_days / total_days) * 100
        
        # Assign CA_Status based on attendance and dropoff_percent
        ca_threshold = 90
        is_ca = attendance_percentage < ca_threshold and np.random.random() < (dropoff_percent / 100)
        ca_status = "CA" if is_ca else "Non-CA"
        
        # Assign Drop_Off based on rules
        drop_off = "N"
        if ca_status == "CA" and drop_off_rules:
            if (drop_off_rules["attendance_min"] <= attendance_percentage <= drop_off_rules["attendance_max"]):
                drop_off = "Y"
                for feature, values in drop_off_rules.get("features", {}).items():
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
        
        row = {
            "Student_ID": student_id,
            "Grade": grade,
            "School": school,
            "Gender": gender,
            "Meal_Code": meal_code,
            "Academic_Performance": academic_performance,
            "Transportation": transport,
            "Suspensions": suspensions,
            "Present_Days": present_days,
            "Absent_Days": absent_days,
            "Attendance_Percentage": attendance_percentage,
            "CA_Status": ca_status,
            "Drop_Off": drop_off
        }
        
        # Add custom fields
        for field_name, field_values in custom_fields:
            if field_values:
                values = [v.strip() for v in field_values.split(",")]
                row[field_name] = np.random.choice(values)
        
        data.append(row)
    
    return pd.DataFrame(data)
