import pandas as pd
import numpy as np
import uuid

def generate_historical_data(num_students, year_start, year_end, school_prefix, num_schools, grades, gender_dist, meal_codes, academic_perf, present_days_range, absent_days_range, total_days, custom_fields):
    data = []
    schools = [f"{school_prefix}{i}" for i in range(1, num_schools + 1)]
    used_ids = set()  # Track used IDs to ensure uniqueness
    
    for _ in range(num_students):
        year = np.random.randint(year_start, year_end + 1)
        school = np.random.choice(schools)
        grade = np.random.choice(grades)
        gender = np.random.choice(["M", "F", "O"], p=np.array(gender_dist)/100)
        meal_code = np.random.choice(meal_codes)
        academic_performance = np.random.randint(academic_perf[0], academic_perf[1] + 1)
        
        # Generate present and absent days within constraints
        present_days = np.random.randint(present_days_range[0], present_days_range[1] + 1)
        max_absent = min(absent_days_range[1], total_days - present_days)
        min_absent = max(absent_days_range[0], total_days - present_days)
        if min_absent > max_absent:
            min_absent = max_absent
        absent_days = np.random.randint(min_absent, max_absent + 1)
        if present_days + absent_days != total_days:
            absent_days = total_days - present_days
        
        attendance_percentage = (present_days / total_days) * 100
        ca_status = "CA" if attendance_percentage <= 90 else "NO-CA"
        
        # Generate unique Student_ID (6-digit number + 'H')
        while True:
            num_id = np.random.randint(0, 1000000)
            student_id = f"{num_id:06d}H"
            if student_id not in used_ids:
                used_ids.add(student_id)
                break
        
        # Base record
        record = {
            "Student_ID": student_id,
            "Year": year,
            "School": school,
            "Grade": grade,
            "Gender": gender,
            "Meal_Code": meal_code,
            "Academic_Performance": academic_performance,
            "Present_Days": present_days,
            "Absent_Days": absent_days,
            "Attendance_Percentage": attendance_percentage,
            "CA_Status": ca_status
        }
        
        # Add custom fields
        for field_name, field_values in custom_fields:
            record[field_name] = np.random.choice(field_values.split(","))
        
        data.append(record)
    
    return pd.DataFrame(data)

def generate_current_year_data(num_students, school_prefix, num_schools, grades, gender_dist, meal_codes, academic_perf, present_days_range, absent_days_range, total_days, custom_fields):
    data = []
    schools = [f"{school_prefix}{i}" for i in range(1, num_schools + 1)]
    current_year = 2025
    
    for i in range(num_students):
        school = np.random.choice(schools)
        grade = np.random.choice(grades)
        gender = np.random.choice(["M", "F", "O"], p=np.array(gender_dist)/100)
        meal_code = np.random.choice(meal_codes)
        academic_performance = np.random.randint(academic_perf[0], academic_perf[1] + 1)
        
        # Generate present and absent days within constraints
        present_days = np.random.randint(present_days_range[0], present_days_range[1] + 1)
        max_absent = min(absent_days_range[1], total_days - present_days)
        min_absent = max(absent_days_range[0], total_days - present_days)
        if min_absent > max_absent:
            min_absent = max_absent
        absent_days = np.random.randint(min_absent, max_absent + 1)
        if present_days + absent_days != total_days:
            absent_days = total_days - present_days
        
        attendance_percentage = (present_days / total_days) * 100
        
        # Generate Student_ID (4-digit zero-padded number + 'C')
        student_id = f"{i:04d}C"
        
        # Base record
        record = {
            "Student_ID": student_id,
            "Year": current_year,
            "School": school,
            "Grade": grade,
            "Gender": gender,
            "Meal_Code": meal_code,
            "Academic_Performance": academic_performance,
            "Present_Days": present_days,
            "Absent_Days": absent_days,
            "Attendance_Percentage": attendance_percentage
        }
        
        # Add custom fields
        for field_name, field_values in custom_fields:
            record[field_name] = np.random.choice(field_values.split(","))
        
        data.append(record)
    
    return pd.DataFrame(data)