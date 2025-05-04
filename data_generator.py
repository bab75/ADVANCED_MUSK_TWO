import pandas as pd
import numpy as np
from datetime import datetime

def generate_historical_data(
    num_students, year_start, year_end, school_prefix, num_schools,
    grades, gender_dist, meal_codes, academic_perf, transportation,
    suspensions_range, present_days_range, absent_days_range, total_days, custom_fields
):
    data = []
    years = list(range(year_start, year_end + 1))
    
    for year in years:
        for _ in range(num_students):
            school_id = f"{school_prefix}{np.random.randint(1, num_schools + 1):03d}"
            student_id = f"S{year}{school_id}{np.random.randint(1000, 9999)}"
            
            grade = np.random.choice(grades)
            gender = np.random.choice(["Male", "Female", "Other"], p=[x/100 for x in gender_dist])
            meal_code = np.random.choice(meal_codes)
            academic_performance = np.random.randint(academic_perf[0], academic_perf[1] + 1)
            transport = np.random.choice(transportation)
            suspensions = np.random.randint(suspensions_range[0], suspensions_range[1] + 1)
            
            present_days = np.random.randint(present_days_range[0], present_days_range[1] + 1)
            absent_days = np.random.randint(absent_days_range[0], min(absent_days_range[1] + 1, total_days - present_days))
            attendance_percentage = (present_days / total_days) * 100
            
            ca_status = "CA" if attendance_percentage < 90 else "NO-CA"
            
            row = {
                "Student_ID": student_id,
                "Year": year,
                "School": school_id,
                "Grade": grade,
                "Gender": gender,
                "Meal_Code": meal_code,
                "Academic_Performance": academic_performance,
                "Transportation": transport,
                "Suspensions": suspensions,
                "Present_Days": present_days,
                "Absent_Days": absent_days,
                "Attendance_Percentage": attendance_percentage,
                "CA_Status": ca_status
            }
            
            for field_name, field_values in custom_fields:
                field_values_list = [v.strip() for v in field_values.split(",")]
                row[field_name] = np.random.choice(field_values_list)
            
            data.append(row)
    
    df = pd.DataFrame(data)
    return df

def generate_current_year_data(
    num_students, school_prefix, num_schools, grades, gender_dist,
    meal_codes, academic_perf, transportation, suspensions_range,
    present_days_range, absent_days_range, total_days, custom_fields,
    historical_ids=None
):
    data = []
    current_year = datetime.now().year
    
    for _ in range(num_students):
        if historical_ids and np.random.random() < 0.5 and len(historical_ids) > 0:
            student_id = np.random.choice(historical_ids)
        else:
            school_id = f"{school_prefix}{np.random.randint(1, num_schools + 1):03d}"
            student_id = f"S{current_year}{school_id}{np.random.randint(1000, 9999)}"
        
        grade = np.random.choice(grades)
        gender = np.random.choice(["Male", "Female", "Other"], p=[x/100 for x in gender_dist])
        meal_code = np.random.choice(meal_codes)
        academic_performance = np.random.randint(academic_perf[0], academic_perf[1] + 1)
        transport = np.random.choice(transportation)
        suspensions = np.random.randint(suspensions_range[0], suspensions_range[1] + 1)
        
        present_days = np.random.randint(present_days_range[0], present_days_range[1] + 1)
        absent_days = np.random.randint(absent_days_range[0], min(absent_days_range[1] + 1, total_days - present_days))
        attendance_percentage = (present_days / total_days) * 100
        
        row = {
            "Student_ID": student_id,
            "Year": current_year,
            "School": school_id,
            "Grade": grade,
            "Gender": gender,
            "Meal_Code": meal_code,
            "Academic_Performance": academic_performance,
            "Transportation": transport,
            "Suspensions": suspensions,
            "Present_Days": present_days,
            "Absent_Days": absent_days,
            "Attendance_Percentage": attendance_percentage
        }
        
        for field_name, field_values in custom_fields:
            field_values_list = [v.strip() for v in field_values.split(",")]
            row[field_name] = np.random.choice(field_values_list)
        
        data.append(row)
    
    df = pd.DataFrame(data)
    return df
