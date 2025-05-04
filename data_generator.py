import pandas as pd
import numpy as np
import random
import uuid

def generate_historical_data(
    num_students, year_start, year_end, school_prefix, num_schools,
    grades, gender_dist, meal_codes, academic_perf, transportation,
    suspensions_range, present_days_range, absent_days_range, total_days,
    custom_fields
):
    data = []
    years = list(range(year_start, year_end + 1))
    
    for _ in range(num_students):
        for year in years:
            school_id = f"{school_prefix}{random.randint(1, num_schools)}"
            grade = random.choice(grades)
            
            gender_probs = [p / 100 for p in gender_dist]
            gender = np.random.choice(["Male", "Female", "Other"], p=gender_probs)
            
            meal_code = random.choice(meal_codes)
            academic_performance = random.uniform(academic_perf[0], academic_perf[1])
            transport = random.choice(transportation)
            suspensions = random.randint(suspensions_range[0], suspensions_range[1])
            
            present_days = random.randint(present_days_range[0], present_days_range[1])
            max_absent_days = total_days - present_days
            absent_days = random.randint(
                max(0, absent_days_range[0]),
                min(absent_days_range[1], max_absent_days)
            )
            
            attendance_percentage = (present_days / total_days) * 100
            ca_status = "CA" if absent_days / total_days >= 0.1 else "Non-CA"
            
            student_id = str(uuid.uuid4())
            
            record = {
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
                values = field_values.split(",")
                record[field_name] = random.choice([v.strip() for v in values])
            
            data.append(record)
    
    return pd.DataFrame(data)

def generate_current_year_data(
    num_students, school_prefix, num_schools, grades, gender_dist,
    meal_codes, academic_perf, transportation, suspensions_range,
    present_days_range, absent_days_range, total_days, custom_fields,
    historical_ids=None
):
    data = []
    year = max(2025, present_days_range[1] // 180 + 2020)  # Estimate current year
    
    for i in range(num_students):
        school_id = f"{school_prefix}{random.randint(1, num_schools)}"
        grade = random.choice(grades)
        
        gender_probs = [p / 100 for p in gender_dist]
        gender = np.random.choice(["Male", "Female", "Other"], p=gender_probs)
        
        meal_code = random.choice(meal_codes)
        academic_performance = random.uniform(academic_perf[0], academic_perf[1])
        transport = random.choice(transportation)
        suspensions = random.randint(suspensions_range[0], suspensions_range[1])
        
        present_days = random.randint(present_days_range[0], present_days_range[1])
        max_absent_days = total_days - present_days
        absent_days = random.randint(
            max(0, absent_days_range[0]),
            min(absent_days_range[1], max_absent_days)
        )
        
        attendance_percentage = (present_days / total_days) * 100
        
        # Use historical IDs if provided, otherwise generate new ones
        if historical_ids and len(historical_ids) > i:
            student_id = historical_ids[i]
        else:
            student_id = str(uuid.uuid4())
        
        record = {
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
            "Attendance_Percentage": attendance_percentage
        }
        
        for field_name, field_values in custom_fields:
            values = field_values.split(",")
            record[field_name] = random.choice([v.strip() for v in values])
        
        data.append(record)
    
    return pd.DataFrame(data)
