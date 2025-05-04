import pandas as pd
import numpy as np

def generate_historical_data(num_students, year_start, year_end, school_prefix, num_schools, grades, gender_dist, meal_codes, academic_perf, transportation, suspensions_range, present_days_range, absent_days_range, total_days, custom_fields):
    data = []
    for year in range(year_start, year_end + 1):
        for _ in range(num_students):
            school_id = f"{school_prefix}{np.random.randint(1, num_schools + 1):03d}"
            student_id = f"{school_id}-{year}-{np.random.randint(1000, 9999)}"
            grade = np.random.choice(grades)
            gender = np.random.choice(["Male", "Female", "Other"], p=np.array(gender_dist) / 100)
            meal_code = np.random.choice(meal_codes)
            academic_performance = np.random.randint(academic_perf[0], academic_perf[1] + 1)
            transport = np.random.choice(transportation)
            suspensions = np.random.randint(suspensions_range[0], suspensions_range[1] + 1)
            
            # Ensure present_days allows for at least absent_days_range[0]
            max_present_days = min(present_days_range[1], total_days - absent_days_range[0])
            present_days = np.random.randint(present_days_range[0], max_present_days + 1)
            absent_days = np.random.randint(absent_days_range[0], min(absent_days_range[1] + 1, total_days - present_days))
            attendance_percentage = (present_days / total_days) * 100
            ca_status = "CA" if attendance_percentage < 90 else "NO-CA"
            
            student_data = {
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
                values = [v.strip() for v in field_values.split(",")]
                student_data[field_name] = np.random.choice(values)
            
            data.append(student_data)
    
    return pd.DataFrame(data)

def generate_current_year_data(num_students, school_prefix, num_schools, grades, gender_dist, meal_codes, academic_perf, transportation, suspensions_range, present_days_range, absent_days_range, total_days, custom_fields, historical_ids=None):
    data = []
    used_ids = set()
    
    for _ in range(num_students):
        if historical_ids and len(historical_ids) > 0:
            student_id = np.random.choice(historical_ids)
            while student_id in used_ids and len(used_ids) < len(historical_ids):
                student_id = np.random.choice(historical_ids)
            used_ids.add(student_id)
        else:
            school_id = f"{school_prefix}{np.random.randint(1, num_schools + 1):03d}"
            student_id = f"{school_id}-CY-{np.random.randint(1000, 9999)}"
        
        grade = np.random.choice(grades)
        gender = np.random.choice(["Male", "Female", "Other"], p=np.array(gender_dist) / 100)
        meal_code = np.random.choice(meal_codes)
        academic_performance = np.random.randint(academic_perf[0], academic_perf[1] + 1)
        transport = np.random.choice(transportation)
        suspensions = np.random.randint(suspensions_range[0], suspensions_range[1] + 1)
        
        # Ensure present_days allows for at least absent_days_range[0]
        max_present_days = min(present_days_range[1], total_days - absent_days_range[0])
        present_days = np.random.randint(present_days_range[0], max_present_days + 1)
        absent_days = np.random.randint(absent_days_range[0], min(absent_days_range[1] + 1, total_days - present_days))
        attendance_percentage = (present_days / total_days) * 100
        
        student_data = {
            "Student_ID": student_id,
            "School": school_id if not historical_ids else student_id.split("-")[0],
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
            values = [v.strip() for v in field_values.split(",")]
            student_data[field_name] = np.random.choice(values)
        
        data.append(student_data)
    
    return pd.DataFrame(data)
