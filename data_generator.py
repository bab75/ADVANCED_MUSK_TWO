import pandas as pd
import numpy as np
import random

def generate_historical_data(
    num_students, year_start, year_end, school_prefix, num_schools,
    grades, gender_dist, meal_codes, academic_perf, transportation,
    suspensions_range, present_days_range, absent_days_range, total_days,
    custom_fields, id_length, dropoff_percent
):
    data = []
    years = list(range(year_start, year_end + 1))
    used_ids = set()
    
    def generate_student_id():
        min_id = 10 ** (id_length - 1)
        max_id = (10 ** id_length) - 1
        while True:
            student_id = random.randint(min_id, max_id)
            if student_id not in used_ids:
                used_ids.add(student_id)
                return str(student_id)
    
    target_ca_count = int(num_students * len(years) * (dropoff_percent / 100))
    ca_indices = random.sample(range(num_students * len(years)), target_ca_count)
    
    record_index = 0
    for student_idx in range(num_students):
        student_id = generate_student_id()
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
            
            is_ca = record_index in ca_indices
            if is_ca:
                absent_days = max(absent_days, int(total_days * 0.1))
                present_days = total_days - absent_days
                attendance_percentage = (present_days / total_days) * 100
            ca_status = "CA" if is_ca else "Non-CA"
            
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
            record_index += 1
    
    return pd.DataFrame(data)

def generate_current_year_data(
    num_students, school_prefix, num_schools, grades, gender_dist,
    meal_codes, academic_perf, transportation, suspensions_range,
    present_days_range, absent_days_range, total_days, custom_fields,
    historical_ids=None, id_length=5, dropoff_percent=20, include_graduates=False
):
    data = []
    year = 2025
    used_ids = set()
    
    def generate_student_id():
        min_id = 10 ** (id_length - 1)
        max_id = (10 ** id_length) - 1
        while True:
            student_id = random.randint(min_id, max_id)
            if student_id not in used_ids:
                used_ids.add(student_id)
                return str(student_id)
    
    # Process historical IDs for grade progression
    historical_students = []
    if historical_ids and isinstance(historical_ids, pd.DataFrame):
        # Deduplicate historical data by selecting the most recent record per student
        historical_data = historical_ids.sort_values(by=["Student_ID", "Year", "Grade"], ascending=[True, False, False])
        historical_data = historical_data.drop_duplicates(subset=["Student_ID"], keep="first")
        
        for _, row in historical_data.iterrows():
            student_id = row["Student_ID"]
            last_grade = row["Grade"]
            last_year = row["Year"]
            
            # Calculate expected grade for 2025
            years_diff = year - last_year
            expected_grade = last_grade + years_diff
            
            # Handle grade progression
            if expected_grade <= max(grades):
                if include_graduates or expected_grade in grades:
                    historical_students.append({
                        "Student_ID": student_id,
                        "Grade": expected_grade if expected_grade in grades else random.choice(grades),
                        # Preserve custom fields from historical data if available
                        **{field: row[field] for field, _ in custom_fields if field in historical_data.columns}
                    })
            elif include_graduates and max(grades) in grades:
                historical_students.append({
                    "Student_ID": student_id,
                    "Grade": max(grades),  # Cap at maximum grade
                    **{field: row[field] for field, _ in custom_fields if field in historical_data.columns}
                })
    
    # Shuffle historical students to randomize selection
    random.shuffle(historical_students)
    num_historical = min(len(historical_students), num_students)
    remaining_students = num_students - num_historical
    
    target_ca_count = int(num_students * (dropoff_percent / 100))
    ca_indices = random.sample(range(num_students), target_ca_count)
    
    # Generate records for historical students
    for i in range(num_historical):
        student = historical_students[i]
        student_id = student["Student_ID"]
        grade = student["Grade"]
        
        school_id = f"{school_prefix}{random.randint(1, num_schools)}"
        
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
        
        is_ca = i in ca_indices
        if is_ca:
            absent_days = max(absent_days, int(total_days * 0.1))
            present_days = total_days - absent_days
            attendance_percentage = (present_days / total_days) * 100
        
        used_ids.add(student_id)
        
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
        
        # Use historical custom field values if available, else generate new
        for field_name, field_values in custom_fields:
            if field_name in student:
                record[field_name] = student[field_name]
            else:
                values = field_values.split(",")
                record[field_name] = random.choice([v.strip() for v in values])
        
        data.append(record)
    
    # Generate records for new students
    for i in range(num_historical, num_students):
        student_id = generate_student_id()
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
        
        is_ca = i in ca_indices
        if is_ca:
            absent_days = max(absent_days, int(total_days * 0.1))
            present_days = total_days - absent_days
            attendance_percentage = (present_days / total_days) * 100
        
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
