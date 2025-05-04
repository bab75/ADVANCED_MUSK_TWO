import pandas as pd
import numpy as np

def calculate_drop_off(data, rules):
    """
    Calculate drop-off probability based on predefined rules.
    
    Args:
        data (pd.DataFrame): Input data containing relevant features.
        rules (dict): Dictionary containing drop-off rules, including:
            - 'attendance_min' (float): Minimum attendance percentage.
            - 'attendance_max' (float): Maximum attendance percentage.
            - 'features' (dict): Dictionary of feature names and their allowed values.
    
    Returns:
        pd.Series: Drop-off probability for each student (between 0 and 1).
    """
    try:
        # Initialize drop-off probability
        drop_off = pd.Series(0.0, index=data.index)
        
        # Check if data contains required columns
        required_cols = ['Attendance_Percentage']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
        
        # Apply attendance-based rules
        attendance_condition = (
            (data['Attendance_Percentage'] >= rules.get('attendance_min', 0)) &
            (data['Attendance_Percentage'] <= rules.get('attendance_max', 100))
        )
        drop_off[attendance_condition] += 0.5  # Base drop-off probability
        
        # Apply feature-based rules
        feature_rules = rules.get('features', {})
        for feature, allowed_values in feature_rules.items():
            if feature not in data.columns:
                continue
            feature_condition = data[feature].isin(allowed_values)
            drop_off[attendance_condition & feature_condition] += 0.3  # Incremental probability
            
            # Adjust for missing or invalid data
            drop_off = drop_off.clip(0, 1)
            
            # Handle edge cases
            if drop_off.isna().any():
                drop_off = drop_off.fillna(0.0)
        
        return drop_off
    except Exception as e:
        raise ValueError(f"Error calculating drop-off: {str(e)}")
