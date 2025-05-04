import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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

def identify_patterns(data, features, target='CA_Status'):
    """
    Identify patterns in the data that correlate with the target variable.
    
    Args:
        data (pd.DataFrame): Input data containing features and target.
        features (list): List of feature names to analyze.
        target (str): Target column name (default: 'CA_Status').
    
    Returns:
        list: List of dictionaries, each containing a pattern and its explanation.
    """
    try:
        patterns = []
        
        # Check for required columns
        required_cols = features + [target]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
        
        # Analyze numerical features
        numerical_features = [f for f in features if data[f].dtype in ['int64', 'float64']]
        for feature in numerical_features:
            # Calculate correlation with target (assuming binary target)
            if data[target].dtype in ['object', 'category']:
                group_means = data.groupby(target)[feature].mean()
                if len(group_means) == 2:  # Binary target
                    diff = abs(group_means.iloc[0] - group_means.iloc[1])
                    if diff > data[feature].std() / 2:  # Significant difference
                        pattern = {
                            'pattern': f"High {feature} variation between {target} groups",
                            'explanation': f"Students with {target} = {group_means.index[1]} have {feature} "
                                         f"mean {group_means.iloc[1]:.2f}, compared to {group_means.iloc[0]:.2f} "
                                         f"for {group_means.index[0]}."
                        }
                        patterns.append(pattern)
        
        # Analyze categorical features
        categorical_features = [f for f in features if data[f].dtype == 'object' or data[f].dtype.name == 'category']
        for feature in categorical_features:
            # Cross-tabulation to find strong associations
            crosstab = pd.crosstab(data[feature], data[target], normalize='index')
            for category in crosstab.index:
                max_prob = crosstab.loc[category].max()
                if max_prob > 0.7:  # Strong association
                    target_value = crosstab.loc[category].idxmax()
                    pattern = {
                        'pattern': f"{feature} = {category} strongly predicts {target} = {target_value}",
                        'explanation': f"{max_prob*100:.1f}% of students with {feature} = {category} "
                                     f"have {target} = {target_value}."
                    }
                    patterns.append(pattern)
        
        return patterns
    except Exception as e:
        raise ValueError(f"Error identifying patterns: {str(e)}")

def explain_prediction(model, X, feature_names, instance_index=0):
    """
    Explain a single prediction by identifying key feature contributions.
    
    Args:
        model: Trained model (e.g., RandomForestClassifier, LogisticRegression).
        X (pd.DataFrame or np.ndarray): Feature matrix.
        feature_names (list): List of feature names.
        instance_index (int): Index of the instance to explain (default: 0).
    
    Returns:
        dict: Dictionary with prediction, probability, and feature contributions.
    """
    try:
        # Convert X to DataFrame if it's a numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=feature_names)
        
        # Get the instance
        instance = X.iloc[instance_index:instance_index+1]
        
        # Get prediction and probability
        prediction = model.predict(instance)[0]
        proba = model.predict_proba(instance)[0]
        
        # Initialize explanation
        explanation = {
            'prediction': prediction,
            'probability': proba.max(),
            'feature_contributions': {}
        }
        
        # Feature contributions based on model type
        if isinstance(model, RandomForestClassifier):
            # Use feature importances scaled by instance values
            importances = model.feature_importances_
            for feature, importance, value in zip(feature_names, importances, instance.iloc[0]):
                explanation['feature_contributions'][feature] = {
                    'importance': importance,
                    'value': value,
                    'contribution': importance * value  # Simplified contribution
                }
        elif isinstance(model, LogisticRegression):
            # Use coefficients scaled by instance values
            coef = model.coef_[0]
            for feature, coef_val, value in zip(feature_names, coef, instance.iloc[0]):
                explanation['feature_contributions'][feature] = {
                    'coefficient': coef_val,
                    'value': value,
                    'contribution': coef_val * value
                }
        else:
            # Fallback for other models
            for feature, value in zip(feature_names, instance.iloc[0]):
                explanation['feature_contributions'][feature] = {
                    'value': value,
                    'contribution': 'Not calculated (unsupported model type)'
                }
        
        return explanation
    except Exception as e:
        raise ValueError(f"Error explaining prediction: {str(e)}")
