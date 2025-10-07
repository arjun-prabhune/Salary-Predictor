"""
Data Preprocessing Module
Handles loading, cleaning, and transforming the salary dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_clean_data(filepath='data/salary_data.csv'):
    """
    Load the salary dataset and perform initial cleaning
    
    Args:
        filepath: Path to the CSV file
    
    Returns:
        Cleaned pandas DataFrame
    """
    # Load data
    df = pd.read_csv(filepath)
    
    # Display basic info
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # Handle missing values
    # For numerical columns, fill with median
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    # For categorical columns, fill with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Remove any duplicate rows
    df.drop_duplicates(inplace=True)
    
    print(f"\nData after cleaning: {df.shape[0]} rows")
    
    return df

def encode_categorical_features(df, categorical_columns):
    """
    Encode categorical variables using one-hot encoding
    
    Args:
        df: Input DataFrame
        categorical_columns: List of column names to encode
    
    Returns:
        DataFrame with encoded features
    """
    # Create a copy to avoid modifying original
    df_encoded = df.copy()
    
    # Apply one-hot encoding
    df_encoded = pd.get_dummies(df_encoded, columns=categorical_columns, drop_first=True)
    
    return df_encoded

def preprocess_data(filepath='data/salary_data.csv', target_column='Annual Salary'):
    """
    Complete preprocessing pipeline
    
    Args:
        filepath: Path to CSV file
        target_column: Name of the target variable column
    
    Returns:
        X: Feature matrix
        y: Target vector
        scaler: Fitted StandardScaler object
        feature_columns: List of feature column names (for future predictions)
    """
    # Load and clean data
    df = load_and_clean_data(filepath)
    
    # Separate features and target
    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    # Identify categorical columns (excluding target)
    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\nCategorical columns to encode: {categorical_columns}")
    
    # Encode categorical features
    X_encoded = encode_categorical_features(X, categorical_columns)
    
    # Store feature columns for later use
    feature_columns = X_encoded.columns.tolist()
    
    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    
    # Convert back to DataFrame for easier handling
    X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)
    
    print(f"\nFinal feature matrix shape: {X_scaled.shape}")
    print(f"Features: {feature_columns}")
    
    return X_scaled, y, scaler, feature_columns

def prepare_user_input(user_data, feature_columns, scaler):
    """
    Prepare user input for prediction
    
    Args:
        user_data: Dictionary with user input values
        feature_columns: List of feature names from training
        scaler: Fitted StandardScaler object
    
    Returns:
        Scaled feature array ready for prediction
    """
    # Create DataFrame from user input
    input_df = pd.DataFrame([user_data])
    
    # Get categorical columns
    categorical_columns = input_df.select_dtypes(include=['object']).columns.tolist()
    
    # Encode categorical features
    input_encoded = pd.get_dummies(input_df, columns=categorical_columns, drop_first=True)
    
    # Ensure all training features are present
    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    # Reorder columns to match training data
    input_encoded = input_encoded[feature_columns]
    
    # Scale features
    input_scaled = scaler.transform(input_encoded)
    
    return input_scaled