"""
Model Training Script
Trains a Random Forest model to predict annual salary
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Import preprocessing functions
from utils.preprocessing import preprocess_data

def train_salary_model():
    """
    Train and evaluate the salary prediction model
    """
    print("=" * 60)
    print("SALARY PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    # Load and preprocess data
    print("\n[1/5] Loading and preprocessing data...")
    X, y, scaler, feature_columns = preprocess_data(target_column='Salary')
    
    # Split into train and test sets
    print("\n[2/5] Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train Random Forest model
    print("\n[3/5] Training Random Forest Regressor...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("Model training complete!")
    
    # Make predictions
    print("\n[4/5] Evaluating model performance...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 60)
    print(f"\nTraining Set:")
    print(f"  R² Score: {train_r2:.4f}")
    print(f"  MAE: ${train_mae:,.2f}")
    
    print(f"\nTest Set:")
    print(f"  R² Score: {test_r2:.4f}")
    print(f"  MAE: ${test_mae:,.2f}")
    print(f"  RMSE: ${test_rmse:,.2f}")
    
    # Feature importance
    print("\n" + "=" * 60)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("=" * 60)
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10).to_string(index=False))
    
    # Save model and scaler
    print("\n[5/5] Saving model and scaler...")
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(model, 'models/salary_predictor.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(feature_columns, 'models/feature_columns.pkl')
    
    print("\nModel saved to: models/salary_predictor.pkl")
    print("Scaler saved to: models/scaler.pkl")
    print("Feature columns saved to: models/feature_columns.pkl")
    
    # Create visualization
    create_visualizations(y_test, y_test_pred, feature_importance)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("\nYou can now run the Streamlit app:")
    print("  streamlit run app.py")
    print("=" * 60)

def create_visualizations(y_test, y_test_pred, feature_importance):
    """
    Create and save model evaluation visualizations
    """
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Actual vs Predicted
    axes[0].scatter(y_test, y_test_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Salary ($)', fontsize=12)
    axes[0].set_ylabel('Predicted Salary ($)', fontsize=12)
    axes[0].set_title('Actual vs Predicted Salaries', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Feature Importance (Top 10)
    top_features = feature_importance.head(10)
    axes[1].barh(top_features['feature'], top_features['importance'], color='steelblue')
    axes[1].set_xlabel('Importance', fontsize=12)
    axes[1].set_title('Top 10 Feature Importances', fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('models/model_evaluation.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved to: models/model_evaluation.png")
    plt.close()

if __name__ == "__main__":
    import pandas as pd
    train_salary_model()