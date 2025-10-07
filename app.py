"""
Streamlit App for Salary Prediction
Interactive interface for predicting future salary
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.preprocessing import prepare_user_input

# Page configuration
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💰",
    layout="centered"
)

# Load model, scaler, and feature columns
@st.cache_resource
def load_model_artifacts():
    """Load saved model, scaler, and feature columns"""
    try:
        model = joblib.load('models/salary_predictor.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_columns = joblib.load('models/feature_columns.pkl')
        return model, scaler, feature_columns
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please run `python train_model.py` first.")
        st.stop()

model, scaler, feature_columns = load_model_artifacts()

def predict_future_salary(user_data, years_ahead=5):
    """
    Predict salary for current and future years
    
    Args:
        user_data: Dictionary with user input
        years_ahead: Number of years to project into the future
    
    Returns:
        current_prediction: Predicted current salary
        future_prediction: Predicted salary after years_ahead
    """
    # Predict current salary
    current_input = prepare_user_input(user_data, feature_columns, scaler)
    current_prediction = model.predict(current_input)[0]
    
    # Predict future salary (add years to experience)
    future_data = user_data.copy()
    future_data['Years of Experience'] += years_ahead
    future_data['Age'] += years_ahead
    
    future_input = prepare_user_input(future_data, feature_columns, scaler)
    future_prediction = model.predict(future_input)[0]
    
    return current_prediction, future_prediction

# App Title
st.title("💰 Salary Prediction System")
st.markdown("### Predict Your Salary 5 Years Into The Future")
st.markdown("---")

# Sidebar with instructions
with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    1. Enter your personal and professional details
    2. Optionally provide your current salary for comparison
    3. Click **Predict Future Salary** to see results
    4. View visualization of salary projection
    """)
    st.markdown("---")
    st.markdown("**Model Info:**")
    st.info("This model uses Random Forest regression trained on historical salary data.")

# Input Form
st.subheader("📝 Enter Your Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=80,
        value=30,
        help="Your current age"
    )
    
    gender = st.selectbox(
        "Gender",
        options=["Male", "Female", "Other"],
        help="Select your gender"
    )
    
    education = st.selectbox(
        "Education Level",
        options=["High School", "Bachelor's", "Master's", "PhD"],
        help="Highest education level completed"
    )

with col2:
    years_exp = st.number_input(
        "Years of Experience",
        min_value=0,
        max_value=50,
        value=5,
        help="Total years of professional experience"
    )
    
    job_title = st.text_input(
        "Job Title",
        value="Software Engineer",
        help="Your current job title"
    )
    
    current_salary = st.number_input(
        "Current Annual Salary (Optional)",
        min_value=0,
        max_value=1000000,
        value=0,
        step=5000,
        help="Leave as 0 if you don't want to compare"
    )

st.markdown("---")

# Prediction button
if st.button("🔮 Predict Future Salary", type="primary", use_container_width=True):
    
    # Prepare user data
    user_data = {
        'Age': age,
        'Gender': gender,
        'Education Level': education,
        'Job Title': job_title,
        'Years of Experience': years_exp
    }
    
    # Show loading spinner
    with st.spinner("Calculating predictions..."):
        try:
            # Get predictions
            current_pred, future_pred = predict_future_salary(user_data, years_ahead=5)
            
            # Display results
            st.success("✅ Prediction Complete!")
            st.markdown("---")
            
            # Create three columns for metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if current_salary > 0:
                    st.metric(
                        label="Your Current Salary",
                        value=f"${current_salary:,.0f}"
                    )
                else:
                    st.metric(
                        label="Predicted Current Salary",
                        value=f"${current_pred:,.0f}"
                    )
            
            with col2:
                st.metric(
                    label="Predicted Salary (5 Years)",
                    value=f"${future_pred:,.0f}",
                    delta=f"+${future_pred - (current_salary if current_salary > 0 else current_pred):,.0f}"
                )
            
            with col3:
                growth_pct = ((future_pred - (current_salary if current_salary > 0 else current_pred)) / 
                             (current_salary if current_salary > 0 else current_pred)) * 100
                st.metric(
                    label="Expected Growth",
                    value=f"{growth_pct:.1f}%"
                )
            
            st.markdown("---")
            
            # Visualization
            st.subheader("📊 Salary Projection")
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if current_salary > 0:
                categories = ['Current Salary\n(Actual)', 'Current Salary\n(Predicted)', 'Future Salary\n(5 Years)']
                values = [current_salary, current_pred, future_pred]
                colors = ['#3498db', '#2ecc71', '#e74c3c']
            else:
                categories = ['Current Salary\n(Predicted)', 'Future Salary\n(5 Years)']
                values = [current_pred, future_pred]
                colors = ['#3498db', '#e74c3c']
            
            bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'${height:,.0f}',
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            ax.set_ylabel('Annual Salary ($)', fontsize=12, fontweight='bold')
            ax.set_title('Salary Comparison and Projection', fontsize=14, fontweight='bold', pad=20)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_ylim(0, max(values) * 1.15)
            
            # Format y-axis
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Additional insights
            st.markdown("---")
            st.subheader("💡 Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"""
                **Career Trajectory:**
                - Starting Point: ${current_salary if current_salary > 0 else current_pred:,.0f}
                - After 5 Years: ${future_pred:,.0f}
                - Annual Growth: ${(future_pred - (current_salary if current_salary > 0 else current_pred))/5:,.0f}/year
                """)
            
            with col2:
                st.success(f"""
                **Projected Earnings:**
                - Total 5-Year Earnings: ${((current_salary if current_salary > 0 else current_pred) + future_pred) * 2.5:,.0f}
                - Average Annual: ${((current_salary if current_salary > 0 else current_pred) + future_pred) / 2:,.0f}
                """)
            
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.info("Please check your inputs and try again.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built with ❤️ using Streamlit and Scikit-learn</p>
    <p style='font-size: 12px;'>Note: Predictions are based on historical data and should be used as estimates only.</p>
</div>
""", unsafe_allow_html=True)