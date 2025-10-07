💰 Salary Prediction System
A complete end-to-end machine learning project that predicts a person's annual salary 5 years into the future using Random Forest regression.

📋 Project Overview
This project provides an interactive web application that:

Trains a machine learning model on historical salary data
Predicts current salary based on user inputs
Projects salary 5 years into the future
Provides visual comparisons and insights
🗂️ Project Structure
salary-prediction/
├── data/
│   └── salary_data.csv          # Dataset (you provide this)
├── models/
│   ├── salary_predictor.pkl     # Trained model (generated)
│   ├── scaler.pkl               # Feature scaler (generated)
│   ├── feature_columns.pkl      # Feature names (generated)
│   └── model_evaluation.png     # Performance visualization (generated)
├── utils/
│   └── preprocessing.py         # Data preprocessing functions
├── app.py                       # Streamlit web application
├── train_model.py               # Model training script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
📊 Dataset Requirements
Your salary_data.csv file should contain the following columns:

Column Name	Type	Description	Example
Age	Numeric	Person's age	30
Gender	Categorical	Gender identity	Male, Female, Other
Education Level	Categorical	Highest education	Bachelor's, Master's, PhD
Job Title	Categorical	Current job title	Software Engineer
Years of Experience	Numeric	Years of work experience	5
Annual Salary	Numeric	Current annual salary (target)	85000
Example CSV Structure:
csv
Age,Gender,Education Level,Job Title,Years of Experience,Annual Salary
30,Male,Bachelor's,Software Engineer,5,85000
28,Female,Master's,Data Scientist,3,95000
35,Male,PhD,Senior Manager,10,120000
🚀 Setup Instructions
1. Clone or Create Project Directory
bash
mkdir salary-prediction
cd salary-prediction
2. Create Folder Structure
bash
mkdir data models utils
3. Add Your Dataset
Place your salary_data.csv file in the data/ folder:

bash
# Copy your CSV file
cp /path/to/your/salary_data.csv data/salary_data.csv
4. Install Dependencies
bash
pip install -r requirements.txt
Or install individually:

bash
pip install pandas numpy scikit-learn streamlit joblib matplotlib seaborn
5. Train the Model
bash
python train_model.py
This will:

Load and preprocess your data
Train a Random Forest regression model
Evaluate performance metrics (R², MAE, RMSE)
Save the trained model to models/salary_predictor.pkl
Save the scaler to models/scaler.pkl
Generate performance visualizations
Expected Output:

===========================================================
SALARY PREDICTION MODEL TRAINING
===========================================================

[1/5] Loading and preprocessing data...
Dataset loaded: 1000 rows, 6 columns
...

MODEL PERFORMANCE METRICS
===========================================================
Test Set:
  R² Score: 0.8523
  MAE: $8,234.56
  RMSE: $10,456.78
...
6. Run the Streamlit App
bash
streamlit run app.py
The app will open in your browser at http://localhost:8501

🎯 Using the Application
Enter Your Information:
Age
Gender
Education Level
Job Title
Years of Experience
Current Annual Salary (optional)
Click "Predict Future Salary"
View Results:
Predicted current salary (if no current salary provided)
Predicted salary in 5 years
Salary growth percentage
Visual comparison chart
Career insights and projections
🔧 Customization
Modify Prediction Horizon
To change the prediction period from 5 years to another value, edit app.py:

python
# Change this line (appears twice in the file)
current_pred, future_pred = predict_future_salary(user_data, years_ahead=5)

# To (for example, 10 years):
current_pred, future_pred = predict_future_salary(user_data, years_ahead=10)
Change Model Parameters
Edit train_model.py to adjust the Random Forest parameters:

python
model = RandomForestRegressor(
    n_estimators=100,        # Number of trees
    max_depth=15,            # Maximum tree depth
    min_samples_split=5,     # Minimum samples to split
    min_samples_leaf=2,      # Minimum samples in leaf
    random_state=42,
    n_jobs=-1
)
Replace Dataset
To use a new dataset:

Ensure it has the required columns (Age, Gender, Education Level, Job Title, Years of Experience, Annual Salary)
Place it in data/salary_data.csv
Retrain the model: python train_model.py
Restart the app: streamlit run app.py
📈 Model Performance
The model's performance is evaluated using:

R² Score: Coefficient of determination (how well the model fits the data)
MAE (Mean Absolute Error): Average prediction error in dollars
RMSE (Root Mean Squared Error): Standard deviation of prediction errors
These metrics are displayed after training and saved in models/model_evaluation.png.

🛠️ Troubleshooting
"Model files not found" Error
Problem: The Streamlit app can't find the trained model.

Solution: Run python train_model.py first to generate the model files.

Import Errors
Problem: Missing Python packages.

Solution: Install all requirements:

bash
pip install -r requirements.txt
Dataset Loading Issues
Problem: CSV file can't be loaded or has wrong format.

Solution:

Check that salary_data.csv is in the data/ folder
Verify column names match exactly (case-sensitive)
Ensure there are no special characters in column names
Poor Model Performance
Problem: Low R² score or high MAE.

Solution:

Ensure you have enough data (minimum 500+ rows recommended)
Check for data quality issues (outliers, missing values)
Try tuning model parameters in train_model.py
📝 Technical Details
Preprocessing Pipeline
Data Loading: Reads CSV and displays basic statistics
Missing Value Handling: Fills numerical with median, categorical with mode
Duplicate Removal: Removes duplicate rows
Categorical Encoding: One-hot encoding for Gender, Education Level, Job Title
Feature Scaling: StandardScaler normalization for all features
Model Architecture
Algorithm: Random Forest Regressor
Features: Age, Gender, Education Level, Job Title, Years of Experience
Target: Annual Salary
Train/Test Split: 80/20
Future Salary Prediction
The model predicts future salary by:

Taking current user inputs
Adding 5 years to "Years of Experience" and "Age"
Making a new prediction with updated features
Comparing current vs. future predictions
📄 License
This project is open source and available for educational and commercial use.

🤝 Contributing
Feel free to:

Report bugs
Suggest features
Submit pull requests
Improve documentation
📧 Support
For questions or issues, please refer to the troubleshooting section or check the Streamlit and scikit-learn documentation.

Built with ❤️ using Python, Scikit-learn, and Streamlit

