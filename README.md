💰 Salary Prediction System

🧠 An end-to-end machine learning project that predicts a person’s annual salary 5 years into the future using Random Forest regression.

🧾 Overview

The Salary Prediction System is an interactive web application that uses machine learning to estimate future salary growth based on demographic and professional information.

It enables users to:
✅ Predict current salary based on their profile
✅ Estimate future salary (in 5 years)
✅ Visualize salary growth trends
✅ Gain data-driven career insights

🧩 Tech Stack
Category	Tools / Libraries
Language	Python 🐍
Framework	Streamlit 🎈
Machine Learning	Scikit-learn 🤖
Data Handling	pandas, numpy
Visualization	matplotlib, seaborn
Model Persistence	joblib
📁 Project Structure
salary-prediction/
├── data/
│   └── salary_data.csv          # Your dataset
├── models/
│   ├── salary_predictor.pkl     # Trained model (auto-generated)
│   ├── scaler.pkl               # Feature scaler (auto-generated)
│   ├── feature_columns.pkl      # Encoded feature map
│   └── model_evaluation.png     # Model performance plot
├── utils/
│   └── preprocessing.py         # Data cleaning and encoding
├── app.py                       # Streamlit web app
├── train_model.py               # Model training script
├── requirements.txt             # Python dependencies
└── README.md                    # Documentation

📊 Dataset Requirements

Your dataset (salary_data.csv) should contain:

Column Name	Type	Description	Example
Age	Numeric	Person's age	30
Gender	Categorical	Gender identity	Male, Female, Other
Education Level	Categorical	Highest education	Bachelor's, Master's, PhD
Job Title	Categorical	Current job title	Software Engineer
Years of Experience	Numeric	Years of work experience	5
Annual Salary	Numeric	Current annual salary (target)	85000

Example CSV:

Age,Gender,Education Level,Job Title,Years of Experience,Annual Salary
30,Male,Bachelor's,Software Engineer,5,85000
28,Female,Master's,Data Scientist,3,95000
35,Male,PhD,Senior Manager,10,120000

⚙️ Installation & Setup
1️⃣ Clone or Create Project Folder
git clone https://github.com/your-username/salary-prediction.git
cd salary-prediction

2️⃣ Create Folder Structure
mkdir -p data models utils

3️⃣ Add Your Dataset

Copy your CSV file to the data/ folder:

cp /path/to/your/salary_data.csv data/salary_data.csv

4️⃣ Install Dependencies
pip install -r requirements.txt


or individually:

pip install pandas numpy scikit-learn streamlit joblib matplotlib seaborn

🧠 Model Training

Train the model locally using:

python train_model.py


This script will:
✅ Load and preprocess your dataset
✅ Train a Random Forest Regressor
✅ Evaluate model performance (R², MAE, RMSE)
✅ Save trained model and preprocessing pipeline
✅ Generate evaluation plots

✅ Example Output:
===========================================================
SALARY PREDICTION MODEL TRAINING
===========================================================

[1/5] Loading and preprocessing data...
Dataset loaded: 1000 rows, 6 columns

MODEL PERFORMANCE METRICS
===========================================================
Test Set:
  R² Score: 0.8523
  MAE: $8,234.56
  RMSE: $10,456.78

🌐 Run the Web App

Launch the Streamlit interface:

streamlit run app.py


Your browser will open at:
👉 http://localhost:8501

🎯 App Features

Input: Age, Gender, Education, Job Title, Experience, Salary (optional)

Output:

Predicted salary 5 years into the future

Growth percentage

Comparison bar chart

Career insights

🧮 Customize the Model
🔹 Change Prediction Horizon

Open app.py and modify:

current_pred, future_pred = predict_future_salary(user_data, years_ahead=5)


Change 5 to 10 or any other number of years.

🔹 Tune Random Forest Parameters

Edit train_model.py:

model = RandomForestRegressor(
    n_estimators=150,
    max_depth=12,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

🔹 Replace Dataset

Replace the CSV file in data/, retrain the model, and rerun the app:

python train_model.py
streamlit run app.py

📈 Model Evaluation

Performance metrics are stored in models/model_evaluation.png and displayed in the terminal.

Metric	Description
R² Score	How well the model fits the data
MAE	Mean Absolute Error (in dollars)
RMSE	Root Mean Squared Error (error dispersion)
🧹 Troubleshooting
Problem	Cause	Solution
⚠️ “Model files not found”	Model not trained yet	Run python train_model.py
🚫 ImportError	Missing packages	Run pip install -r requirements.txt
🧾 CSV not loading	Wrong path or missing columns	Ensure file is data/salary_data.csv with correct headers
📉 Low R² score	Insufficient or noisy data	Add more samples, tune hyperparameters
🧩 Technical Details

Preprocessing Pipeline

Fill missing numeric values with median

Fill missing categorical values with mode

One-hot encode categorical variables

Normalize features using StandardScaler

Model Details

Algorithm: RandomForestRegressor

Train/Test Split: 80/20

Target: Annual Salary

Features: Age, Gender, Education Level, Job Title, Years of Experience

Future Projection Logic

Adds +5 years to “Years of Experience” and “Age”

Predicts new salary using the trained model

Computes % increase and displays visual chart

📄 License

This project is open-source and available under the MIT License.
Use it freely for personal, educational, or commercial purposes.

🤝 Contributing

Contributions are welcome!
If you’d like to add new models, improve preprocessing, or enhance UI:

Fork this repo

Create a feature branch

Submit a pull request

📧 Contact

📍 Author: Arjun Prabhune
💼 GitHub: @arjun-prabhune

✉️ Email: arjun.prabhune@gmail.com

Built with ❤️ using Python, Scikit-learn, and Streamlit
