🚀 Quick Start Guide
Get the salary prediction system running in 5 minutes!

Option A: Using Your Own Data
Step 1: Prepare Your Dataset
Create a CSV file named salary_data.csv with these columns:

csv
Age,Gender,Education Level,Job Title,Years of Experience,Annual Salary
30,Male,Bachelor's,Software Engineer,5,85000
28,Female,Master's,Data Scientist,3,95000
35,Male,PhD,Senior Manager,10,120000
Step 2: Place the File
bash
# Place your CSV in the data folder
cp /path/to/your/salary_data.csv data/salary_data.csv
Step 3: Install & Run
bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python train_model.py

# Launch the app
streamlit run app.py
Done! The app will open at http://localhost:8501

Option B: Using Sample Data (Recommended for Testing)
Step 1: Generate Sample Data
bash
# Install dependencies
pip install -r requirements.txt

# Generate sample dataset
python generate_sample_data.py
This creates data/salary_data.csv with 1000 realistic salary records.

Step 2: Train & Run
bash
# Train the model (takes 10-30 seconds)
python train_model.py

# Launch the app
streamlit run app.py
The app opens automatically in your browser!

📁 Complete File Checklist
Make sure you have all these files in your project:

salary-prediction/
├── data/
│   └── salary_data.csv ✓
├── models/
│   (created automatically when you train)
├── utils/
│   └── preprocessing.py ✓
├── app.py ✓
├── train_model.py ✓
├── generate_sample_data.py ✓
├── requirements.txt ✓
├── README.md ✓
└── QUICKSTART.md ✓ (this file)
🎯 Testing the App
After running streamlit run app.py, try these example inputs:

Example 1: Junior Developer
Age: 25
Gender: Male
Education: Bachelor's
Job Title: Software Engineer
Years of Experience: 2
Current Salary: $65,000
Expected Result: ~$85,000 - $95,000 in 5 years

Example 2: Senior Data Scientist
Age: 35
Gender: Female
Education: PhD
Job Title: Senior Data Scientist
Years of Experience: 10
Current Salary: $140,000
Expected Result: ~$165,000 - $180,000 in 5 years

Example 3: Mid-Level Manager
Age: 40
Gender: Other
Education: Master's
Job Title: Product Manager
Years of Experience: 15
Current Salary: $125,000
Expected Result: ~$145,000 - $160,000 in 5 years

🔧 Troubleshooting
"No module named 'utils'"
Fix: Make sure you're running commands from the project root directory where the utils/ folder exists.

bash
cd salary-prediction
python train_model.py
"Model files not found"
Fix: Train the model first:

bash
python train_model.py
"File not found: data/salary_data.csv"
Fix: Either add your own data or generate sample data:

bash
python generate_sample_data.py
📊 Understanding the Results
The app shows:

Predicted Current Salary: What the model thinks you should earn now
Predicted Future Salary: Expected salary in 5 years
Expected Growth: Percentage increase
Visualization: Bar chart comparing salaries
Insights: Total earnings and annual growth
🎨 Customizing the Project
Change Prediction Period
Edit app.py line 37 and line 164:

python
# Change from 5 to any number of years
predict_future_salary(user_data, years_ahead=10)
Add More Features
Edit your CSV to include new columns like:

Location (City, State)
Company Size
Industry
Remote Work Status
Then retrain: python train_model.py

Adjust Model Accuracy
Edit train_model.py line 26-32:

python
model = RandomForestRegressor(
    n_estimators=200,  # More trees = better accuracy (but slower)
    max_depth=20,      # Deeper trees = more complex patterns
    # ... other parameters
)
📈 Next Steps
✅ Get the basic app running
✅ Test with sample data
✅ Replace with your real data
✅ Share predictions with friends/colleagues
✅ Customize for your specific use case
💡 Pro Tips
More data = better predictions: Aim for 500+ records
Clean data matters: Remove outliers and fix errors before training
Feature engineering: Consider adding derived features like "Age when started career"
Regular updates: Retrain monthly with new data for best results
Need Help? Check the full README.md for detailed documentation!

