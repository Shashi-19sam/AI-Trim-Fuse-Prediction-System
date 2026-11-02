
# AI Trim Fuse Prediction System

## Project Overview
The AI Trim Fuse Prediction System uses Machine Learning to predict the optimal Bandgap (BG) Trim Code based on MCU manufacturing parameters.

The system uses a Gradient Boosting Regression model to analyze input parameters and predict trim codes within the valid range of 0 to 63.

## Objectives
- Predict optimal BG Trim Codes using Machine Learning.
- Reduce manual trim code selection.
- Analyze important MCU manufacturing parameters.
- Improve prediction accuracy and consistency.
- Provide an interactive web application for predictions.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

## Machine Learning Algorithm
Gradient Boosting Regressor

## Input Features
1. BG Untrim Voltage
2. PCM VTH N
3. PCM VTH P
4. PCM Resistance
5. PCM TOX
6. PCM Mobility Relative
7. Die X
8. Die Y
9. Radius

## Target Variable
BG Trim Code Optimal

## Model Performance
- Mean Absolute Error (MAE): 0.2597
- Root Mean Squared Error (RMSE): 0.3598
- R2 Score: 0.9951
- Predictions within +/- 1 Trim Code: 99.94%

## Project Structure
AI_Trim_Fuse_Project/
|
|-- dataset/
|-- app.py
|-- data_exploration.py
|-- data_preprocessing.py
|-- data_quality_check.py
|-- feature_analysis.py
|-- model_training.py
|-- model_evaluation.py
|-- predict_trim_code.py
|-- prediction_validation.py
|-- gradient_boosting_model.pkl
|-- model_performance.csv
|-- actual_vs_predicted.png
|-- feature_importance.png
|-- prediction_error.png
|-- trim_code_comparison.png
|-- README.md

## How to Run the Project

### 1. Create Virtual Environment
python -m venv venv

### 2. Activate Virtual Environment
venv\Scripts\activate

### 3. Install Required Libraries
pip install pandas numpy scikit-learn matplotlib streamlit joblib

### 4. Run the Web Application
streamlit run app.py

### 5. Open in Browser
http://localhost:8501

## Expected Output
The application accepts MCU manufacturing parameters and predicts the optimal BG Trim Code between 0 and 63.

## Conclusion
The project demonstrates how Machine Learning can be applied to MCU manufacturing parameter analysis and BG Trim Code prediction. The Streamlit application provides an interactive interface for generating predictions.

Note: Predicted trim codes are model estimates and must be validated using real hardware measurements before fuse programming.
