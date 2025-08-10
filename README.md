# Loan Approval Prediction - Streamlit App

This is a Streamlit web application for predicting **loan approval** using a trained RandomForestClassifier.
The model is trained with encoded categorical features and scaled numeric features using MinMaxScaler.

## Features

- User-friendly web UI built with Streamlit
- Encodes categorical features using `.map()`
- Scales numeric features with `MinMaxScaler`
- Loads model and scaler saved with `joblib`
- Displays results clearly with approval/rejection messages

## Project Structure

- `app.py` - Streamlit application
- `train_model.py` - Script to train the model and save it as `loan_approval_rf_98.pkl` and `loan_approval_scaler.pkl`
- `loan_approval_rf_98.pkl` - Saved trained RandomForest model
- `scaler.pkl` - Saved fitted MinMaxScaler

````

2. Run the Streamlit app:
```bash
streamlit run loan_approval_rf_98.py
````

## Requirements

See `requirements.txt` for Python dependencies.
