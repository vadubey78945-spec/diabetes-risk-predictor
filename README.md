# Diabetes Risk Prediction System

A machine learning-based web application that predicts diabetes risk based on health parameters..

==============================================

## 🚀 Live Demo
👉 [https://diabetes-risk-predictor-git-vaibhav.streamlit.app/]

==============================================

## 📌 Features

- 🔍 Predicts diabetes risk using ML model
- 📊 Interactive dashboard with gauge visualization
- 📈 Benchmarking with real dataset
- 🧠 Feature importance (model explainability)
- 📄 Downloadable health report
- 🎨 Clean and modern UI (Streamlit)

==============================================

## 🛠️ Tech Stack

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib, Plotly
- Streamlit

===============================================

## 📂 Project Structure

health-risk-predictor/
* app/ 
     app.py # Streamlit UI
* data/
     pima_diabetes_data.csv # Dataset

* src/
     data_preprocessing.py  # Data cleaning
     train_model.py         # Model training
     evaluate.py            # Evaluation

* diabetes_model.pkl        # Trained model
* main.py                   # Pipeline script
* scaler.pkl                # Saved scaler
* requirements.txt
* README.md

==============================================

## 🧠 Model Details

- Model: Random Forest Classifier
- Dataset: PIMA Indians Diabetes Dataset
- Focus: High recall (minimizing false negatives in healthcare)

===============================================

## ⚙️ How to Run Locally

```bash
git clone https://github.com/vadubey78945-spec/diabetes-risk-predictor.git
cd your-repo
pip install -r requirements.txt
streamlit run app/app.py

===============================================

## ⚠️ Disclaimer

This application is for educational purposes only and not a medical diagnosis tool.