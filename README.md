# Predicting Eye Strain from Digital Device Usage 

A machine learning project that predicts whether a person is likely to experience **digital eye strain** based on their device-usage habits and behavioral, environmental, and physiological factors — deployed as a live interactive web app.

**Live Demo:** https://GhadahAlmaawy.github.io/eye-strain-prediction/

> Course project — DS323: Machine Learning, 2nd Semester 2025–2026.
> This was a **group project**; this repository reflects my contribution to the shared codebase.

##  Problem Statement

Digital eye strain is a growing issue caused by prolonged use of smartphones, computers, and other digital devices. Early prediction of eye strain risk can help people adjust their habits before the condition worsens.

##  Objective

Build and compare machine learning models that classify whether a person is experiencing eye strain (`1`) or not (`0`), based on features such as screen time, device type, environmental lighting, blink rate, and other usage patterns — then deploy the best model as an interactive app.

##  Project Workflow

1. **Data Understanding & Cleaning** — inspecting the dataset, checking for missing values and outliers (Shapiro-Wilk normality test).
2. **Exploratory Data Analysis (EDA)** — visualizing distributions and relationships between features and the target variable.
3. **Feature Preprocessing** — comparing 4 scaling techniques: `MinMaxScaler`, `StandardScaler`, `RobustScaler`, `Normalizer`.
4. **Model Building** — training and tuning three classifiers:
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
5. **Hyperparameter Tuning** — manual grid search with cross-validation for each model.
6. **Model Evaluation** — accuracy, precision, recall, f1-score, and confusion matrices.
7. **Deployment** — the best model was saved (`eye_strain_model.pkl`) and wrapped in a **Streamlit** app for interactive, real-time predictions.

##  Results

| Model | Test Accuracy (tuned) |
|---|---|
| Logistic Regression | **76.7%** |
| Decision Tree | 76.7% |
| Random Forest | 75.3% |

Logistic Regression and the tuned Decision Tree performed best on this dataset, with Random Forest close behind.

##  Web App

The deployed app lets a user enter their own device-usage habits (screen time, sleep hours, break frequency, brightness, etc.) through sliders and dropdowns, and get an instant **Low / High Eye Strain** prediction.

**Try it:** https://GhadahAlmaawy.github.io/eye-strain-prediction/

> This static version runs the trained Logistic Regression model directly in the browser (no server needed). A Streamlit version of the same app (`app.py`) is also included in this repo and can be run locally or deployed separately.

##  Tech Stack

- Python
- pandas, numpy — data handling
- matplotlib, seaborn — visualization
- scipy — statistical testing
- scikit-learn — modeling, scaling, evaluation
- joblib — model persistence
- Streamlit — web app deployment

##  Running the Project

**Notebook (model training & analysis):**
```bash
git clone <this-repo-url>
cd eye-strain-prediction
pip install -r requirements.txt
jupyter notebook eye_strain_prediction.ipynb
```

**Web app (locally):**
```bash
streamlit run app.py
```

##  Repository Structure

```
├── eye_strain_prediction.ipynb   # full EDA, modeling & evaluation notebook
├── app.py                        # Streamlit web app
├── eye_strain_model.pkl          # trained model used by the app
├── eye_strain_data.csv           # dataset (used by the app for input options)
└── requirements.txt
```

##  Team & Contribution

This was a group project completed as part of the DS323 Machine Learning course. My role focused on **"model building, tuning & Streamlit deployment"**.

##  License

This project is shared for educational and portfolio purposes.
