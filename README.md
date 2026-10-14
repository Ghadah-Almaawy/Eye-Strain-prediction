# Predicting Eye Strain from Digital Device Usage

A machine learning project that predicts whether a person is likely to experience **digital eye strain** based on their device usage habits and behavioral, environmental, and physiological factors, deployed as a live interactive web app.

**Live App:** https://ghadah-eye-strain.streamlit.app/

> Course project for DS323: Machine Learning, 2nd Semester 2025–2026.
> This was a **group project**; this repository reflects my contribution to the shared codebase.

## Problem Statement

Digital eye strain is a growing issue caused by prolonged use of smartphones, computers, and other digital devices. Early prediction of eye strain risk can help people adjust their habits before the condition worsens.

## Objective

Build and compare machine learning models that classify whether a person is experiencing eye strain (`1`) or not (`0`), based on features such as screen time, screen distance, brightness, room lighting, sleep, and reported symptoms, then deploy the best model as an interactive app.

## Dataset

- **243 questionnaire responses**, 15 features plus the target
- Every answer is encoded as an ordinal or nominal category (for example, phone usage: `0` = less than 2 hours, `1` = 2–4 hours, `2` = 4–6 hours, `3` = more than 6 hours)
- The target is imbalanced: **68%** of respondents report eye strain

## Project Workflow

1. **Data Understanding and Cleaning:** checked missing values, encoding validity, and outliers.
2. **Exploratory Data Analysis:** visualized distributions and relationships between features and the target.
3. **Feature Preprocessing:** compared 4 scaling techniques (`MinMaxScaler`, `StandardScaler`, `RobustScaler`, `Normalizer`).
4. **Model Building:** trained Logistic Regression, Decision Tree, and Random Forest classifiers.
5. **Hyperparameter Tuning:** grid search with 5 fold cross validation for each model.
6. **Evaluation:** accuracy, precision, recall, F1 score, confusion matrices, and a comparison against a majority class baseline.
7. **Deployment:** the final model was saved (`eye_strain_model.pkl`) and deployed as a Streamlit app.

## Results

Test set: 73 respondents (30% stratified split).

| Model | Test Accuracy |
|---|---|
| Majority class baseline (always predicts "eye strain") | 68.5% |
| **Logistic Regression (tuned, final model)** | **76.7%** |
| Decision Tree (tuned) | 76.7% |
| Random Forest (tuned) | 75.3% |

**Logistic Regression** was selected as the final model: it ties for the best test accuracy, shows no overfitting (train 77.1%, test 76.7%), and its coefficients are easy to interpret. It improves on the baseline by about 8 percentage points.

The strongest predictors are reported **headaches or fatigue** and **eye dryness**, followed by age and sleep hours: more sleep is associated with lower risk.

## Web App

Users answer the same questions as the original survey through dropdown menus and get an instant **Lower / Higher eye strain risk** prediction with an estimated likelihood.

The app (`app.py`) loads the tuned Logistic Regression model from `eye_strain_model.pkl` and is deployed on Streamlit Community Cloud.

**Try it:** https://ghadah-eye-strain.streamlit.app/

## Tech Stack

- Python
- pandas, numpy for data handling
- matplotlib, seaborn for visualization
- scipy for statistical testing
- scikit-learn for modeling, scaling, and evaluation
- joblib for model persistence
- Streamlit for the web app

## Running the Project

```bash
git clone https://github.com/Ghadah-Almaawy/Eye-Strain-prediction.git
cd Eye-Strain-prediction
pip install -r requirements.txt
```

**Notebook (training and analysis):**
```bash
jupyter notebook eye_strain_prediction.ipynb
```

**Streamlit app:**
```bash
streamlit run app.py
```

## Repository Structure

```
├── eye_strain_prediction.ipynb   # EDA, modeling, evaluation, and model export
├── eye_strain_model.pkl          # final tuned Logistic Regression model
├── app.py                        # Streamlit web app
└── requirements.txt
```

## Limitations

- The dataset is small (243 responses), so results may vary with a different sample.
- Recall for the "no eye strain" class is limited (48%), because that class has far fewer examples.
- Eye dryness and headaches are self reported symptoms that are closely related to eye strain itself, so the model relies heavily on them. A model based on habits alone would be more useful for prevention and is a natural next step.

## Team & Contribution

This was a group project completed as part of the DS323 Machine Learning course. My role focused on **model building, tuning, and deployment**.

## License

This project is shared for educational and portfolio purposes.
