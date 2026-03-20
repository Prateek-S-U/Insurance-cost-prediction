import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
df = pd.read_csv("insurance.csv")
print("Dataset Loaded Successfully!")
print(df.head())
# BMI Category
def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"
df['bmi_category'] = df['bmi'].apply(bmi_category)
# Risk Score
def risk_score(row):
    score = 0
    if row['smoker'] == 'yes':
        score += 5
    if row['bmi'] > 30:
        score += 3
    if row['age'] > 50:
        score += 2
    return score
df['risk_score'] = df.apply(risk_score, axis=1)
# Interaction Features
df['bmi_smoker'] = df['bmi'] * (df['smoker'] == 'yes')
df['age_bmi'] = df['age'] * df['bmi']
print("\nFeature Engineering Done!")
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['smoker'] = le.fit_transform(df['smoker'])
df['region'] = le.fit_transform(df['region'])
df['bmi_category'] = le.fit_transform(df['bmi_category'])
X = df.drop('charges', axis=1)
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
# Random Forest
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("\nModel Performance:")
print("Linear Regression R2:", r2_score(y_test, lr_pred))
print("Random Forest R2:", r2_score(y_test, rf_pred))
model = rf
def risk_level(cost):
    if cost < 10000:
        return "Low Risk"
    elif cost < 30000:
        return "Medium Risk"
    else:
        return "High Risk"
def recommendations(age, bmi, smoker):
    rec = []
    if bmi > 30:
        rec.append("Reduce BMI by 5 → Save approx ₹5000")
    if smoker == 1:
        rec.append("Quit smoking → Save approx ₹15000")
    if age > 50:
        rec.append("Regular health checkups recommended")
    return rec
sample = pd.DataFrame({
    'age': [45],
    'sex': [1],          
    'bmi': [32],
    'children': [2],
    'smoker': [1],       
    'region': [2],
    'bmi_category': [3],
    'risk_score': [8],
    'bmi_smoker': [32],
    'age_bmi': [1440]
})
predicted_cost = model.predict(sample)[0]
print("\n===== Prediction =====")
print("Predicted Insurance Cost: ₹", round(predicted_cost, 2))
print("Risk Level:", risk_level(predicted_cost))
print("\nRecommendations:")
for r in recommendations(45, 32, 1):
    print("-", r)
print("\n===== Fairness Analysis =====")
print("\nAverage Cost by Gender:")
print(df.groupby('sex')['charges'].mean())

print("\nAverage Cost by Region:")
print(df.groupby('region')['charges'].mean())
