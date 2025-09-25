# Diabetes Prediction Project
# Author: Swami Ganesh Deshpande, Roll No: 72

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("diabetes.csv")  # download from Kaggle

# Display first 5 rows
print("First 5 rows of dataset:")
print(df.head())

# Basic info
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing values in dataset:")
print(df.isnull().sum())

# Features and Target
X = df.drop("Outcome", axis=1)  # Features
y = df["Outcome"]               # Target (0 = No Diabetes, 1 = Diabetes)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Support Vector Machine (SVM)
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

def evaluate_model(name, y_test, y_pred):
    print(f"\n--- {name} ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

# Evaluate all models
evaluate_model("Logistic Regression", y_test, y_pred_log)
evaluate_model("Random Forest", y_test, y_pred_rf)
evaluate_model("SVM", y_test, y_pred_svm)

# Example patient data: [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
new_data = np.array([[2, 120, 70, 20, 79, 25.5, 0.5, 30]])
new_data_scaled = scaler.transform(new_data)

new_data = np.array([[3, 150, 72, 35, 88, 32.0, 0.7, 45]])  # 👈 Replace values as needed
new_data_scaled = scaler.transform(new_data)  # scale like training data

prediction = rf_model.predict(new_data_scaled)  # using Random Forest model

if prediction[0] == 1:
    print("\nPrediction for new patient: 🚨 Diabetic")
else:
    print("\nPrediction for new patient: ✅ Not Diabetic")

prediction = rf_model.predict(new_data_scaled)
print("\nPrediction for new data:", "Diabetic" if prediction[0] == 1 else "Not Diabetic")


models = ["Logistic Regression", "Random Forest", "SVM"]
accuracies = [
    accuracy_score(y_test, y_pred_log),
    accuracy_score(y_test, y_pred_rf),
    accuracy_score(y_test, y_pred_svm),
]

plt.bar(models, accuracies, color=['blue','green','orange'])
plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.show()

