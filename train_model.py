import pandas as pd

data = pd.read_csv("data/student-mat.csv", sep=";")

print("STUDENT PERFORMANCE DATA")

print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nFinal Exam Marks:")
print(data["G3"].head())

# -------------------------------
# DATA ANALYSIS (EDA)
# -------------------------------

# Check dataset columns
print("\nDataset Columns:")
print(data.columns)

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Check duplicate records
print("\nDuplicate Records:")
print(data.duplicated().sum())

# Statistical summary
print("\nStatistical Summary:")
print(data.describe())

# Analyze final grades
print("\nFinal Grade Analysis:")
print(data["G3"].value_counts().sort_index())

# -------------------------------
# DATA VISUALIZATION
# -------------------------------

import matplotlib.pyplot as plt
import seaborn as sns

# Graph 1: Final Grade Distribution

plt.figure(figsize=(8, 5))

sns.histplot(data["G3"], bins=15, kde=True)

plt.title("Student Final Grade Distribution")
plt.xlabel("Final Grade")
plt.ylabel("Number of Students")

plt.savefig("grade_distribution.png")
plt.show()


# Graph 2: Study Time vs Final Grade

plt.figure(figsize=(8, 5))

sns.barplot(x="studytime", y="G3", data=data)

plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time")
plt.ylabel("Average Final Grade")

plt.savefig("studytime_vs_grade.png")
plt.show()

print("Graphs created successfully!")

# -------------------------------
# MACHINE LEARNING PREPARATION
# -------------------------------

# Select input features
X = data[["studytime", "G1", "G2", "absences"]]

# Select target variable
y = data["G3"]

print("\nInput Features:")
print(X.head())

print("\nTarget Variable:")
print(y.head())

print("\nTotal Records:", len(data))

from sklearn.model_selection import train_test_split

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

from sklearn.linear_model import LinearRegression

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

print("\nMachine Learning Model Trained Successfully!")

from sklearn.metrics import mean_absolute_error, r2_score

# Predict test data
predictions = model.predict(X_test)

# Calculate performance
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nMODEL PERFORMANCE")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))
# -------------------------------
# STUDENT PERFORMANCE PREDICTION
# -------------------------------

# Sample student details
new_student = pd.DataFrame({
    "studytime": [3],
    "G1": [15],
    "G2": [16],
    "absences": [2]
})

# Predict final grade
predicted_grade = model.predict(new_student)

print("\nSTUDENT PERFORMANCE PREDICTION")
print("Study Time: 3")
print("First Period Grade: 15")
print("Second Period Grade: 16")
print("Absences: 2")
print("Predicted Final Grade:", round(predicted_grade[0], 2))

# -------------------------------
# SAVE TRAINED MODEL
# -------------------------------

import os
import joblib

# Create models folder
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(model, "models/student_model.pkl")

print("\nModel Saved Successfully!")
