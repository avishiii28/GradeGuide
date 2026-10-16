
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load the student performance dataset
df = pd.read_csv("student-mat.csv", sep=";")

# 2. Choose the information the model will learn from
features = ["G1", "G2", "studytime", "failures", "absences"]

X = df[features]
y = df["G3"]

# X = input information
# y = final grade we want to predict

# 3. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 4. Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict grades for students the model hasn't seen
predictions = model.predict(X_test)

# 6. Evaluate the model
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("GradeGuide Model Results")
print("-------------------------")
print(f"Students in dataset: {len(df)}")
print(f"Training examples: {len(X_train)}")
print(f"Testing examples: {len(X_test)}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R-squared score: {r2:.2f}")

# 7. Try a prediction for one student
student = pd.DataFrame([{
    "G1": 12,
    "G2": 14,
    "studytime": 3,
    "failures": 0,
    "absences": 2
}])

predicted_grade = model.predict(student)[0]

print(f"\nEstimated final grade: {predicted_grade:.2f}/20")
