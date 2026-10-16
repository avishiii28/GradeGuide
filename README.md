# GradeGuide — Student Performance Predictor

GradeGuide is a beginner-friendly machine learning web application that estimates a student's final mathematics grade using earlier academic grades and selected student information.

The project demonstrates a complete introductory machine learning workflow, from data preparation and model training to evaluation and an interactive web interface.

## Features

- Predicts an estimated final grade on a 0–20 scale.
- Accepts earlier-period grades, study-time category, past failures, and absences.
- Uses linear regression implemented with scikit-learn.
- Evaluates predictions using Mean Absolute Error (MAE) and R-squared.
- Visualizes actual versus predicted grades on held-out test data.
- Provides an interactive interface built with Streamlit.

## Technologies

- Python
- pandas
- scikit-learn
- Streamlit
- Matplotlib

## Dataset

The project uses the Student Performance dataset from the UCI Machine Learning Repository, specifically the mathematics dataset (`student-mat.csv`).

The selected input features are:
- `G1`: first-period grade
- `G2`: second-period grade
- `studytime`: coded weekly study-time category
- `failures`: number of past class failures
- `absences`: number of school absences

The target variable is `G3`, the final mathematics grade.

Dataset source: https://archive.ics.uci.edu/dataset/320/student+performance

## How It Works

1. Load the student dataset using pandas.
2. Select the input features and final-grade target.
3. Split the data into training and testing sets.
4. Train a linear regression model on the training data.
5. Evaluate the model on held-out test data using MAE and R-squared.
6. Use the trained model to estimate a grade from user-provided inputs.
7. Visualize test-set predictions in the Streamlit application.

## Run Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd GradeGuide
```

### 2. Create and activate a virtual environment

On macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

### 4. Run the application

Ensure `student-mat.csv` is in the project folder, then run:

```bash
streamlit run app.py
```

### 5. Train and evaluate the model separately

```bash
python3 train_model.py
```

## Limitations

- The model is trained on a specific student dataset and may not generalize to other schools or populations.
- Earlier-period grades are important predictors of the final grade.
- Predictions are estimates, not guarantees of academic performance.
- The model is an educational demonstration and should not be used to make consequential decisions about students.

## Future Improvements

- Compare linear regression with another regression algorithm.
- Add cross-validation and further model diagnostics.
- Improve input validation and interface accessibility.
- Deploy the app so it can be accessed through a public link.

## Acknowledgment

Dataset: Cortez, P. and Silva, A. (2008), *Using Data Mining to Predict Secondary School Student Performance*, University of Minho. See the UCI dataset page for details and licensing.

Built as a learning project to explore machine learning, model evaluation, and interactive data applications.