
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# -----------------------------
# 1. Page setup
# -----------------------------
st.set_page_config(
    page_title="GradeGuide | Student Insights",
    page_icon="📚",
    layout="wide"
)


# -----------------------------
# 2. Load and train the model
# -----------------------------
@st.cache_resource
def train_model():
    df = pd.read_csv("student-mat.csv", sep=";")

    features = [
        "G1",
        "G2",
        "studytime",
        "failures",
        "absences"
    ]

    X = df[features]
    y = df["G3"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    results = pd.DataFrame({
        "Actual grade": y_test,
        "Predicted grade": predictions
    })

    return model, mae, r2, results


# -----------------------------
# 3. Page heading
# -----------------------------
st.title("GradeGuide")
st.caption("A machine learning project for exploring student performance")

st.markdown(
    """
    Predict an estimated final grade using earlier grades,
    study-time category, past failures, and absences.

    **Built with:** Python · pandas · scikit-learn · Streamlit
    """
)

st.write(
    "Explore how previous grades, study habits, "
    "and attendance relate to estimated final performance."
)

st.info(
    "This is an educational ML demo. Predictions are estimates, "
    "not guarantees of a student's future performance."
)


# -----------------------------
# 4. Get model and results
# -----------------------------
try:
    model, mae, r2, results = train_model()
except FileNotFoundError:
    st.error(
        "Dataset not found. Please put student-mat.csv "
        "in the same folder as app.py."
    )
    st.stop()


# -----------------------------
# 5. Student input section
# -----------------------------
st.header("Enter student information")

left, right = st.columns(2)

with left:
    g1 = st.slider(
        "First-period grade (G1)",
        min_value=0,
        max_value=20,
        value=12
    )

    g2 = st.slider(
        "Second-period grade (G2)",
        min_value=0,
        max_value=20,
        value=14
    )

    study_labels = {
        "Less than 2 hours": 1,
        "2–5 hours": 2,
        "5–10 hours": 3,
        "More than 10 hours": 4
    }

    study_choice = st.selectbox(
        "Weekly study time",
        list(study_labels.keys()),
        index=1
    )

    studytime = study_labels[study_choice]

with right:
    failures = st.selectbox(
        "Number of past class failures",
        [0, 1, 2, 3],
        index=0
    )

    absences = st.slider(
        "Number of school absences",
        min_value=0,
        max_value=30,
        value=2
    )

    st.caption(
        "Study-time categories follow the coding in the "
        "original dataset."
    )


# -----------------------------
# 6. Make a prediction
# -----------------------------
if st.button("Predict final grade", type="primary"):

    student = pd.DataFrame([{
        "G1": g1,
        "G2": g2,
        "studytime": studytime,
        "failures": failures,
        "absences": absences
    }])

    prediction = float(model.predict(student)[0])

    # Keep the displayed score within the dataset's 0–20 scale.
    displayed_prediction = max(0, min(20, prediction))

    st.divider()
    st.header("Your GradeGuide result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Estimated final grade",
            f"{displayed_prediction:.1f} / 20"
        )

        st.progress(displayed_prediction / 20)

    with col2:
        st.metric(
            "Model's average test error (MAE)",
            f"{mae:.2f} grade points"
        )

    st.caption(
        "The estimate is limited to the 0–20 scale for display. "
        "Actual outcomes may differ."
    )

    st.subheader("Your input summary")

    summary = pd.DataFrame({
        "Input": [
            "First-period grade",
            "Second-period grade",
            "Study-time category",
            "Past failures",
            "Absences"
        ],
        "Value": [
            g1,
            g2,
            study_choice,
            failures,
            absences
        ]
    })

    st.dataframe(summary, hide_index=True, use_container_width=True)


# -----------------------------
# 7. Model evaluation
# -----------------------------
st.divider()
st.header("How well does the model perform?")

col1, col2 = st.columns(2)

st.write(
    "The metrics below evaluate the model on held-out test data. "
    "They do not measure the accuracy of every individual prediction."
)

with col1:
    st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
    st.write(
        "On average, predictions differ from actual grades "
        "by about this many grade points on the test set."
    )

with col2:
    st.metric("R-squared score", f"{r2:.2f}")
    st.write(
        "This indicates how much variation in test-set grades "
        "is explained by the model, relative to a simple baseline."
    )

st.subheader("Actual vs predicted grades")

fig, ax = plt.subplots(figsize=(7, 4))

ax.scatter(
    results["Actual grade"],
    results["Predicted grade"],
    alpha=0.7
)

ax.plot([0, 20], [0, 20], linestyle="--")

ax.set_xlabel("Actual final grade")
ax.set_ylabel("Predicted final grade")
ax.set_xlim(0, 20)
ax.set_ylim(0, 20)
ax.set_title("Model predictions on unseen test data")

st.pyplot(fig)
plt.close(fig)

st.caption(
    "Points closer to the dashed line indicate more accurate "
    "predictions. Evaluation results are based on the held-out "
    "test set, not on the student entered above."
)

st.divider()
st.caption("GradeGuide | Built with Python, pandas, scikit-learn and Streamlit")
