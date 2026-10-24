"""
Part 1: Student dropout risk prediction.
Run: python run_analysis.py
Place the five supplied dataset files inside the data/ folder.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

students = pd.read_csv(DATA / "students.csv")
quizzes = pd.read_csv(DATA / "quiz_attempts.csv")
outcomes = pd.read_csv(DATA / "outcome.csv")

# Basic quality checks
print("Rows — students:", len(students), "| quiz attempts:", len(quizzes), "| outcomes:", len(outcomes))
print("Duplicate student IDs:", students["student_id"].duplicated().sum())
print("Missing values in students:\n", students.isna().sum())
quizzes["score"] = pd.to_numeric(quizzes["score"], errors="coerce")
quizzes["time_spent_min"] = pd.to_numeric(quizzes["time_spent_min"], errors="coerce")
quizzes["attempted_at"] = pd.to_datetime(quizzes["attempted_at"], errors="coerce")

# Aggregate many quiz attempts into one row per student.
quiz_features = quizzes.groupby("student_id").agg(
    quiz_attempts=("attempt_id", "count"),
    average_score=("score", "mean"),
    lowest_score=("score", "min"),
    average_time_min=("time_spent_min", "mean"),
    lessons_attempted=("lesson_id", "nunique"),
).reset_index()

df = students.merge(quiz_features, on="student_id", how="left")
df = df.merge(outcomes, on="student_id", how="inner")
df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
df["signup_month"] = df["signup_date"].dt.month
df["dropped_out"] = df["dropped_out"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
df = df.dropna(subset=["dropped_out"])
df["dropped_out"] = df["dropped_out"].astype(int)

# Charts: dropout balance, quiz score by outcome, attempts by outcome.
counts = df["dropped_out"].map({0: "Stayed", 1: "Dropped out"}).value_counts()
counts.plot(kind="bar", title="Student outcome counts", xlabel="Outcome", ylabel="Number of students", rot=0)
plt.tight_layout(); plt.savefig(OUT / "outcome_counts.png", dpi=150); plt.close()

df.boxplot(column="average_score", by="dropped_out")
plt.title("Average quiz score by outcome"); plt.suptitle("")
plt.xlabel("Dropped out (0 = no, 1 = yes)"); plt.ylabel("Average quiz score")
plt.tight_layout(); plt.savefig(OUT / "scores_by_outcome.png", dpi=150); plt.close()

df.boxplot(column="quiz_attempts", by="dropped_out")
plt.title("Quiz attempts by outcome"); plt.suptitle("")
plt.xlabel("Dropped out (0 = no, 1 = yes)"); plt.ylabel("Number of quiz attempts")
plt.tight_layout(); plt.savefig(OUT / "attempts_by_outcome.png", dpi=150); plt.close()

# Exclude identifiers and direct outcome. Keep a simple, explainable feature set.
features = [
    "age_group", "country", "device", "refund_requested", "signup_month",
    "quiz_attempts", "average_score", "lowest_score", "average_time_min", "lessons_attempted"
]
X = df[features]
y = df["dropped_out"]
categorical = ["age_group", "country", "device", "refund_requested"]
numeric = [c for c in features if c not in categorical]

preprocess = ColumnTransformer([
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), numeric),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
models = {
    "Majority baseline": DummyClassifier(strategy="most_frequent"),
    "Logistic regression": Pipeline([("preprocess", preprocess),
                                     ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])
}
rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rows.append({
        "model": name,
        "accuracy": round(accuracy_score(y_test, pred), 3),
        "precision_dropout": round(precision_score(y_test, pred, zero_division=0), 3),
        "recall_dropout": round(recall_score(y_test, pred, zero_division=0), 3),
        "f1_dropout": round(f1_score(y_test, pred, zero_division=0), 3),
    })
    print("\n", name)
    print(classification_report(y_test, pred, target_names=["Stayed", "Dropped out"], zero_division=0))
    if name == "Logistic regression":
        ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Stayed", "Dropped out"])
        plt.title("Logistic regression confusion matrix")
        plt.tight_layout(); plt.savefig(OUT / "confusion_matrix.png", dpi=150); plt.close()
        risk = df.copy()
        risk["dropout_risk_probability"] = model.predict_proba(X)[:, 1]
        risk.sort_values("dropout_risk_probability", ascending=False).head(20).to_csv(
            OUT / "top_20_risk_students.csv", index=False
        )

results = pd.DataFrame(rows)
results.to_csv(OUT / "model_metrics.csv", index=False)
print("\nModel comparison:\n", results.to_string(index=False))
print(f"\nSaved charts and metrics to: {OUT}")
print("Important: this is a small synthetic dataset; predictions should support human outreach, not make decisions automatically.")
