
import pandas as pd, mlflow, mlflow.sklearn, os, json, joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns, matplotlib.pyplot as plt

def main():
    os.makedirs("metrics", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    df = pd.read_csv("data/processed_data.csv")
    target = "NObeyesdad" if "NObeyesdad" in df.columns else df.columns[-1]
    X = pd.get_dummies(df.drop(columns=[target]), drop_first=True)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, class_weight="balanced")

    mlflow.set_experiment("ObesityPredictionV9")
    with mlflow.start_run(run_name="rf_baseline"):
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1w = f1_score(y_test, y_pred, average="weighted")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1w)

        rep = classification_report(y_test, y_pred, output_dict=True)
        with open("metrics/classification_report_rf.json","w") as f: json.dump(rep, f, indent=2)

        cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
        plt.figure(figsize=(6,5)); sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
        plt.title("Confusion Matrix - RandomForest"); plt.tight_layout()
        plt.savefig("metrics/confusion_matrix_rf.png"); plt.close()

        joblib.dump(rf, "models/best_model.joblib")
        mlflow.sklearn.log_model(rf, "model")
        mlflow.log_artifact("metrics/classification_report_rf.json")
        mlflow.log_artifact("metrics/confusion_matrix_rf.png")

if __name__ == "__main__":
    main()
