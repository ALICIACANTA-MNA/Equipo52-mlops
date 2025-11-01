
import pandas as pd, joblib, os
import mlflow

def main():
    mlflow.set_experiment("ObesityPredictionV9")
    model = joblib.load("models/best_model.joblib")
    df = pd.read_csv("data/processed_data.csv")
    target = "NObeyesdad" if "NObeyesdad" in df.columns else df.columns[-1]
    X = pd.get_dummies(df.drop(columns=[target]), drop_first=True)
    preds = model.predict(X.head(10))
    os.makedirs("reports", exist_ok=True)
    pd.DataFrame({"prediction": preds}).to_csv("reports/inference_predictions.csv", index=False)
    with mlflow.start_run(run_name="inference"):
        mlflow.log_artifact("reports/inference_predictions.csv")
    print("[inference] predictions saved")

if __name__ == "__main__":
    main()
