
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path

def main():
    df = pd.read_csv("data/processed_data.csv")
    target = "NObeyesdad" if "NObeyesdad" in df.columns else df.columns[-1]
    X = df.drop(columns=[target]) if target in df.columns else df
    y = df[target] if target in df.columns else None

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    pre = ColumnTransformer([("num", StandardScaler(), num_cols),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])

    pipe = Pipeline([("pre", pre)])
    Xt = pipe.fit_transform(X)
    import joblib, numpy as np
    Path("models").mkdir(exist_ok=True)
    joblib.dump(pipe, "models/preprocess.joblib")
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    pd.DataFrame(Xt).to_csv("data/features_data.csv", index=False)
    print("[features] -> data/features_data.csv")

if __name__ == "__main__":
    main()
