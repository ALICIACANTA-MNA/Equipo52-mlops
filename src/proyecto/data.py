
import pandas as pd
from pathlib import Path

def main():
    path = Path("data/obesity_estimation_original.csv")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    out = Path("data/processed_data.csv")
    df.to_csv(out, index=False)
    print(f"[prepare] -> {out} ({df.shape})")

if __name__ == "__main__":
    main()
