
import pandas as pd
def main():
    try:
        ref = pd.read_csv("data/dataset_limpio.csv")
        cur = pd.read_csv("data/processed_data.csv")
        same = set(ref.columns) == set(cur.columns)
        print(f"[verify_eda] columnas equivalentes: {same}")
    except Exception as e:
        print(f"[verify_eda] error: {e}")
if __name__ == "__main__":
    main()
