
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt, os
os.makedirs("reports/figures", exist_ok=True)
def main():
    df = pd.read_csv("data/processed_data.csv")
    num = df.select_dtypes(include="number")
    if not num.empty:
        plt.figure(figsize=(6,4)); sns.heatmap(num.corr()); plt.tight_layout()
        plt.savefig("reports/figures/heatmap_correlations.png"); plt.close()
        print("[eda_visuals] heatmap generated")
if __name__ == "__main__":
    main()
