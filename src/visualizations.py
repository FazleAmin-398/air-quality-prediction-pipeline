import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def generate_eda_plots(df: pd.DataFrame, output_dir: str = "outputs"):
    """Generates and saves target distribution, correlation heatmap, and outlier boxplot."""
    # Create the output folder if it does not exist.
    os.makedirs(output_dir, exist_ok=True)

    # Plot the distribution of carbon monoxide values.
    plt.figure(figsize=(8, 5))
    sns.histplot(df["carbon_monoxide"], bins=30, kde=True, color="teal")
    plt.title("Distribution of Carbon Monoxide (CO) Levels")
    plt.xlabel("Carbon Monoxide (mg/m^3)")
    plt.ylabel("Frequency (Count)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_target_distribution.png"), dpi=300)
    plt.close()

    # Plot a heatmap to show correlations between numeric features.
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=["number"])
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Sensor & Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_correlation_heatmap.png"), dpi=300)
    plt.close()

    # Plot a boxplot to identify carbon monoxide outliers.
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df["carbon_monoxide"], color="gold")
    plt.title("Carbon Monoxide Outlier Detection Boxplot")
    plt.xlabel("Carbon Monoxide (mg/m^3)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_boxplot.png"), dpi=300)
    plt.close()

def plot_actual_vs_predicted(y_test, preds, output_dir: str = "outputs"):
    """Generates and saves the Actual vs. Predicted scatter plot for model evaluation."""
    # Create the output folder if it does not exist.
    os.makedirs(output_dir, exist_ok=True)
    # Plot actual values against model predictions.
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, preds, alpha=0.5, color="royalblue")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.title("Actual vs. Predicted Carbon Monoxide (Random Forest)")
    plt.xlabel("Actual CO (mg/m^3)")
    plt.ylabel("Predicted CO (mg/m^3)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "actual_vs_predicted.png"), dpi=300)
    plt.close()
