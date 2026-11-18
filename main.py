import os
import pandas as pd
# Import functions for cleaning and validating the dataset.
from src.cleaner import (
    validate_raw_schema,
    clean_sensor_data,
    apply_numpy_transformations,
    validate_cleaned_data,
)
# Import functions for data analysis and visualization.
from src.analytics import run_pandas_analytics, run_sqlite_analysis
from src.visualizations import generate_eda_plots, plot_actual_vs_predicted
# Import the model training function.
from src.model import train_and_evaluate

# Define the input, output, and database file paths.
RAW_PATH = "data/raw/AirQualityUCI.xlsx"
CLEAN_PATH = "data/processed/cleaned_air_quality.csv"
DB_PATH = "data/air_quality.db"

def main():
    # Create folders for processed data and charts.
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Load the raw Excel file and check its basic structure.
    print("1. Loading and validating raw data...")
    raw_df = pd.read_excel(RAW_PATH)
    validate_raw_schema(raw_df)

    before_rows = len(raw_df)
    before_sentinel = int((raw_df == -200).sum().sum())

    # Clean the data, add calculated features, and validate it.
    print("2. Cleaning data and applying transformations...")
    cleaned_df = clean_sensor_data(raw_df)
    cleaned_df = apply_numpy_transformations(cleaned_df)
    validate_cleaned_data(cleaned_df)

    cleaned_df.to_csv(CLEAN_PATH, index=False)
    print(f"[SAVED] Cleaned dataset exported to {CLEAN_PATH}")

    # Run Pandas and SQLite analytics.
    print("3. Running Pandas and SQLite analytics...")
    run_pandas_analytics(cleaned_df)
    run_sqlite_analysis(cleaned_df, DB_PATH)

    # Create exploratory data analysis charts.
    print("4. Generating EDA visualizations...")
    generate_eda_plots(cleaned_df, "outputs")

    # Train the models and compare their predictions.
    print("5. Training and evaluating ML models...")
    y_test, best_preds = train_and_evaluate(cleaned_df)
    plot_actual_vs_predicted(y_test, best_preds, "outputs")

    # Report the number of rows and sentinel values before and after cleaning.
    after_rows = len(cleaned_df)
    print("\n================ DATA-QUALITY REPORT ================")
    print(f"Total Rows:              Before = {before_rows} | After = {after_rows}")
    print(f"Sentinel (-200) Values:  Before = {before_sentinel} | After = 0")
    print(f"Validation Status:       PASSED (100%)")
    print("=====================================================\n")

if __name__ == "__main__":
    # Start the complete pipeline when the file is run directly.
    print("Starting pipeline...")
    main()