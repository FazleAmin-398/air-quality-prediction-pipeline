# Air Quality Prediction & Carbon Monoxide Analysis

An end-to-end, modular, and reproducible machine learning pipeline for cleaning, analyzing, and predicting **Carbon Monoxide (CO)** concentrations using the **UCI Air Quality Dataset**.

Reference: [UCI Machine Learning Repository – Air Quality Dataset](https://archive.ics.uci.edu/dataset/360/air+quality)

---

## Project Structure

```text
air-quality-project/
├── data/
│   ├── raw/
│   │   └── AirQualityUCI.xlsx       # Untouched raw dataset
│   ├── processed/
│   │   └── cleaned_air_quality.csv  # Output cleaned dataset
│   └── air_quality.db               # SQLite database generated on run
├── outputs/
│   ├── eda_target_distribution.png
│   ├── eda_correlation_heatmap.png
│   ├── eda_boxplot.png
│   └── actual_vs_predicted.png
├── src/
│   ├── __init__.py
│   ├── cleaner.py                   # Schema validation, cleaning & NumPy transforms
│   ├── analytics.py                 # Pandas summary stats & SQLite join query
│   ├── visualizations.py            # Matplotlib & Seaborn plotting functions
│   └── model.py                     # Model training & evaluation
├── tests/
│   └── test_pipeline.py             # Pytest unit tests
├── main.py                          # Main execution pipeline
├── requirements.txt                 # Project dependencies
└── README.md                        # Documentation
```

---

## Setup & Run Instructions

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Place the raw dataset:**
   Put `AirQualityUCI.xlsx` inside `data/raw/`.

4. **Run the full pipeline:**
   ```bash
   python main.py
   ```

5. **Run unit tests:**
   ```bash
   pytest tests/test_pipeline.py
   ```
