import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

def train_and_evaluate(df: pd.DataFrame):
    """Trains regression models, compares R2 and RMSE, and returns best model predictions."""
    # Explicitly exclude non-numeric columns like Date, Time, targets, and generated categories
    exclude_cols = ["Date", "Time", "carbon_monoxide", "pollution_severity", "log_temperature", "date", "time"]
    
    # Strictly select only numeric columns for model training
    feature_cols = [
        c for c in df.columns 
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    X = df[feature_cols]
    y = df["carbon_monoxide"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    best_model_name = None
    best_r2 = -float("inf")
    best_preds = None

    print("\n--- MODEL EVALUATION RESULTS ---")
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        r2 = r2_score(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)
        print(f"{name:20s} | R2 Score: {r2:.4f} | RMSE: {rmse:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_preds = preds

    print(f"\n[WINNER] Best Model: {best_model_name} (R2 = {best_r2:.4f})")
    return y_test, best_preds