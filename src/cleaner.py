import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# Keep the expected source columns in one place for consistent schema handling.
EXPECTED_COLUMNS = [
    "Date", "Time", "CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)",
    "PT08.S2(NMHC)", "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)",
    "PT08.S5(O3)", "T", "RH", "AH"
]

def validate_raw_schema(df: pd.DataFrame) -> bool:
    """Validates that the raw dataset is non-empty."""
    # Reject an empty dataset before any cleaning or transformation runs.
    if df.empty:
        raise ValueError("VALIDATION FAILED: Raw dataset is completely empty.")
    return True

def clean_sensor_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans sensor data, strips spaces, handles sentinel values, extracts temporal features,
    and safely imputes only valid numeric features.
    """
    df = df.copy()
    
    # Normalize column names so source formatting differences do not affect matching.
    df.columns = [str(c).strip() for c in df.columns]

    # Convert non-Date/Time values to numeric types while coercing invalid values to NaN.
    for col in df.columns:
        if col.lower() not in ["date", "time"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sensor datasets use -200 to represent missing or invalid measurements.
    df.replace(-200, np.nan, inplace=True)
    
    # NMHC is excluded because it is not used by the cleaning or modeling workflow.
    if "NMHC(GT)" in df.columns:
        df.drop(columns=["NMHC(GT)"], inplace=True)
        
    # Standardize the target name for downstream functions.
    if "CO(GT)" in df.columns:
        df.rename(columns={"CO(GT)": "carbon_monoxide"}, inplace=True)
    elif "CO(gt)" in df.columns:
        df.rename(columns={"CO(gt)": "carbon_monoxide"}, inplace=True)

    # Remove rows where the model target is missing or was already converted to NaN.
    if "carbon_monoxide" in df.columns:
        df.dropna(subset=["carbon_monoxide"], inplace=True)
    else:
        raise KeyError("Could not find 'CO(GT)' target column in dataset.")
    
    # Derive calendar features and use safe fallback values for invalid dates.
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["month"] = df["Date"].dt.month.fillna(1).astype(int)
        df["dayofweek"] = df["Date"].dt.dayofweek.fillna(0).astype(int)
    else:
        df["month"] = 1
        df["dayofweek"] = 0
    
    # Extract the hour from datetime objects or HH:MM formatted strings.
    if "Time" in df.columns:
        df["hour"] = df["Time"].apply(lambda t: t.hour if hasattr(t, "hour") else int(str(t).split(":")[0]) if pd.notnull(t) and ":" in str(t) else 0)
    else:
        df["hour"] = 0

    # Exclude identifiers, timestamps, the target, and derived columns from feature imputation.
    exclude_cols = ["Date", "Time", "carbon_monoxide", "pollution_severity", "log_temperature", "time"]
    numeric_feature_cols = []
    for c in df.columns:
        if c.lower() not in [ex.lower() for ex in exclude_cols]:
            if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().sum() > 0:
                numeric_feature_cols.append(c)
    
    # Fill missing values in usable numeric features with each feature's median.
    if numeric_feature_cols:
        imputer = SimpleImputer(strategy="median")
        imputed_array = imputer.fit_transform(df[numeric_feature_cols])
        df[numeric_feature_cols] = pd.DataFrame(imputed_array, columns=numeric_feature_cols, index=df.index)
    
    return df

def apply_numpy_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """Applies vectorized NumPy operations for severity classification and log scaling safely."""
    df = df.copy()
    if "carbon_monoxide" in df.columns:
        # Classify records using the project threshold for pollution severity.
        severity_vals = np.where(df["carbon_monoxide"] > 5.0, "High", "Normal")
        df["pollution_severity"] = pd.Series(severity_vals, index=df.index)
        
    if "T" in df.columns:
        # Clamp negative temperatures before applying a log-one-plus transformation.
        temp_vals = np.log1p(np.maximum(df["T"], 0))
        df["log_temperature"] = pd.Series(temp_vals, index=df.index)
        
    return df

def validate_cleaned_data(df: pd.DataFrame) -> bool:
    """Validates range, type, and null constraints on the cleaned dataframe."""
    # Cleaned data must not contain the original sentinel value.
    if (df == -200).any().any():
        raise ValueError("VALIDATION FAILED: Sentinel value -200 still exists after cleaning!")
    # The target variable must be complete before model training.
    if df["carbon_monoxide"].isna().any():
        raise ValueError("VALIDATION FAILED: Target variable 'carbon_monoxide' contains NaN values!")
    # Carbon monoxide measurements cannot be negative.
    if (df["carbon_monoxide"] < 0).any():
        raise ValueError("VALIDATION FAILED: Negative Carbon Monoxide values detected!")
    return True