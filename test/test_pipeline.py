import pandas as pd
import pytest
from src.cleaner import validate_raw_schema, clean_sensor_data, validate_cleaned_data

def test_clean_sensor_data():
    # Create sample sensor data for the cleaning test.
    sample_data = pd.DataFrame({
        "Date": ["2004-03-10", "2004-03-10"],
        "Time": ["18:00:00", "19:00:00"],
        "CO(GT)": [2.6, -200],
        "NMHC(GT)": [-200, -200],
        "PT08.S1(CO)": [1360, 1292],
        "T": [13.6, 13.3]
    })
    # Clean the sample data and check the expected results.
    cleaned = clean_sensor_data(sample_data)
    assert "NMHC(GT)" not in cleaned.columns
    assert "carbon_monoxide" in cleaned.columns
    assert len(cleaned) == 1

def test_validate_cleaned_data_success():
    # Check that valid carbon monoxide data passes validation.
    valid_df = pd.DataFrame({"carbon_monoxide": [1.2, 2.5, 3.1]})
    assert validate_cleaned_data(valid_df) is True

def test_validate_cleaned_data_fails_on_sentinel():
    # Check that the invalid sentinel value fails validation.
    bad_df = pd.DataFrame({"carbon_monoxide": [1.2, -200]})
    with pytest.raises(ValueError):
        validate_cleaned_data(bad_df)
