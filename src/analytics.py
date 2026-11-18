import sqlite3
import pandas as pd

def run_pandas_analytics(df: pd.DataFrame) -> dict:
    """Computes summary statistics using Pandas groupby, agg, pivot_table, and value_counts."""
    # Calculate the average, maximum, and count of CO values for each month.
    monthly_stats = df.groupby("month").agg(
        mean_co=("carbon_monoxide", "mean"),
        max_co=("carbon_monoxide", "max"),
        total_readings=("carbon_monoxide", "count")
    )
    # Create a table of average CO values by month and weekday.
    pivot = pd.pivot_table(
        df, values="carbon_monoxide", index="month", columns="dayofweek", aggfunc="mean"
    )
    # Count each pollution severity category.
    severity_counts = df["pollution_severity"].value_counts() if "pollution_severity" in df.columns else None

    # Display the Pandas analysis results.
    print("\n--- PANDAS GROUPBY & AGGREGATION ---")
    print(monthly_stats)
    print("\n--- PANDAS PIVOT TABLE ---")
    print(pivot.head())
    if severity_counts is not None:
        print("\n--- PANDAS VALUE COUNTS ---")
        print(severity_counts)

    return {"monthly_stats": monthly_stats, "pivot": pivot, "severity_counts": severity_counts}

def run_sqlite_analysis(df: pd.DataFrame, db_path: str = "data/air_quality.db") -> pd.DataFrame:
    """Loads cleaned data into SQLite, joins with a rush-hour lookup table, and aggregates CO levels."""
    # Open the database and save the cleaned data as a table.
    conn = sqlite3.connect(db_path)
    df.to_sql("air_quality_clean", conn, if_exists="replace", index=False)

    # Create a table that marks each hour as rush hour or off-peak.
    rush_hour_df = pd.DataFrame({
        "hour": list(range(24)),
        "period_type": [
            "Rush Hour" if h in [7, 8, 9, 17, 18, 19, 20] else "Off-Peak"
            for h in range(24)
        ]
    })
    rush_hour_df.to_sql("time_periods", conn, if_exists="replace", index=False)

    # Join the data tables and calculate CO statistics for each period.
    query = """
    SELECT 
        t.period_type,
        COUNT(a.carbon_monoxide) AS total_readings,
        ROUND(AVG(a.carbon_monoxide), 2) AS avg_co_level,
        ROUND(MAX(a.carbon_monoxide), 2) AS max_co_level
    FROM air_quality_clean a
    JOIN time_periods t ON a.hour = t.hour
    GROUP BY t.period_type
    ORDER BY avg_co_level DESC;
    """
    result = pd.read_sql_query(query, conn)
    # Close the database connection after the query is complete.
    conn.close()

    # Display the SQLite query result.
    print("\n--- SQLITE JOIN & GROUP BY RESULT ---")
    print(result.to_string(index=False))
    return result
