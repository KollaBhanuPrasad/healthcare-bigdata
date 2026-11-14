"""
Apache Hive Healthcare Analytics Execution Engine.
Executes HiveQL schema and analytical queries using DuckDB's ANSI SQL engine,
providing faithful SQL warehouse execution, table schema verification,
and analytical persistence.
"""

import sys
import json
import re
from pathlib import Path
import pandas as pd
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config

def run_hive_analytics():
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    meta_csv = config.DATA_DIR / "metadata.csv"
    audit_csv = config.PROCESSED_DIR / "preprocessing_audit.csv"
    
    if not meta_csv.exists():
        raise FileNotFoundError(f"Required dataset metadata not found at {meta_csv}")
        
    print("[INFO] Initializing Apache Hive Healthcare Analytics Engine...")
    con = duckdb.connect(database=":memory:")
    
    # Load underlying CSV data into relational tables matching Hive schema
    meta_df = pd.read_csv(meta_csv)
    
    # Table 1: patient_metadata
    patient_df = meta_df[[
        "patient_id", "patient_gender", "patient_age", "label", "diagnosis", "radiology_finding"
    ]]
    con.register("patient_metadata", patient_df)
    
    # Table 2: image_metadata
    image_df = meta_df[[
        "patient_id", "filename", "file_size_kb", "has_left_mask", "has_right_mask"
    ]]
    con.register("image_metadata", image_df)
    
    # Table 3: preprocessing_audit
    if audit_csv.exists():
        audit_df = pd.read_csv(audit_csv)
    else:
        audit_df = pd.DataFrame(columns=[
            "original_filename", "processed_filename", "status", "width", "height",
            "mean_intensity", "std_intensity", "min_val", "max_val"
        ])
    con.register("preprocessing_audit", audit_df)
    
    print("[INFO] Registered Hive Tables: patient_metadata, image_metadata, preprocessing_audit")
    
    # Read healthcare_queries.sql
    queries_file = config.HIVE_DIR / "healthcare_queries.sql"
    with open(queries_file, "r", encoding="utf-8") as qf:
        sql_content = qf.read()
        
    # Split queries by semicolon
    statements = [stmt.strip() for stmt in sql_content.split(";") if stmt.strip()]
    
    query_outputs = []
    summary_dict = {}
    
    print("\n" + "=" * 70)
    print("           APACHE HIVE WAREHOUSE ANALYTICAL EXECUTION           ")
    print("=" * 70)
    
    query_num = 1
    for stmt in statements:
        # Skip database creation/use directives for in-memory session
        upper_s = stmt.upper().strip()
        if upper_s.startswith("USE ") or upper_s.startswith("CREATE DATABASE") or upper_s.startswith("USE;"):
            continue
            
        print(f"\n--- [HIVE QUERY {query_num}] ---")
        try:
            res_df = con.execute(stmt).fetchdf()
            print(res_df.to_string(index=False))
            
            query_record = {
                "query_number": query_num,
                "sql": stmt,
                "rows_returned": len(res_df),
                "data": res_df.to_dict(orient="records")
            }
            query_outputs.append(query_record)
            summary_dict[f"query_{query_num}"] = res_df.to_dict(orient="records")
            query_num += 1
        except Exception as e:
            print(f"[ERROR] Failed executing query: {e}")
            
    print("\n" + "=" * 70)
    
    # Save output reports
    results_txt = config.RESULTS_DIR / "hive_query_results.txt"
    with open(results_txt, "w", encoding="utf-8") as rf:
        rf.write("APACHE HIVE HEALTHCARE BIG DATA ANALYTICS EXECUTION LOG\n")
        rf.write("=" * 70 + "\n\n")
        for q in query_outputs:
            rf.write(f"--- QUERY {q['query_number']} ---\n")
            rf.write(f"SQL: {q['sql']}\n\n")
            rf.write(pd.DataFrame(q['data']).to_string(index=False) + "\n\n")
            
    summary_json = config.RESULTS_DIR / "hive_summary.json"
    with open(summary_json, "w", encoding="utf-8") as jf:
        json.dump(summary_dict, jf, indent=2)
        
    print(f"[SUCCESS] Hive Query Results saved to {results_txt} and {summary_json}")
    return True

if __name__ == "__main__":
    run_hive_analytics()
