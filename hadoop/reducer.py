#!/usr/bin/env python3
"""
Hadoop Streaming Reducer for Healthcare CXR Metadata Analytics.
Reads sorted key-value pairs from standard input (sys.stdin).
Aggregates:
- Normal vs TB-positive counts
- Total records and prevalence percentage
- Demographic distributions (Gender breakdown)
- Patient statistics (Age min, max, mean per class)
- Storage statistics (Total and average KB per class)
Emits structured medical analytics output.
"""

import sys

def run_reducer():
    current_key = None
    sum_value = 0.0
    count_value = 0
    min_value = float('inf')
    max_value = float('-inf')
    
    # Store aggregated metrics
    results = {}
    
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split("\t")
        if len(parts) != 2:
            continue
            
        key, val_str = parts[0], parts[1]
        try:
            val = float(val_str)
        except ValueError:
            continue
            
        if current_key == key:
            sum_value += val
            count_value += 1
            if val < min_value:
                min_value = val
            if val > max_value:
                max_value = val
        else:
            if current_key is not None:
                results[current_key] = {
                    "sum": sum_value,
                    "count": count_value,
                    "mean": sum_value / count_value if count_value > 0 else 0,
                    "min": min_value if min_value != float('inf') else 0,
                    "max": max_value if max_value != float('-inf') else 0
                }
            current_key = key
            sum_value = val
            count_value = 1
            min_value = val
            max_value = val
            
    if current_key is not None:
        results[current_key] = {
            "sum": sum_value,
            "count": count_value,
            "mean": sum_value / count_value if count_value > 0 else 0,
            "min": min_value if min_value != float('inf') else 0,
            "max": max_value if max_value != float('-inf') else 0
        }
        
    # Pretty-print structured MapReduce summary report
    print("=" * 65)
    print("       HADOOP MAPREDUCE HEALTHCARE ANALYTICS REPORT       ")
    print("=" * 65)
    
    total_records = int(results.get("TOTAL_RECORDS", {}).get("sum", 0))
    normal_count = int(results.get("COUNT_Normal", {}).get("sum", 0))
    tb_count = int(results.get("COUNT_TB_Positive", {}).get("sum", 0))
    
    tb_pct = (tb_count / total_records * 100) if total_records > 0 else 0.0
    norm_pct = (normal_count / total_records * 100) if total_records > 0 else 0.0
    
    print(f"\n[1] DISEASE DISTRIBUTION & CASE COUNTS:")
    print(f"    - Total Validated Cases : {total_records}")
    print(f"    - Normal Diagnoses      : {normal_count} ({norm_pct:.2f}%)")
    print(f"    - TB-Positive Diagnoses : {tb_count} ({tb_pct:.2f}%)")
    
    print(f"\n[2] DEMOGRAPHIC STRATIFICATION (GENDER):")
    for key in sorted(results.keys()):
        if key.startswith("GENDER_"):
            count = int(results[key]["sum"])
            label = key.replace("GENDER_", "").replace("_", " -> ")
            print(f"    - {label:<25} : {count} patients")
            
    print(f"\n[3] PATIENT AGE STATISTICS:")
    for key in ["AGE_Normal", "AGE_TB_Positive"]:
        if key in results:
            d = results[key]
            lbl = "Normal Patients" if "Normal" in key else "TB-Positive Patients"
            print(f"    - {lbl:<22} : Mean={d['mean']:.1f} yrs | Min={d['min']:.0f} | Max={d['max']:.0f} (N={d['count']})")
            
    print(f"\n[4] DATA STORAGE UTILIZATION:")
    for key in ["STORAGE_Normal", "STORAGE_TB_Positive"]:
        if key in results:
            d = results[key]
            lbl = "Normal CXR Images" if "Normal" in key else "TB CXR Images"
            total_mb = d['sum'] / 1024
            mean_kb = d['mean']
            print(f"    - {lbl:<22} : Total={total_mb:.2f} MB | Average Size={mean_kb:.1f} KB")
            
    print("=" * 65)

if __name__ == "__main__":
    run_reducer()
