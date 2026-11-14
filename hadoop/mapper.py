#!/usr/bin/env python3
"""
Hadoop Streaming Mapper for Healthcare CXR Metadata Analytics.
Reads tabular patient & image metadata from standard input (sys.stdin).
Emits key-value pairs to standard output (sys.stdout) for Hadoop shuffle and sort.

Keys emitted:
- COUNT\t<diagnosis> : counts for Normal and TB_Positive
- GENDER\t<diagnosis>_<gender> : demographic correlation
- AGE\t<diagnosis> : patient age values for statistics calculation
- SIZE_KB\t<diagnosis> : image storage size for resource distribution
- TOTAL\tALL : overall record counter
"""

import sys

def run_mapper():
    is_header = True
    header_indices = {}
    
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
            
        parts = [p.strip() for p in line.split(",")]
        
        # Detect and parse header
        if is_header:
            if "patient_id" in parts or "label" in parts:
                header_indices = {col: idx for idx, col in enumerate(parts)}
                is_header = False
                continue
            is_header = False
            
        try:
            # Flexible field extraction
            if header_indices:
                pid = parts[header_indices.get("patient_id", 0)]
                label = parts[header_indices.get("label", 2)]
                diagnosis = parts[header_indices.get("diagnosis", 3)]
                gender = parts[header_indices.get("patient_gender", 4)]
                age = parts[header_indices.get("patient_age", 5)]
                size_kb = parts[header_indices.get("file_size_kb", 6)]
            else:
                # Default positional fallback
                pid = parts[0]
                label = parts[2]
                diagnosis = parts[3]
                gender = parts[4]
                age = parts[5]
                size_kb = parts[6]
                
            # Normalize diagnosis
            if diagnosis not in ["Normal", "TB_Positive"]:
                diagnosis = "Normal" if str(label) == "0" else "TB_Positive"
                
            # Emit Key-Value pairs (Tab-delimited)
            print(f"COUNT_{diagnosis}\t1")
            print(f"TOTAL_RECORDS\t1")
            
            if gender in ["Male", "Female"]:
                print(f"GENDER_{diagnosis}_{gender}\t1")
                
            try:
                age_val = float(age)
                if 0 < age_val < 120:
                    print(f"AGE_{diagnosis}\t{age_val}")
            except (ValueError, TypeError):
                pass
                
            try:
                size_val = float(size_kb)
                if size_val > 0:
                    print(f"STORAGE_{diagnosis}\t{size_val}")
            except (ValueError, TypeError):
                pass
                
        except Exception as e:
            # In MapReduce streaming, malformed lines are logged or skipped
            sys.stderr.write(f"Mapper Warning: skipping line due to {e}\n")
            continue

if __name__ == "__main__":
    run_mapper()
