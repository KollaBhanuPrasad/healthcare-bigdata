"""
Hadoop MapReduce Execution Runner.
Executes the MapReduce pipeline locally using standard I/O streaming,
mimicking the exact execution pattern of an Apache Hadoop Streaming cluster,
and saves the output to results/mapreduce_output.txt.
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config

def execute_mapreduce_pipeline():
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metadata_csv = config.DATA_DIR / "metadata.csv"
    mapper_py = config.HADOOP_DIR / "mapper.py"
    reducer_py = config.HADOOP_DIR / "reducer.py"
    output_txt = config.RESULTS_DIR / "mapreduce_output.txt"
    
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Input metadata not found at {metadata_csv}. Please run dataset_loader first.")
        
    print("[INFO] Starting Hadoop MapReduce execution...")
    print(f"[INPUT]   {metadata_csv}")
    print(f"[MAPPER]  {mapper_py}")
    print(f"[REDUCER] {reducer_py}")
    
    # 1. Run Mapper: python mapper.py < metadata.csv
    with open(metadata_csv, "r", encoding="utf-8") as infile:
        p_map = subprocess.Popen(
            [sys.executable, str(mapper_py)],
            stdin=infile,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        map_out, map_err = p_map.communicate()
        
    if p_map.returncode != 0:
        print(f"[ERROR] Mapper failed: {map_err}")
        return False
        
    # 2. Hadoop Shuffle & Sort Phase (sort key-value pairs alphabetically by key)
    lines = [l for l in map_out.splitlines() if l.strip()]
    sorted_lines = sorted(lines, key=lambda x: x.split("\t")[0] if "\t" in x else x)
    sorted_input = "\n".join(sorted_lines) + "\n"
    
    # 3. Run Reducer: sorted_stream | python reducer.py
    p_red = subprocess.Popen(
        [sys.executable, str(reducer_py)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    red_out, red_err = p_red.communicate(input=sorted_input)
    
    if p_red.returncode != 0:
        print(f"[ERROR] Reducer failed: {red_err}")
        return False
        
    # Print Reducer output
    print(red_out)
    
    # Save results
    with open(output_txt, "w", encoding="utf-8") as out_f:
        out_f.write(red_out)
        out_f.write("\n\n" + "=" * 65 + "\n")
        out_f.write("EQUIVALENT HADOOP CLUSTER STREAMING CLI COMMAND:\n")
        out_f.write("=" * 65 + "\n")
        out_f.write(
            "hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-*.jar \\\n"
            "  -D mapreduce.job.name='Healthcare_TB_Metadata_Analysis' \\\n"
            "  -input /healthcare/data/metadata.csv \\\n"
            "  -output /healthcare/output/mapreduce_analytics \\\n"
            "  -mapper 'python mapper.py' \\\n"
            "  -reducer 'python reducer.py' \\\n"
            "  -file mapper.py \\\n"
            "  -file reducer.py\n"
        )
        
    print(f"[SUCCESS] Hadoop MapReduce execution output saved to {output_txt}")
    return True

if __name__ == "__main__":
    execute_mapreduce_pipeline()
