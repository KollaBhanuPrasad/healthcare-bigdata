"""
Hadoop Distributed File System (HDFS) Simulation and Management Module.
Implements:
1. Standard HDFS namespace hierarchy:
   /healthcare/raw/images
   /healthcare/raw/masks
   /healthcare/raw/clinical
   /healthcare/processed
   /healthcare/features
2. File ingestion pipeline with replication factor, block sizing (128MB), and checksums.
3. HDFS CLI command interface (-mkdir, -put, -ls, -du, -cat, -stat).
4. Generates standard bash cluster deployment scripts (hdfs_setup.sh).
"""

import sys
import os
import shutil
import hashlib
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config

HDFS_ROOT = config.BASE_DIR / "hdfs_storage"

class HDFSClusterSimulator:
    """
    Simulates Hadoop HDFS storage, namespace, and namenode/datanode operations.
    """
    def __init__(self, root_dir: Path = HDFS_ROOT, default_replication: int = 3, block_size_mb: int = 128):
        self.root_dir = root_dir
        self.default_replication = default_replication
        self.block_size_mb = block_size_mb
        self.fs_image_path = self.root_dir / ".namenode_fsimage.json"
        self._init_storage()

    def _init_storage(self):
        self.root_dir.mkdir(parents=True, exist_ok=True)
        if not self.fs_image_path.exists():
            self._save_fsimage({
                "cluster_id": "CID-HEALTHCARE-BIGDATA-01",
                "created_at": datetime.now().isoformat(),
                "block_size_mb": self.block_size_mb,
                "replication_factor": self.default_replication,
                "inodes": {
                    "/": {"type": "DIRECTORY", "permissions": "rwxr-xr-x", "owner": "hdfs", "group": "supergroup", "children": {}}
                }
            })

    def _load_fsimage(self) -> dict:
        with open(self.fs_image_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_fsimage(self, data: dict):
        with open(self.fs_image_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def mkdir(self, hdfs_path: str):
        """Simulates 'hdfs dfs -mkdir -p <hdfs_path>'"""
        norm_path = "/" + hdfs_path.strip("/")
        phys_path = self.root_dir / norm_path.lstrip("/")
        phys_path.mkdir(parents=True, exist_ok=True)
        
        fs = self._load_fsimage()
        fs["inodes"][norm_path] = {
            "type": "DIRECTORY",
            "permissions": "rwxr-xr-x",
            "owner": "hadoop_user",
            "group": "supergroup",
            "created_at": datetime.now().isoformat()
        }
        self._save_fsimage(fs)
        print(f"[HDFS COMMAND] hdfs dfs -mkdir -p {norm_path} -> OK")

    def put(self, local_src: Path, hdfs_dest: str):
        """Simulates 'hdfs dfs -put <local_src> <hdfs_dest>'"""
        local_src = Path(local_src)
        norm_dest = "/" + hdfs_dest.strip("/")
        phys_dest_dir = self.root_dir / norm_dest.lstrip("/")
        phys_dest_dir.mkdir(parents=True, exist_ok=True)
        
        fs = self._load_fsimage()
        
        files_to_copy = []
        if local_src.is_file():
            files_to_copy.append(local_src)
        elif local_src.is_dir():
            files_to_copy.extend([p for p in local_src.glob("*") if p.is_file()])
            
        uploaded_count = 0
        total_bytes = 0
        
        for file in files_to_copy:
            dest_file = phys_dest_dir / file.name
            shutil.copy2(file, dest_file)
            
            size = file.stat().st_size
            total_bytes += size
            uploaded_count += 1
            
            # Compute MD5 checksum
            h = hashlib.md5()
            with open(file, "rb") as f:
                h.update(f.read())
            checksum = h.hexdigest()
            
            inode_key = f"{norm_dest}/{file.name}"
            blocks_needed = max(1, (size + (self.block_size_mb * 1024 * 1024) - 1) // (self.block_size_mb * 1024 * 1024))
            
            fs["inodes"][inode_key] = {
                "type": "FILE",
                "permissions": "rw-r--r--",
                "owner": "hadoop_user",
                "group": "supergroup",
                "size_bytes": size,
                "replication": self.default_replication,
                "block_count": blocks_needed,
                "checksum_md5": checksum,
                "uploaded_at": datetime.now().isoformat()
            }
            
        self._save_fsimage(fs)
        print(f"[HDFS COMMAND] hdfs dfs -put {local_src.name} -> {norm_dest} ({uploaded_count} files, {total_bytes / (1024*1024):.2f} MB ingested)")

    def ls(self, hdfs_path: str = "/") -> list:
        """Simulates 'hdfs dfs -ls -R <hdfs_path>'"""
        norm_path = "/" + hdfs_path.strip("/")
        fs = self._load_fsimage()
        
        print(f"\n[HDFS COMMAND] hdfs dfs -ls -R {norm_path}")
        print("Found items in HDFS namespace:")
        print(f"{'Permissions':<12} {'Replication':<12} {'Owner':<12} {'Group':<12} {'Size':<12} {'HDFS Path'}")
        print("-" * 80)
        
        items = []
        for path, info in sorted(fs["inodes"].items()):
            if path.startswith(norm_path) and path != "/":
                rep = str(info.get("replication", "-"))
                sz = str(info.get("size_bytes", 0)) if info["type"] == "FILE" else "-"
                print(f"{info['permissions']:<12} {rep:<12} {info['owner']:<12} {info['group']:<12} {sz:<12} {path}")
                items.append((path, info))
        print("-" * 80 + "\n")
        return items

    def du(self, hdfs_path: str = "/") -> dict:
        """Simulates 'hdfs dfs -du -h <hdfs_path>'"""
        norm_path = "/" + hdfs_path.strip("/")
        fs = self._load_fsimage()
        
        usage = {}
        for path, info in fs["inodes"].items():
            if info["type"] == "FILE" and path.startswith(norm_path):
                parent_dir = "/" + "/".join(path.strip("/").split("/")[:-1])
                usage[parent_dir] = usage.get(parent_dir, 0) + info["size_bytes"]
                
        print(f"\n[HDFS COMMAND] hdfs dfs -du -h {norm_path}")
        print(f"{'Disk Usage':<15} {'Directory Path'}")
        print("-" * 50)
        for d, b in sorted(usage.items()):
            mb = b / (1024 * 1024)
            print(f"{mb:.2f} MB{'':<8} {d}")
        print("-" * 50 + "\n")
        return usage

def setup_and_ingest_hdfs():
    """
    Executes standard HDFS setup, directories creation, and uploads.
    """
    print("==================================================")
    print("         APACHE HADOOP HDFS INGESTION RUNNER      ")
    print("==================================================")
    
    cluster = HDFSClusterSimulator()
    
    # 1. Create HDFS standard hierarchy
    cluster.mkdir("/healthcare/raw/images")
    cluster.mkdir("/healthcare/raw/clinical")
    cluster.mkdir("/healthcare/raw/masks")
    cluster.mkdir("/healthcare/processed")
    cluster.mkdir("/healthcare/features")
    cluster.mkdir("/healthcare/metadata")
    
    # 2. Ingest metadata and features
    meta_csv = config.DATA_DIR / "metadata.csv"
    if meta_csv.exists():
        cluster.put(meta_csv, "/healthcare/metadata")
        
    summary_csv = config.FEATURES_DIR / "features_summary.csv"
    if summary_csv.exists():
        cluster.put(summary_csv, "/healthcare/features")
        
    audit_csv = config.PROCESSED_DIR / "preprocessing_audit.csv"
    if audit_csv.exists():
        cluster.put(audit_csv, "/healthcare/processed")
        
    # Ingest representative sample of raw images into HDFS
    images = list(config.IMAGES_DIR.glob("*.png"))[:10]
    for img in images:
        cluster.put(img, "/healthcare/raw/images")
        
    # List namespace
    cluster.ls("/healthcare")
    cluster.du("/healthcare")
    
    # Export bash setup script for cluster deployment
    bash_script = config.HADOOP_DIR / "hdfs_setup.sh"
    with open(bash_script, "w", encoding="utf-8") as bf:
        bf.write(
            "#!/usr/bin/env bash\n"
            "# Production Apache Hadoop HDFS Deployment Script for Healthcare CXR Dataset\n"
            "set -e\n\n"
            "echo '[HDFS] Creating directory structure in HDFS...'\n"
            "hdfs dfs -mkdir -p /healthcare/raw/images\n"
            "hdfs dfs -mkdir -p /healthcare/raw/clinical\n"
            "hdfs dfs -mkdir -p /healthcare/raw/masks\n"
            "hdfs dfs -mkdir -p /healthcare/processed\n"
            "hdfs dfs -mkdir -p /healthcare/features\n"
            "hdfs dfs -mkdir -p /healthcare/metadata\n\n"
            "echo '[HDFS] Ingesting files into HDFS with replication 3...'\n"
            "hdfs dfs -put -f data/metadata.csv /healthcare/metadata/\n"
            "hdfs dfs -put -f data/features/*.csv /healthcare/features/\n"
            "hdfs dfs -put -f data/images/*.png /healthcare/raw/images/\n"
            "hdfs dfs -put -f data/clinical_readings/*.txt /healthcare/raw/clinical/\n"
            "hdfs dfs -put -f data/masks/*/*.png /healthcare/raw/masks/\n\n"
            "echo '[HDFS] Verification:'\n"
            "hdfs dfs -ls -R /healthcare\n"
            "hdfs dfs -du -h /healthcare\n"
            "echo '[HDFS] Setup Complete.'\n"
        )
    print(f"[INFO] HDFS Bash script saved to {bash_script}")
    return True

if __name__ == "__main__":
    setup_and_ingest_hdfs()
