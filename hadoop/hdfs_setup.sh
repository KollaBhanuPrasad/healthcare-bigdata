#!/usr/bin/env bash
# Production Apache Hadoop HDFS Deployment Script for Healthcare CXR Dataset
set -e

echo '[HDFS] Creating directory structure in HDFS...'
hdfs dfs -mkdir -p /healthcare/raw/images
hdfs dfs -mkdir -p /healthcare/raw/clinical
hdfs dfs -mkdir -p /healthcare/raw/masks
hdfs dfs -mkdir -p /healthcare/processed
hdfs dfs -mkdir -p /healthcare/features
hdfs dfs -mkdir -p /healthcare/metadata

echo '[HDFS] Ingesting files into HDFS with replication 3...'
hdfs dfs -put -f data/metadata.csv /healthcare/metadata/
hdfs dfs -put -f data/features/*.csv /healthcare/features/
hdfs dfs -put -f data/images/*.png /healthcare/raw/images/
hdfs dfs -put -f data/clinical_readings/*.txt /healthcare/raw/clinical/
hdfs dfs -put -f data/masks/*/*.png /healthcare/raw/masks/

echo '[HDFS] Verification:'
hdfs dfs -ls -R /healthcare
hdfs dfs -du -h /healthcare
echo '[HDFS] Setup Complete.'
