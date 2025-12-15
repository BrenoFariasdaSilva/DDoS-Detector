#!/bin/bash

# Permissions: chmod +x download_datasets.sh
# Run: ./download_datasets.sh

set -e

# =========================
# CICDDoS2019
# =========================
mkdir -p Datasets/CICDDoS2019
cd Datasets/CICDDoS2019 || exit 1

URL1="http://cicresearch.ca/CICDataset/CICDDoS2019/Dataset/CSVs/CSV-01-12.zip"
URL2="http://cicresearch.ca/CICDataset/CICDDoS2019/Dataset/CSVs/CSV-03-11.zip"

wget -c "$URL1"
wget -c "$URL2"

unzip -o CSV-01-12.zip
unzip -o CSV-03-11.zip

cd ../../

# =========================
# CIC-IDS-2017
# =========================
mkdir -p Datasets/CICIDS2017
cd Datasets/CICIDS2017 || exit 1

URL3="http://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/GeneratedLabelledFlows.zip"

wget -c "$URL3"

unzip -o GeneratedLabelledFlows.zip
