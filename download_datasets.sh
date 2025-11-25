#!/bin/bash

# Permissions: chmod +x download_datasets.sh
# Run: ./download_datasets.sh

# Create directory structure
mkdir -p Datasets/CICDDoS2019
cd Datasets/CICDDoS2019 || exit

# URLs
URL1="http://cicresearch.ca/CICDataset/CICDDoS2019/Dataset/CSVs/CSV-01-12.zip"
URL2="http://cicresearch.ca/CICDataset/CICDDoS2019/Dataset/CSVs/CSV-03-11.zip"

# Download ZIP files
wget "$URL1"
wget "$URL2"

# Extract both
unzip CSV-01-12.zip
unzip CSV-03-11.zip
