#!/bin/bash

# Permissions: chmod +x download_datasets.sh
# Run: ./download_datasets.sh

set -e # Exit on error

# CONFIGURATION: Dataset URLs and Target Directories (Comment/Uncomment as needed)
declare -A DATASET_URLS=(
	# CICDDoS2019
	["CICDDoS2019_CSV_01_12"]="http://cicresearch.ca/CICDataset/CICDDoS2019/Dataset/CSVs/CSV-01-12.zip"
	["CICDDoS2019_CSV_03_11"]="http://cicresearch.ca/CICDataset/CICDDoS2019/Dataset/CSVs/CSV-03-11.zip"

	# CIC-IDS-2017
	["CICIDS2017"]="http://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/GeneratedLabelledFlows.zip"
)

declare -A DATASET_DIRS=(
	["CICDDoS2019_CSV_01_12"]="Datasets/CICDDoS2019"
	["CICDDoS2019_CSV_03_11"]="Datasets/CICDDoS2019"
	["CICIDS2017"]="Datasets/CICIDS2017"
)

# DOWNLOAD AND EXTRACT DATASETS
for DATASET in "${!DATASET_URLS[@]}"; do
	URL="${DATASET_URLS[$DATASET]}"
	TARGET_DIR="${DATASET_DIRS[$DATASET]}"
	ZIP_NAME="$(basename "$URL")"

	echo "==> Setting up dataset: $DATASET"
	echo "    From: $URL"
	echo "    To:   $TARGET_DIR/$ZIP_NAME"
	mkdir -p "$TARGET_DIR"
	echo "==> Changing to directory $TARGET_DIR"
	cd "$TARGET_DIR" || exit 1

	echo "==> Downloading $ZIP_NAME" 
	wget -c "$URL"
	echo "==> Extracting $ZIP_NAME"
	unzip -o "$ZIP_NAME"

	cd - >/dev/null
done

echo "All selected datasets downloaded successfully."
