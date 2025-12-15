#!/bin/bash

# Permissions: chmod +x download_datasets.sh
# Run: ./download_datasets.sh

set -e # Exit on error

# CONFIGURATION: Dataset keys (order matters only for readability)
DATASET_KEYS=(
	"CICDDoS2019_CSV_01_12"
	"CICDDoS2019_CSV_03_11"
	"CICIDS2017"
)

# Per-key URL and target directory (edit/comment/uncomment as needed)
DATASET_URL_CICDDoS2019_CSV_01_12="http://cicresearch.ca/CICDataset/CICDDoS2019/Dataset/CSVs/CSV-01-12.zip"
DATASET_URL_CICDDoS2019_CSV_03_11="http://cicresearch.ca/CICDataset/CICDDoS2019/Dataset/CSVs/CSV-03-11.zip"
DATASET_URL_CICIDS2017="http://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/GeneratedLabelledFlows.zip"

DATASET_DIR_CICDDoS2019_CSV_01_12="Datasets/CICDDoS2019"
DATASET_DIR_CICDDoS2019_CSV_03_11="Datasets/CICDDoS2019"
DATASET_DIR_CICIDS2017="Datasets/CICIDS2017"

# DOWNLOAD AND EXTRACT DATASETS
# Prerequisites
command -v wget >/dev/null 2>&1 || { echo "ERROR: 'wget' is required but not found. Install it and retry." >&2; exit 1; }
command -v unzip >/dev/null 2>&1 || { echo "ERROR: 'unzip' is required but not found. Install it and retry." >&2; exit 1; }

mkdir -p Datasets # Create main Datasets directory if it doesn't exist
for DATASET in "${DATASET_KEYS[@]}"; do
	# Build variable names and read them via eval for POSIX/bash-3 compatibility
	URL_VAR="DATASET_URL_${DATASET}"
	DIR_VAR="DATASET_DIR_${DATASET}"
	URL=$(eval "echo \"\${${URL_VAR}}\"")
	TARGET_DIR=$(eval "echo \"\${${DIR_VAR}}\"")

	# Skip entries with empty URL
	if [ -z "${URL}" ]; then
		echo "==> Skipping ${DATASET}: no URL configured"
		continue
	fi

	ZIP_NAME="$(basename "${URL}")"

	echo "==> Setting up dataset: ${DATASET}"
	echo "    From: ${URL}"
	echo "    To:   ${TARGET_DIR}/${ZIP_NAME}"
	mkdir -p "${TARGET_DIR}"
	echo "==> Changing to directory ${TARGET_DIR}"
	cd "${TARGET_DIR}" || exit 1

	echo "==> Downloading ${ZIP_NAME}"
	wget -c "${URL}"
	echo "==> Extracting ${ZIP_NAME}"
	unzip -o "${ZIP_NAME}"

	cd - >/dev/null
done

echo "All selected datasets downloaded successfully."
