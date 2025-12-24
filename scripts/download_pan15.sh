#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RAW_DIR="$PROJECT_ROOT/data/raw"

mkdir -p "$RAW_DIR"
cd "$RAW_DIR"

echo "=== Downloading PAN15 Author Profiling Dataset ==="

TRAIN_URL="https://zenodo.org/records/3745945/files/pan15-author-profiling-training-dataset-2015-04-23.zip?download=1"
TEST_URL="https://zenodo.org/records/3745945/files/pan15-author-profiling-test-dataset-2015-04-23.zip?download=1"

if [ ! -f "pan15_train.zip" ]; then
    echo "Downloading training set..."
    curl -L -o pan15_train.zip "$TRAIN_URL"
else
    echo "Training set already downloaded."
fi

if [ ! -f "pan15_test.zip" ]; then
    echo "Downloading test set..."
    curl -L -o pan15_test.zip "$TEST_URL"
else
    echo "Test set already downloaded."
fi

echo "=== Extracting datasets ==="

if [ ! -d "pan15_train" ]; then
    echo "Extracting training set..."
    unzip -o pan15_train.zip -d pan15_train
else
    echo "Training set already extracted."
fi

if [ ! -d "pan15_test" ]; then
    echo "Extracting test set..."
    unzip -o pan15_test.zip -d pan15_test
else
    echo "Test set already extracted."
fi

echo "=== Extracting English subset ==="

ENGLISH_ZIP=$(find pan15_train -name "*english*.zip" -o -name "*English*.zip" | head -1)

if [ -n "$ENGLISH_ZIP" ] && [ ! -d "pan15_train_en" ]; then
    echo "Found English subset: $ENGLISH_ZIP"
    mkdir -p pan15_train_en
    unzip -o "$ENGLISH_ZIP" -d pan15_train_en
else
    if [ -d "pan15_train_en" ]; then
        echo "English subset already extracted."
    else
        echo "Warning: English subset zip not found. Looking for English folder..."
        ENGLISH_DIR=$(find pan15_train -type d -iname "*english*" | head -1)
        if [ -n "$ENGLISH_DIR" ]; then
            echo "Found English directory: $ENGLISH_DIR"
            mkdir -p pan15_train_en
            cp -r "$ENGLISH_DIR"/* pan15_train_en/ 2>/dev/null || true
        fi
    fi
fi

echo "=== Download complete ==="
echo "Data location: $RAW_DIR"
ls -la "$RAW_DIR"

