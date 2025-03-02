#!/bin/bash

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p /home/fac/krishnandu/miniconda
    eval ""
fi

# Create and activate conda environment
conda env create -f environment.yml
conda activate data_processing_env

# Verify GPU availability
nvidia-smi

# Run the processing script
python main.py

