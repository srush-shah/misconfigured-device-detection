# Complete Setup Guide

This guide provides step-by-step instructions to clone the repository, set up the environment, and run the application successfully on your local system.

## Prerequisites

Before starting, ensure you have the following installed:

- **Python 3.9** or higher (Python 3.9 recommended for compatibility)
- **Conda** or **Miniconda** (recommended for environment management)
- **Git** (for cloning the repository)
- **Jupyter Notebook** (for training models - will be installed via requirements)
- **At least 4GB RAM** (8GB+ recommended for model training)
- **Disk Space**: ~2GB for dataset and dependencies

**Optional but Recommended:**
- **CUDA-capable GPU** (for faster deep learning model training, but CPU works fine)

## Step 1: Clone the Repository

```bash
# Clone the repository from GitHub
git clone https://github.com/srush-shah/misconfigured-device-detection.git
cd misconfigured-device-detection
```

## Step 2: Set Up Python Environment

We recommend using Conda for environment management, but you can also use `venv` or `virtualenv`.

### Option A: Using Conda (Recommended)

```bash
# Create a new conda environment with Python 3.9
conda create -n detect_device python=3.9 -y

# Activate the environment
conda activate detect_device

# Verify Python version
python --version
# Should show: Python 3.9.x
```

### Option B: Using venv

```bash
# Create virtual environment
python3.9 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Verify Python version
python --version
```

## Step 3: Install Dependencies

```bash
# Make sure you're in the project root directory
cd misconfigured-device-detection

# Ensure your environment is activated
# (You should see (detect_device) or (venv) in your terminal prompt)

# Upgrade pip to latest version
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

# Verify installation (optional)
pip list | grep -E "pandas|numpy|scikit-learn|torch|streamlit"
```

**Expected Output**: You should see all packages listed with their versions.

**Troubleshooting**:
- If you encounter errors with `torch`, try installing it separately:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```
- For GPU support (if you have CUDA):
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

## Step 4: Download and Set Up the Dataset

The project uses the **Westermo Network Traffic Dataset**. Follow these steps:

### 4.1 Download the Dataset

1. Visit: https://www.westermo.com/support/downloads/network-traffic-dataset
   - Alternative: https://github.com/westermo/network-traffic-dataset
2. Download the dataset archive (usually a `.zip` or `.tar.gz` file)

### 4.2 Extract the Dataset

```bash
# Navigate to the data directory
cd misconfigured-device-detection

# Create the directory structure if it doesn't exist
mkdir -p data/raw/westermo

# Extract the downloaded dataset to data/raw/westermo/
# Replace 'dataset.zip' with your actual filename
unzip dataset.zip -d data/raw/westermo/
# OR if it's a tar.gz:
# tar -xzf dataset.tar.gz -C data/raw/westermo/
```

### 4.3 Verify Dataset Structure

After extraction, your directory structure should look like this:

```
data/raw/westermo/
├── README.md
├── LICENSE
├── topology.svg
└── data/
    ├── events.txt                    # Required: Contains misconfiguration event timestamps
    ├── reduced/
    │   ├── flows/
    │   │   ├── output_bottom.zip     # Required: Flow data
    │   │   ├── output_left.zip       # Required: Flow data
    │   │   └── output_right.zip      # Required: Flow data
    │   └── pcaps/
    │       ├── bottom.zip            # Optional: Raw packet captures
    │       ├── left.zip
    │       └── right.zip
    └── extended/
        ├── flows/
        │   ├── output_bottom.zip    # Optional: Extended flow data
        │   ├── output_left.zip
        │   └── output_right.zip
        └── pcaps/
            ├── bottom.zip
            ├── left.zip
            └── right.zip
```

**Verify the critical file exists:**
```bash
# Check that events.txt exists
ls -la data/raw/westermo/data/events.txt

# Check that flow zip files exist
ls -la data/raw/westermo/data/reduced/flows/*.zip
```

**Note**: The `reduced/flows/` directory is required. The `extended/` and `pcaps/` directories are optional but recommended for better results.

## Step 5: Train the Models

You have two options for training: **Jupyter Notebook** (recommended) or **Command Line**.

### Option A: Using Jupyter Notebook (Recommended)

```bash
# Make sure your environment is activated
conda activate detect_device  # or: source venv/bin/activate

# Start Jupyter Notebook
jupyter notebook

# This will open a browser window. Navigate to:
# notebooks/01_misconfig_detection.ipynb
```

**In the Notebook:**

1. **Run all cells sequentially** (Cell → Run All, or Shift+Enter for each cell)
2. **Wait for completion**: Training can take 10-30 minutes depending on your hardware
3. **Save models**: The notebook will automatically save trained models to `models/saved/`

**Expected Output:**
- Data loading progress
- Feature extraction statistics
- Model training progress for each model type
- Performance metrics (accuracy, balanced accuracy, recall, etc.)
- Models saved to `models/saved/*.pkl`

**Key Cells to Pay Attention To:**
- **Cell 3**: Data ingestion - verify it finds the Westermo dataset
- **Cell 17**: Model saving - ensure models are saved successfully
- **Final cells**: Evaluation metrics and visualizations

### Option B: Using Command Line

```bash
# Make sure your environment is activated
conda activate detect_device

# Run the pipeline script
# Note: This requires pre-processed CSV files
python run_pipeline.py --flow-csv path/to/flows.csv --output data/output/misconfig_report.csv
```

**Note**: The command-line option requires pre-processed flow CSV files. For first-time setup, use the Jupyter Notebook method.

## Step 6: Verify Model Training

After training, verify that models were saved:

```bash
# Check that model files exist
ls -la models/saved/

# Expected files:
# - improved_rf_model.pkl (or enhanced_rf_model.pkl)
# - ensemble_model.pkl (or enhanced_ensemble_model.pkl)
# - feature_columns.pkl
```

**If models are missing:**
- Re-run the notebook, specifically Cell 17 (model saving cell)
- Check for any error messages in the notebook output
- Ensure you have write permissions in the `models/saved/` directory

## Step 7: Run the Web Application

Once models are trained, you can run the interactive web application:

```bash
# Make sure your environment is activated
conda activate detect_device

# Navigate to web_app directory
cd web_app

# Run the Streamlit app
streamlit run app.py

# OR use the provided script
bash run_app.sh
```

**Expected Output:**
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

**Using the Web Application:**

1. **Open your browser** and navigate to `http://localhost:8501`
2. **Single Device Analysis Tab**:
   - Enter device features manually, or
   - Upload a CSV file with device data
   - Click "Analyze Device" to get predictions
3. **Batch Analysis Tab**:
   - Upload a CSV file with multiple devices
   - Click "Analyze All Devices"
   - View summary statistics and download results

**Note**: The web app can run in "demo mode" without trained models, but predictions will be less accurate.

## Step 8: Verify Complete Setup

Run this verification script to check your setup:

```bash
# From project root
python -c "
import sys
import os

print('=== Setup Verification ===')
print(f'Python version: {sys.version}')

# Check critical imports
try:
    import pandas, numpy, sklearn, torch, streamlit
    print('✓ All core packages imported successfully')
except ImportError as e:
    print(f'✗ Import error: {e}')

# Check dataset
if os.path.exists('data/raw/westermo/data/events.txt'):
    print('✓ Westermo dataset found')
else:
    print('✗ Westermo dataset not found')

# Check models
if os.path.exists('models/saved/') and len(os.listdir('models/saved/')) > 0:
    print('✓ Trained models found')
else:
    print('⚠ Trained models not found (run notebook to train)')

print('=== Verification Complete ===')
"
```

## Expected Results After Complete Setup

After following all steps, you should be able to:

1. ✅ **Load the dataset** without errors
2. ✅ **Train models** and see performance metrics:
   - Balanced Accuracy: ~0.91
   - Misconfiguration Recall: ~100%
   - F1-Score: ~0.50
3. ✅ **Run the web application** and make predictions
4. ✅ **Generate explanations** for predictions

## Troubleshooting Common Issues

### Issue 1: Import Errors

**Symptoms**: `ModuleNotFoundError` or `ImportError`

**Solutions**:
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Check environment activation
which python  # Should point to your conda/venv environment

# Verify installation
pip list | grep <package-name>
```

### Issue 2: Dataset Not Found

**Symptoms**: `FileNotFoundError: events.txt` or similar

**Solutions**:
```bash
# Verify dataset location
ls -la data/raw/westermo/data/events.txt

# Check path in notebook (Cell 3)
# Ensure path is correct relative to project root

# Re-download and extract dataset if missing
```

### Issue 3: Model Training Fails

**Symptoms**: Training crashes or hangs

**Solutions**:
- **Memory issues**: Reduce batch size in notebook (Cell 3: `batch_size=16` instead of 32)
- **Slow training**: This is normal on CPU; expect 10-30 minutes
- **CUDA errors**: Set `device='cpu'` in notebook configuration

### Issue 4: Web App Won't Start

**Symptoms**: `streamlit: command not found` or port already in use

**Solutions**:
```bash
# Install streamlit
pip install streamlit

# Use different port
streamlit run app.py --server.port 8502

# Kill existing streamlit process
pkill -f streamlit
```

### Issue 5: Models Not Loading in Web App

**Symptoms**: "Models not found" warning in web app

**Solutions**:
- Verify models exist: `ls -la models/saved/`
- Re-run notebook Cell 17 to save models
- Check file permissions: `chmod 644 models/saved/*.pkl`

## Quick Start Summary

For experienced users, here's the condensed setup:

```bash
# 1. Clone and setup
git clone <repo-url> && cd misconfigured-device-detection
conda create -n detect_device python=3.9 -y
conda activate detect_device
pip install -r requirements.txt

# 2. Download dataset to data/raw/westermo/

# 3. Train models
jupyter notebook notebooks/01_misconfig_detection.ipynb
# Run all cells

# 4. Run web app
cd web_app && streamlit run app.py
```

## Next Steps

After successful setup:

1. **Explore the Notebook**: Review each cell to understand the pipeline
2. **Experiment with Models**: Try different hyperparameters in the notebook
3. **Process Your Own Data**: Use `scripts/process_realtime_data.py` for your network data
4. **Customize Features**: Modify feature extraction in `features/feature_aggregator.py`
5. **Read Documentation**: Review the README.md for architecture details

## Getting Help

If you encounter issues not covered here:

1. Check the **README.md** for additional troubleshooting
2. Review **notebook comments** for detailed explanations
3. Check **GitHub Issues** (if repository has issue tracking)
4. Verify your Python version matches requirements (3.9+)

