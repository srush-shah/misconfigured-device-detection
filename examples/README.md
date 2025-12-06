# Examples

This folder contains example data files for testing the web application.

## Synthetic Test Data

- `synthetic_balanced_test_data.csv` - Balanced dataset with 50 normal and 49 misconfigured devices (includes label column)
- `synthetic_batch_test_data.csv` - Dataset with 80 devices (60% normal, 40% misconfigured)

## Generating Synthetic Data

To generate your own synthetic test data:

**Balanced dataset (50% normal, 50% misconfigured):**
```bash
python generate_balanced_synthetic_data.py
```

**Batch dataset (60% normal, 40% misconfigured):**
```bash
python generate_batch_synthetic_data.py
```

These scripts create CSV files with both normal and misconfigured devices that can be uploaded to the web application for batch testing.

## Usage

Upload these CSV files to the web application's "Batch Analysis" tab to test the misconfiguration detection system.

