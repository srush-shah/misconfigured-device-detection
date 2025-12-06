#!/bin/bash
# Script to run the Streamlit web application

echo "=========================================="
echo "Network Device Misconfiguration Detector"
echo "Web Application"
echo "=========================================="
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing..."
    pip install streamlit plotly
fi

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to script directory
cd "$SCRIPT_DIR"

# Run Streamlit
echo "🚀 Starting web application..."
echo "📱 Open your browser at http://localhost:8501"
echo ""

streamlit run app.py

