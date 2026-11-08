#!/usr/bin/env bash
# run_dashboard.sh
# Launcher script for NIRF Streamlit Dashboard using conda environment 'myenv'

PROJECT_DIR="/home/rishav/MY_PERSONAL_WORKS/DHV/ProjectAssignment"
STREAMLIT_BIN="/home/rishav/miniconda3/envs/myenv/bin/streamlit"
PORT=8501

cd "$PROJECT_DIR" || exit 1

echo "=========================================================="
echo " Starting NIRF Analytics Streamlit Dashboard (2017 - 2025)"
echo " Project Directory: $PROJECT_DIR"
echo " Streamlit URL:     http://localhost:$PORT"
echo "=========================================================="

"$STREAMLIT_BIN" run app.py --server.port $PORT --server.headless true
