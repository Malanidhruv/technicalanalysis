@echo off
cd /d "%~dp0"
echo Starting Harion Research Stock Screener...
echo Open http://localhost:8501 in your browser
python -m streamlit run app.py
