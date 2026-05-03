@echo off
call conda activate f1predictor
streamlit run "%~dp0app.py"
