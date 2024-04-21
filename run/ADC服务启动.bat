title ADC_SERVER
@echo off
D:
cd D:\Solution\code\smic\automatic_defect_classification_server\service
@REM CALL D:\Anaconda3\Scripts\activate.bat
call conda activate torch1.0_cuda8.0
python server_smic_eagle.py
@REM pause