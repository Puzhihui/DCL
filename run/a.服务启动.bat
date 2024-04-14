@echo off
start cmd /K "D:&&cd D:\Solution\code\smic\automatic_defect_classification_server\service&&conda activate torch1.0_cuda8.0&&title server_offline&&python server_smic.py"
@REM start cmd /K "D:&&cd D:\Solution\code\smic\automatic_defect_classification_server\service&&conda activate torch1.0_cuda8.0&&title server_eagle&&python server_smic_eagle.py"
@REM pause