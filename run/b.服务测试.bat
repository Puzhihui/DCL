title test_server
@echo off
D:
cd D:\Solution\code\smic\automatic_defect_classification_server\service\tools
@REM CALL D:\Anaconda3\Scripts\activate.bat
call conda activate torch1.0_cuda8.0
python http_test_smic.py --url http://127.0.0.1:3081/ADC/  --data D:\Solution\datas\test_smic_server\
pause