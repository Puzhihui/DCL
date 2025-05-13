@echo off
D:
cd D:\Solution\code\smic\DCL\smic_tools
@REM CALL D:\Anaconda3\Scripts\activate.bat
call conda activate yolo8
python delete_disc.py --imagedata F:\ImageData --days_thres 15
pause