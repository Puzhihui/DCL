@echo off
D:
cd D:\Solution\code\smic\DCL\gen_defect_imgs
@REM CALL D:\Anaconda3\Scripts\activate.bat
call conda activate yolo8
python merge_defect.py --imagedata F:\ImageData --other_bright 20 --other_dark 20 --scratch_dark 4
pause