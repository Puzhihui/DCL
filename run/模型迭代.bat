@echo off
D:
cd D:\Solution\code\smic\DCL\smic_tools_V2
@REM CALL D:\Anaconda3\Scripts\activate.bat
call conda activate torch1.0_cuda8.0

set client=M47

echo Back:  train Back model
echo Front: train Front model
:input
set /p userInput=Please input command, Back or Front:
echo ==================================================
python get_data_from_reviewed.py --mode %userInput% --img_path D:\Solution\datas\get_report

cd D:\Solution\code\smic\DCL\smic_tools
python gen_smic_txt.py --mode %userInput% --client %client%

cd D:\Solution\code\smic\DCL\
python train_smic.py --mode %userInput% --epoch 2 --client %client%

cd D:\Solution\code\smic\DCL\smic_tools
python replace_online_model.py --mode %userInput%
pause