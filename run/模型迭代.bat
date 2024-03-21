@echo off
D:
cd D:\Solution\code\smic\DCL\smic_tools_V2
@REM CALL D:\Anaconda3\Scripts\activate.bat
call conda activate torch1.0_cuda8.0

set client=M6
set imagedata=F:\ImageData
set img_path=D:\Solution\datas\get_report

echo Back:  train Back model
echo Front: train Front model
:input
set /p userInput=Please input command, Back or Front:
echo ==================================================

@REM python stat_acc.py --mode %userInput% --img_path %img_path% --client %client%
@REM python get_data_from_reviewed.py --mode %userInput% --img_path %img_path% --client %client%

cd D:\Solution\code\smic\DCL\smic_tools
python gen_smic_txt.py --mode %userInput% --client %client%

cd D:\Solution\code\smic\DCL\
python train_smic.py --mode %userInput% --epoch 25 --client %client%

cd D:\Solution\code\smic\DCL\smic_tools
python replace_online_model.py --mode %userInput%
pause