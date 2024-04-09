@echo off
D:
cd D:\Solution\code\smic\DCL\smic_tools_V2
@REM CALL D:\Anaconda3\Scripts\activate.bat
call conda activate torch1.0_cuda8.0

set imagedata=F:\ImageData
set client=%1
set mode=%2
set epoch=%3
set batch_size=%4

echo ==================================================

cd D:\Solution\code\smic\DCL\smic_tools
python gen_smic_txt.py --mode %mode% --client %client%

cd D:\Solution\code\smic\DCL\
python train_smic.py --mode %mode% --epoch %epoch% --client %client% --tb %batch_size% --vb %batch_size%

cd D:\Solution\code\smic\DCL\smic_tools
python replace_online_model.py --mode %mode%
pause