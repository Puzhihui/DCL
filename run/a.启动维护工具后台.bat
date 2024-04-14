title ADC_tools
@echo off
D:
cd D:\Solution\code\maintain_tool
@REM CALL D:\Anaconda3\Scripts\activate.bat
call conda activate torch1.0_cuda8.0
python main.py
pause