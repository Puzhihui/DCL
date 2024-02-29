@echo off
D:
cd D:\Solution\code\smic\DCL\smic_tools_V2
@REM CALL D:\Anaconda3\Scripts\activate.bat
call conda activate torch1.0_cuda8.0

echo 1. Reuse ADC classification data on all recipes and lots
echo 2. Reuse ADC classification data on report.txt
echo 3. Statistical accuracy on all recipes and lots
echo 4. Statistical accuracy on report.txt
echo Back or Front concat 1 or 2 or 3 or 4, example:Back1 Front4
:input
set /p userInput=Please input command, example Back4:
echo ==================================================
if "%userInput%"=="Back1" (
    python classify_by_server.py --mode Back --is_all_recipe
) else if "%userInput%"=="Back2" (
    python classify_by_server.py --mode Back
) else if "%userInput%"=="Back3" (
    python stat_acc.py --mode Back --is_all_recipe
) else if "%userInput%"=="Back4" (
    python stat_acc.py --mode Back
) else if "%userInput%"=="Front1" (
    python classify_by_server.py --mode Front --is_all_recipe
) else if "%userInput%"=="Front2" (
    python classify_by_server.py --mode Front
) else if "%userInput%"=="Front3" (
    python stat_acc.py --mode Front --is_all_recipe --data F:\Solution\datas\get_report
) else if "%userInput%"=="Front4" (
    python stat_acc.py --mode Front
) else (
    echo Input ERROR!! Please input:
    goto input
)
pause