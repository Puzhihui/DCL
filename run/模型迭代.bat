@echo off
D:
@REM CALL D:\Anaconda3\Scripts\activate.bat
call conda activate pytorch_dcl_pyinstall

set client=%1
set process_stage=%2
set epoch=%3
set batch_size=%4
set resume_checkpoint=%5
set replace_online_model=%6
set log_interval=%7
set num_workers=%8

set dataset_name=%client%_%process_stage%

cd D:\Solution\code\DCL
python train_my.py --data %dataset_name% --tb %batch_size% --vb %batch_size% ^
                   --tnw %num_workers% --vnw %num_workers% --crop 448  --cls_mul --backbone efficientnet-b4 ^
                   --epoch %epoch% --resume_checkpoint %resume_checkpoint% ^
                   --replace_online_model %replace_online_model% --log_interval %log_interval%
@REM pause