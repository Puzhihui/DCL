@echo off
D:
@REM CALL D:\Anaconda3\Scripts\activate.bat
call conda activate torch1.0_cuda8.0

set client=%1
set train_model=%2
set epoch=%3
set batch_size=%4
set resume_checkpoint=%5
set replace_online_model=%6
set log_interval=%7
set num_workers=%8

cd D:\Solution\code\smic\DCL\
set dataset_name=%train_model%_%client%
python train_smic.py --data %dataset_name% --tb %batch_size% --vb %batch_size% ^
                     --tnw %num_workers% --vnw %num_workers% --backbone resnet50 ^
                     --epoch %epoch%  --resume_checkpoint %resume_checkpoint% ^
                     --replace_online_model %replace_online_model% --log_interval %log_interval%

@REM pause