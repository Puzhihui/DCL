import sys
sys.path.insert(0, '../')
import os
import shutil
import datetime
from config import smic_back_online, smic_front_online
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='replace online model')
    parser.add_argument('--mode', default='Back', type=str)
    args = parser.parse_args()
    return args

args = parse_args()
mode = args.mode
if mode == "Back":
    cfg = smic_back_online()
elif mode == "Front":
    cfg = smic_front_online()
else:
    raise "Mode error!!!"
best_path = None
online_model_dir = cfg.online_model_dir
online_model_name = cfg.online_model_name
if os.path.exists(cfg.best_model_txt):
    f = open(cfg.best_model_txt, "r", encoding="utf-8")
    best_path = f.readline()
    f.close()
if best_path:
    best_model_path = os.path.join(best_path, online_model_name)
    if not os.path.exists(best_model_path):
        print("文件不存在: {}".format(best_model_path))
        raise "file not exists"
    print("找到最新的模型路径为：{}".format(best_model_path))
    print("请核对模型路径是否正确，若不正确，请手动更新模型")
    online_model_path = os.path.join(online_model_dir, online_model_name)
    now = datetime.datetime.now()
    now_str = '{}y{:02d}m{:02d}d{:02d}h{:02d}m'.format(now.year, now.month, now.day, now.hour, now.minute)
    try:
        shutil.copy2(online_model_path, os.path.join(online_model_dir, "{}@{}".format(now_str, online_model_name)))
        os.remove(online_model_path)
        print("原模型备份：{} -> {}".format(online_model_name, "{}@{}".format(now_str, online_model_name)))
    except:
        print("{}模型备份失败，直接进行模型替换".format(online_model_path))
    try:
        shutil.copy2(os.path.join(best_path, online_model_name), online_model_dir)
        print("新模型已替换完毕！")
    except:
        print("{}模型不存在，请手动更新模型".format(os.path.join(best_path, online_model_name)))
f = open(cfg.best_model_txt, "w", encoding="utf-8")
f.write("")
f.close()

