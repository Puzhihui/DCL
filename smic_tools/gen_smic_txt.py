import sys
sys.path.insert(0, '../')
import os
import glob
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
    cfg_mode = smic_back_online()
elif mode == "Front":
    cfg_mode = smic_front_online()
else:
    raise "Mode error!!!"

multi_classes = {'discolor': "0", 'other': "1", 'scratch': "2", "false": "3", 'cScratch': "4"}

train_data_path = os.path.join(cfg_mode.train_data_path, "train")
val_data_path = os.path.join(cfg_mode.train_data_path, "val")

txt_root_path = cfg_mode.txt_root_path
os.makedirs(txt_root_path, exist_ok=True)
f_train = open(os.path.join(txt_root_path, 'train.txt'), 'w', encoding='utf-8')
f_val = open(os.path.join(txt_root_path, 'val.txt'), 'w', encoding='utf-8')


def img2txt(img_list, label, f_txt):
    copy_num = 0
    for per_img in img_list:
        img_folder = per_img.split('\\')[4:]
        dst_img_name = '\\'.join(img_folder)
        f_txt.write(dst_img_name + ', ' + label + '\n')
        copy_num += 1
    return copy_num


def glob_img(train_path, f_txt):
    discolor = glob.glob(os.path.join(train_path, "*", "*", "discolor", "*.bmp"))
    img2txt(discolor, multi_classes["discolor"], f_txt)
    scratch = glob.glob(os.path.join(train_path, "*", "*", "scratch", "*.bmp"))
    img2txt(scratch, multi_classes["scratch"], f_txt)
    other = glob.glob(os.path.join(train_path, "*", "*", "other", "*.bmp"))
    img2txt(other, multi_classes["other"], f_txt)
    false = glob.glob(os.path.join(train_path, "*", "*", "false", "*.bmp"))
    img2txt(false, multi_classes["false"], f_txt)
    cScratch = glob.glob(os.path.join(train_path, "*", "*", "cScratch", "*.bmp"))
    if len(cScratch) > 0:
        img2txt(cScratch, multi_classes["cScratch"], f_txt)
    return len(discolor), len(other), len(scratch), len(false), len(cScratch)


print('2.start:开始生成数据集txt文件')
discolor, other, scratch, false, cScratch = glob_img(train_data_path, f_train)
print("训练集：共{}, discolor:{}, other:{}, scratch:{}, false:{}, cScratch:{}".format(discolor+other+scratch+false, discolor, other, scratch, false, cScratch))
discolor, other, scratch, false, cScratch = glob_img(val_data_path, f_val)
print("验证集：共{}, discolor:{}, other:{}, scratch:{}, false:{}, cScratch:{}".format(discolor+other+scratch+false, discolor, other, scratch, false, cScratch))

f_train.close()
f_val.close()
print('训练集文件为{}\n验证集文件为{}'.format(os.path.join(txt_root_path, 'train.txt'), os.path.join(txt_root_path, 'val.txt')))
print('-------------------------------------------------')
