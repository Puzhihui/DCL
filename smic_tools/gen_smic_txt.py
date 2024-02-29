import sys
sys.path.insert(0, '../')
import os
import glob
from collections import defaultdict
from config import smic_back_online, smic_front_online, LoadConfig

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='replace online model')
    parser.add_argument('--mode', default='Back', type=str)
    parser.add_argument('--client', default='M47', type=str)
    args = parser.parse_args()
    return args

args = parse_args()
client = args.client
mode = args.mode

dataset = "{}_{}".format(mode, client)
args.dataset, args.swap_num, args.backbone = dataset, None, None
cfg = LoadConfig(args, 'train', True)
multi_classes = cfg.multi_classes

train_data_path = cfg.train_path
val_data_path = cfg.val_path
txt_root_path = cfg.anno_root
os.makedirs(txt_root_path, exist_ok=True)
f_train = open(os.path.join(txt_root_path, 'train.txt'), 'w', encoding='utf-8')
f_val = open(os.path.join(txt_root_path, 'val.txt'), 'w', encoding='utf-8')


def img2txt(img_list, label, f_txt):
    copy_num = 0
    for per_img in img_list:
        f_txt.write(per_img + ', ' + label + '\n')
        copy_num += 1
    return copy_num


def glob_img(data_path, f_txt, data="train"):
    data_dict = defaultdict(int)
    for category, label in multi_classes.items():
        img_list = glob.glob(os.path.join(data_path, "*", category, "*.bmp"))
        data_dict[category] += img2txt(img_list, label, f_txt)

    total_sum = sum(data_dict.values())
    output = f"{data}: " + f"共{total_sum}, "
    for key, value in data_dict.items():
        output += f"{key}: {value}, "
    output = output.rstrip(", ")
    print(output)
    return data_dict

print('2.start:开始生成数据集txt文件')
train_dict = glob_img(train_data_path, f_train, "train")
val_dict = glob_img(val_data_path, f_val, "val")
f_train.close()
f_val.close()
print('训练集文件为{}\n验证集文件为{}'.format(os.path.join(txt_root_path, 'train.txt'), os.path.join(txt_root_path, 'val.txt')))
print('-------------------------------------------------')
