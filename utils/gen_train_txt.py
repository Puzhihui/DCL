import os
import glob
from collections import defaultdict


def img2txt(img_list, label, f_txt):
    copy_num = 0
    for per_img in img_list:
        f_txt.write(per_img + ', ' + label + '\n')
        copy_num += 1
    return copy_num


def glob_img(data_path, f_txt, multi_classes, log_server, data_print="train"):
    data_dict = defaultdict(int)
    for category, label in multi_classes.items():
        img_list = glob.glob(os.path.join(data_path, "*", category, "*.bmp"))
        img_list.extend(glob.glob(os.path.join(data_path, "*", category, "*.jpg")))
        img_list.extend(glob.glob(os.path.join(data_path, "*", category, "*.jpeg")))
        data_dict[category] += img2txt(img_list, label, f_txt)

    total_sum = sum(data_dict.values())
    output = f"{data_print}: " + f"共{total_sum}, "
    for key, value in data_dict.items():
        output += f"{key}: {value}, "
    output = output.rstrip(", ")
    log_server.logging(output)
    return data_dict


def generate_txt(cfg, log_server):
    txt_root_path = cfg.anno_root
    os.makedirs(txt_root_path, exist_ok=True)
    f_train = open(os.path.join(txt_root_path, 'train.txt'), 'w', encoding='utf-8')
    f_val = open(os.path.join(txt_root_path, 'val.txt'), 'w', encoding='utf-8')
    log_server.logging('开始生成数据集txt文件......')
    multi_classes = cfg.multi_classes

    for train_path in cfg.train_path_list:
        train_dict = glob_img(train_path, f_train, multi_classes, log_server, os.path.basename(train_path)+"_train")
    for val_path in cfg.val_path_list:
        val_dict = glob_img(val_path, f_val, multi_classes, log_server, os.path.basename(val_path)+"_val")
    f_train.close()
    f_val.close()
    print('训练集文件为{}\n验证集文件为{}'.format(os.path.join(txt_root_path, 'train.txt'), os.path.join(txt_root_path, 'val.txt')))
    log_server.logging('-------------------------------------------------')