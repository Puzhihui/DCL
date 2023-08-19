import sys
sys.path.insert(0, '../')
import os
import shutil
import glob
import argparse
import datetime
import csv
# from config import smic_back_online, smic_front_online

def parse_args():
    parser = argparse.ArgumentParser(description='replace online model')
    parser.add_argument('--mode', default='Back', type=str)
    args = parser.parse_args()
    return args


def get_str_datetime():
    return str(datetime.datetime.now().year) + "{:02d}".format(datetime.datetime.now().month) + "{:02d}".format(datetime.datetime.now().day)


def write2csv(save_csv_path, write_type, headers, rows):
    with open(save_csv_path, write_type, newline='', encoding='utf-8')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(rows)


args = parse_args()
mode = args.mode

smic_data_root = r"D:\Solution\datas\smic_data"
if mode == "Back":
    csv_path = os.path.join(smic_data_root, "Back", "trained_lot.csv")
    smic_data = os.path.join(smic_data_root, "Back")
elif mode == "Front":
    csv_path = os.path.join(smic_data_root, "Front", "trained_lot.csv")
    smic_data = os.path.join(smic_data_root, "Front")

trained_dict = dict()
if os.path.exists(csv_path):
    f = open(csv_path, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()

    for line in lines:
        line = line.replace("\n", "")
        recipe, lot = line.split(",")[0], line.split(",")[1]
        if recipe not in trained_dict.keys():
            trained_dict[recipe] = set()
        trained_dict[recipe].add(lot)

rows = []
headers = ["recipe", "lot"]
date_today = get_str_datetime()
trained_dict_now = dict()
if mode == "Back":
    back_data_path = r''
    img_list = glob.glob(os.path.join(back_data_path, "*", "*", "*", "*", "*", "*.bmp"))
    for img in img_list:
        img_info_list = img.split("\\")
        recipe, lot = img_info_list[-5], img_info_list[-4]
        if recipe in trained_dict.keys():
            trained_lot = list(trained_dict[recipe])
            if lot in trained_lot:
                continue
        op_label = img_info_list[-2]
        adc_label = img_info_list[-1].split("@")[0]
        mode = img_info_list[-3]
        save_img_path = os.path.join(smic_data, date_today, "{}_{}".format(date_today, mode), mode, op_label)
        os.makedirs(save_img_path, exist_ok=True)
        shutil.copy2(img, save_img_path)

        if recipe not in trained_dict_now.keys():
            trained_dict_now[recipe] = set()
        trained_dict_now[recipe].add(lot)

    for recipe, lot_set in trained_dict_now.items():
        lot_list = list(lot_set)
        for lot in lot_list:
            rows.append([recipe, lot])
    write2csv(csv_path, "a+", headers, rows)
