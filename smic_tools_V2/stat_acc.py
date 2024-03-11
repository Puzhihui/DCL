import sys
sys.path.insert(0, '../')
import os
import glob
import argparse
import datetime

from utils import write_csv, get_added_lot
from config import LoadConfig


def manual_result_stat(mode_path):
    manual_reslut_dict = dict()
    for category in multi_classes:
        img_list = glob.glob(os.path.join(mode_path, category, "*.bmp"))
        new_img_list = []
        for img in img_list:
            label = os.path.basename(img).split("@")[0]
            if label not in multi_classes:
                continue
            new_img_list.append(img)
        manual_reslut_dict[category] = len(new_img_list)
    return manual_reslut_dict


def adc_reslut_stat(mode_path):
    adc_reslut_dict = dict()
    for category in multi_classes:
        adc_reslut_dict[category] = 0
    img_list = glob.glob(os.path.join(mode_path, "*", "*.bmp"))
    for img in img_list:
        label = os.path.basename(img).split("@")[0]
        if label not in adc_reslut_dict.keys():
            continue
        adc_reslut_dict[label] += 1
    return adc_reslut_dict


def find_adc_wrong_img(mode_path, label):
    img_list = glob.glob(os.path.join(mode_path, "*", "*.bmp"))
    wrong = 0
    for img in img_list:
        manual = img.split("\\")[-2]
        if manual == label or manual not in multi_classes:
            continue
        adc = os.path.basename(img).split("@")[0]
        if adc not in multi_classes:
            continue
        if manual != adc and adc == label:
            wrong += 1
    return wrong


def adc_wrong_stat(mode_path):
    adc_wrong_dict = dict()
    for category in multi_classes:
        adc_wrong_dict[category] = 0
    for category in multi_classes:
        adc_wrong_dict[category] = find_adc_wrong_img(mode_path, category)
    return adc_wrong_dict


def stat_lot_wafer(lot_path):
    img_list = glob.glob(os.path.join(lot_path, "*", "*.bmp"))
    wafer_dict = set()
    for img in img_list:
        wafer = os.path.basename(img).split('_')[-2].split('-')[-1]
        wafer_dict.add(wafer)
    return len(list(wafer_dict))


def stat(lot_path):
    # 统计ADC结果
    adc_reslut_dict = adc_reslut_stat(lot_path)
    # ADC错误数量
    adc_wrong_dict = adc_wrong_stat(lot_path)
    # 人工复判后每个类别的数量
    manual_reslut_dict = manual_result_stat(lot_path)
    wafer_num = stat_lot_wafer(lot_path)
    return adc_reslut_dict, adc_wrong_dict, manual_reslut_dict, wafer_num


def get_folder_time(folder_path):
    timestamp = os.path.getctime(folder_path)
    folder_time = datetime.datetime.fromtimestamp(timestamp)
    folder_time_str = folder_time.strftime("%Y%m%d")
    return folder_time_str


def get_result(result_dict, header, header_start):
    result = []
    for category in multi_classes:
        header.append("{}_{}".format(header_start, category))
        result.append(result_dict[category])
    return result, header


def txtrow(recipe, lot_id, lot_time, wafer_num, adc_reslut_dict, adc_wrong_dict, manual_reslut_dict):
    row = [lot_time, recipe, lot_id, wafer_num]
    header = ["datetime", "recipe", "lot_num", "wafer_num"]
    result, header = get_result(adc_reslut_dict, header, "adc")
    row.extend(result)
    result, header = get_result(adc_wrong_dict, header, "wrong")
    row.extend(result)
    result, header = get_result(manual_reslut_dict, header, "manual")
    row.extend(result)
    global headers
    if not headers:
        headers = header

    return row


def stat_recipe(reviewed_path, recipe_dict):
    rows_all = []
    for time_recipe_lot in os.listdir(reviewed_path):
        time_recipe_lot_path = os.path.join(reviewed_path, time_recipe_lot)
        if not os.path.isdir(time_recipe_lot_path):
            continue
        lot_time, recipe, lot = time_recipe_lot.split("@")

        pass_lot_set = recipe_dict[recipe] if recipe in recipe_dict.keys() else []
        if lot in pass_lot_set:
            continue

        adc_reslut_dict_bb, adc_wrong_dict_bb, manual_reslut_dict_bb, wafer_num = stat(time_recipe_lot_path)
        row_lot = txtrow(recipe, lot, lot_time, wafer_num, adc_reslut_dict_bb, adc_wrong_dict_bb, manual_reslut_dict_bb)
        rows_all.append(row_lot)
    return rows_all


global multi_classes
headers = None


def parse_args():
    parser = argparse.ArgumentParser(description='stat acc')
    parser.add_argument('--mode', default='Back', type=str)
    parser.add_argument('--img_path', default=r'D:\Solution\datas\get_report', type=str)
    parser.add_argument('--client', default='M47', type=str)
    args = parser.parse_args()
    return args

added_txt = "added_lot.txt"
added_txt_Back = "added_lot_Back.txt"

if __name__ == '__main__':
    print("-----------------------------------calculate ACC-----------------------------------")
    args = parse_args()
    mode = args.mode
    if mode == "Back":
        added_txt = added_txt_Back
    client = args.client
    dataset = "{}_{}".format(mode, client)
    args.dataset, args.swap_num, args.backbone = dataset, None, None
    cfg = LoadConfig(args, 'train', True)
    global multi_classes
    multi_classes = list(cfg.multi_classes.keys())

    print("{}: Start calculate ACC!".format(mode))
    args.is_all_recipe = True
    mode_reviewed_path = os.path.join(args.img_path, args.mode, "reviewed")

    pass_lot = get_added_lot(added_txt)
    rows_all = stat_recipe(mode_reviewed_path, pass_lot)
    csv_path = os.path.join(os.path.join(args.img_path, args.mode, "reviewed"),
                            "{}_{}_{}_{}_{}_report.csv".format(datetime.datetime.now().year,
                                                                     datetime.datetime.now().month,
                                                                     datetime.datetime.now().day,
                                                                     datetime.datetime.now().hour,
                                                                     datetime.datetime.now().minute))
    if rows_all:
        write_csv(csv_path, rows_all, headers)
    else:
        print("reviewed 文件夹中的所有批次以前已统计，本次无更新！")
    print("-----------------------------------calculate ACC-----------------------------------\n\n\n")
