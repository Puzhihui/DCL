import os
import glob
import argparse
import datetime

from utils import get_recipe_lot, write_csv

def parse_args():
    parser = argparse.ArgumentParser(description='stat acc')
    parser.add_argument('--mode', default='Back', type=str)
    parser.add_argument('--imagedata', default=r'F:\ImageData', type=str)
    parser.add_argument('--img_path', default=r'D:\Solution\datas\get_report', type=str)
    parser.add_argument('--is_all_recipe', action='store_true')
    args = parser.parse_args()
    return args


def manual_result_stat(mode_path):
    manual_reslut_dict = dict()
    for category in categories:
        img_list = glob.glob(os.path.join(mode_path, category, "*.bmp"))
        new_img_list = []
        for img in img_list:
            label = os.path.basename(img).split("@")[0]
            if label not in categories:
                continue
            new_img_list.append(img)
        manual_reslut_dict[category] = len(new_img_list)
    return manual_reslut_dict


def adc_reslut_stat(mode_path):
    adc_reslut_dict = dict()
    for category in categories:
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
        if manual == label or manual not in categories:
            continue
        adc = os.path.basename(img).split("@")[0]
        if adc not in categories:
            continue
        if manual != adc and adc == label:
            wrong += 1
    return wrong


def adc_wrong_stat(mode_path):
    adc_wrong_dict = dict()
    for category in categories:
        adc_wrong_dict[category] = 0
    for category in categories:
        adc_wrong_dict[category] = find_adc_wrong_img(mode_path, category)
    return adc_wrong_dict


def stat_lot_wafer(lot_path):
    img_list = glob.glob(os.path.join(lot_path, "*", "*.bmp"))
    wafer_dict = set()
    for img in img_list:
        wafer = os.path.basename(img).split('_')[4].split('-')[0]
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


def txtrow(recipe, lot_id, wafer_num, adc_reslut_dict, adc_wrong_dict, manual_reslut_dict):
    row = [recipe, lot_id, wafer_num,
           adc_reslut_dict["scratch"], adc_reslut_dict["cScratch"], adc_reslut_dict["discolor"], adc_reslut_dict["other"], adc_reslut_dict["PASD"], adc_reslut_dict["SINR"], adc_reslut_dict["false"],
           adc_wrong_dict["scratch"], adc_wrong_dict["cScratch"], adc_wrong_dict["discolor"], adc_wrong_dict["other"], adc_wrong_dict["PASD"], adc_wrong_dict["SINR"], adc_wrong_dict["false"],
           manual_reslut_dict["scratch"], manual_reslut_dict["cScratch"], manual_reslut_dict["discolor"], manual_reslut_dict["other"], manual_reslut_dict["PASD"], manual_reslut_dict["SINR"], manual_reslut_dict["false"],
           ]
    return row


def stat_recipe(reviewed_path, recipe_dict):
    rows_all = []
    for recipe in os.listdir(reviewed_path):
        if recipe not in recipe_dict.keys():
            continue
        recipe_path = os.path.join(reviewed_path, recipe)
        if not os.path.isdir(recipe_path):
            continue
        lot_list = os.listdir(recipe_path)
        target_lot_list = recipe_dict[recipe]
        rows_recipe = []
        for lot in lot_list:
            if lot not in target_lot_list:
                continue
            lot_path = os.path.join(recipe_path, lot)
            if not os.path.isdir(lot_path):
                continue
            adc_reslut_dict_bb, adc_wrong_dict_bb, manual_reslut_dict_bb, wafer_num = stat(lot_path)
            row_lot = txtrow(recipe, lot, wafer_num, adc_reslut_dict_bb, adc_wrong_dict_bb, manual_reslut_dict_bb)
            rows_recipe.append(row_lot)
            rows_all.append(row_lot)
        recipe_csv_path = os.path.join(recipe_path, "{}_ACC.csv".format(recipe))
        write_csv(recipe_csv_path, rows_recipe, headers)
    return rows_all

categories = ["discolor", "false", "other", "scratch", "cScratch", "PASD", "SINR"]
headers = ["recipe", "lot_num", "wafer_num",
           "adc_scratch", "adc_cScratch", "adc_discolor", "adc_other", "adc_PASD", "adc_SINR", "adc_false",
           "wrong_scratch", "wrong_cScratch", "wrong_discolor", "wrong_other", "wrong_PASD", "wrong_SINR", "wrong_false",
           "manual_scratch", "manual_cScratch", "manual_discolor", "manual_other", "manual_PASD", "manual_SINR", "manual_false"
           ]

if __name__ == '__main__':
    args = parse_args()
    print("Start stat ACC, PlEASE CHECK, model: {}, is_all_recipes: {}".format(args.mode, args.is_all_recipe))
    target_recipe_dict = get_recipe_lot(args)
    rows_all = stat_recipe(os.path.join(args.img_path, args.mode, "reviewed"), target_recipe_dict)
    csv_path = os.path.join(os.path.join(args.img_path, args.mode),
                            "{}_{}_{}_{}_{}_report.csv".format(datetime.datetime.now().year,
                                                                     datetime.datetime.now().month,
                                                                     datetime.datetime.now().day,
                                                                     datetime.datetime.now().hour,
                                                                     datetime.datetime.now().minute))
    write_csv(csv_path, rows_all, headers)
