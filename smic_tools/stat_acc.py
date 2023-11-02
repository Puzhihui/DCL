import os, glob
import csv, shutil
import datetime, argparse


def parse_args():
    parser = argparse.ArgumentParser(description='stat acc')
    parser.add_argument('--mode', default='Front', type=str)
    parser.add_argument('--data', default=r'D:\Solution\datas\get_report', type=str)
    parser.add_argument('--txt', default=r'D:\Solution\datas\get_report\report.txt', type=str)
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


def find_adc_wrong_img(mode_path, label, save_img=False):
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
            if save_img:
                underkill_path = os.path.join(path, "underkill", "{}_{}_{}_{}_{}".format(datetime.datetime.now().year,
                                                                                         datetime.datetime.now().month,
                                                                                         datetime.datetime.now().day,
                                                                                         datetime.datetime.now().hour,
                                                                                         datetime.datetime.now().minute),
                                              manual)
                os.makedirs(underkill_path, exist_ok=True)
                try:
                    shutil.copy2(img, underkill_path)
                except:
                    continue
    return wrong


def adc_wrong_stat(mode_path):
    adc_wrong_dict = dict()
    for category in categories:
        adc_wrong_dict[category] = 0
    adc_wrong_dict["discolor"] = find_adc_wrong_img(mode_path, "discolor")
    adc_wrong_dict["false"] = find_adc_wrong_img(mode_path, "false", True)
    adc_wrong_dict["other"] = find_adc_wrong_img(mode_path, "other")
    adc_wrong_dict["scratch"] = find_adc_wrong_img(mode_path, "scratch")
    adc_wrong_dict["cScratch"] = find_adc_wrong_img(mode_path, "cScratch")
    adc_wrong_dict["PASD"] = find_adc_wrong_img(mode_path, "PASD")
    adc_wrong_dict["SINR"] = find_adc_wrong_img(mode_path, "SINR")
    return adc_wrong_dict


def stat_lot_wafer(lot_path):
    img_list = glob.glob(os.path.join(lot_path, "*", "*.bmp"))
    lot_dict = dict()
    for img in img_list:
        lot, wafer = os.path.basename(img).split('_')[4].split('-')[0], os.path.basename(img).split('_')[4].split('-')[
            1]
        if lot not in lot_dict:
            lot_dict[lot] = set()
        lot_dict[lot].add(wafer)
    wafer_num = 0
    for lot, wafers in lot_dict.items():
        wafer_num += len(wafers)
    return len(lot_dict.keys()), wafer_num


def stat(lot_path):
    # mode_path = os.path.join(lot_path, mode)
    # 统计ADC结果
    adc_reslut_dict = adc_reslut_stat(lot_path)
    # ADC错误数量
    adc_wrong_dict = adc_wrong_stat(lot_path)
    # 人工复判后每个类别的数量
    manual_reslut_dict = manual_result_stat(lot_path)
    lot_num, wafer_num = stat_lot_wafer(lot_path)
    return adc_reslut_dict, adc_wrong_dict, manual_reslut_dict, lot_num, wafer_num


def write_csv(csv_path, rows, headers):
    if os.path.exists(csv_path):
        with open(csv_path, 'a+', newline='', encoding='utf-8') as f:
            f_csv = csv.writer(f)
            f_csv.writerows(rows)
    else:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(headers)
            f_csv.writerows(rows)


def txtrow(recipe, lot_id, wafer_num, adc_reslut_dict, adc_wrong_dict, manual_reslut_dict):
    row = [recipe, lot_id, wafer_num, adc_reslut_dict["scratch"], adc_reslut_dict["cScratch"],
           adc_reslut_dict["discolor"], adc_reslut_dict["other"], adc_reslut_dict["PASD"], adc_reslut_dict["SINR"],
           adc_reslut_dict["false"], adc_wrong_dict["scratch"], adc_wrong_dict["cScratch"], adc_wrong_dict["discolor"],
           adc_wrong_dict["other"], adc_wrong_dict["PASD"], adc_wrong_dict["SINR"], adc_wrong_dict["false"],
           manual_reslut_dict["scratch"], manual_reslut_dict["cScratch"], manual_reslut_dict["discolor"],
           manual_reslut_dict["other"], manual_reslut_dict["PASD"], manual_reslut_dict["SINR"],
           manual_reslut_dict["false"], ]
    return row


# back_categories = ["discolor", "false", "other", "scratch", "cScratch"]
# front_categories = ["discolor", "false", "other", "scratch", "PASD", "SINR"]
categories = ["discolor", "false", "other", "scratch", "cScratch", "PASD", "SINR"]
args = parse_args()
mode = args.mode
if mode == "Back":
    # cfg_mode = smic_back_online()
    path = os.path.join(args.data, "Back")
    bright = "Back"
    dark = "BackDark"  # categories = ["discolor", "false", "other", "scratch", "cScratch"]
elif mode == "Front":
    # cfg_mode = smic_front_online()
    path = os.path.join(args.data, "Front")
    bright = "Front"
    dark = "FrontDark"  # categories = ["discolor", "false", "other", "scratch", "PASD", "SINR"]
else:
    raise "Mode error!!!"
txt = os.path.join(path, "report.txt")
f = open(txt, 'r', encoding='utf-8')
lines = f.readlines()
f.close()
recipe_dict = dict()
for line in lines:
    line = line.replace("\n", "")
    recipe, lot = line.split('\t')[0] + "_VSI_OM", line.split('\t')[1]
    recipe_test, lot_test = line.split('\t')[0] + "_VSI_OM-Test", line.split('\t')[1]
    print(recipe, lot)
    if recipe not in recipe_dict.keys():
        recipe_dict[recipe] = []
    recipe_dict[recipe].append(lot)

    if recipe_test not in recipe_dict.keys():
        recipe_dict[recipe_test] = []
    recipe_dict[recipe_test].append(lot_test)

date_list = os.listdir(path)
rows = []
for per_date in date_list:
    date_path = os.path.join(path, per_date)
    if not os.path.isdir(date_path):
        continue
    recipe_list = os.listdir(date_path)
    for recipe in recipe_list:
        if recipe not in recipe_dict.keys():
            continue
        recipe_path = os.path.join(date_path, recipe)
        if not os.path.isdir(recipe_path):
            continue
        lot_list = os.listdir(recipe_path)
        target_lot_list = recipe_dict[recipe]
        for lot in lot_list:
            if lot not in target_lot_list:
                continue
            lot_path = os.path.join(recipe_path, lot)
            if not os.path.isdir(lot_path):
                continue
            adc_reslut_dict_bb, adc_wrong_dict_bb, manual_reslut_dict_bb, lot_num, wafer_num = stat(lot_path)
            row_bb = txtrow(recipe, lot, wafer_num, adc_reslut_dict_bb, adc_wrong_dict_bb, manual_reslut_dict_bb)
            rows.append(row_bb)

headers = ["recipe", "lot_num", "wafer_num", "adc_scratch", "adc_cScratch", "adc_discolor", "adc_other", "adc_PASD",
           "adc_SINR", "adc_false", "wrong_scratch", "wrong_cScratch", "wrong_discolor", "wrong_other", "wrong_PASD",
           "wrong_SINR", "wrong_false", "manual_scratch", "manual_cScratch", "manual_discolor", "manual_other",
           "manual_PASD", "manual_SINR", "manual_false"]
csv_path = os.path.join(path,
                        "{}_{}_{}_{}_{}_report.csv".format(datetime.datetime.now().year, datetime.datetime.now().month,
                                                           datetime.datetime.now().day, datetime.datetime.now().hour,
                                                           datetime.datetime.now().minute))
print(len(rows))
write_csv(csv_path, rows, headers)

