import os, shutil, glob
import datetime
import json
import csv
import argparse


def parse_ADCStream(adc_stream_path):
    adc_result = dict()
    if not os.path.exists(adc_stream_path):
        return adc_result
    with open(adc_stream_path, 'r', encoding='utf-8') as f:
        ret_dic = json.load(f)
    if ret_dic['errorcode'] != 0:
        return adc_result
    data_list = ret_dic['data']
    for per_data_dict in data_list:
        image_name = per_data_dict["image_name"]
        confidence = per_data_dict["confidence"]
        label = per_data_dict["label"]
        adc_result[image_name] = {"confidence": confidence, "label": label}
    return adc_result


def write_csv(csv_path, rows, headers):
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(rows)


def parse_ManualResult(manual_path):
    manual_result = dict()
    if not os.path.exists(manual_path):
        return manual_result
    with open(manual_path, 'r', encoding='utf-8') as f:
        ret_dic = json.load(f)
    record = ret_dic["Record"]
    if not record:
        return manual_result
    for per_record in record:
        IsReview = per_record["IsReview"]
        if not IsReview and not stat_unreviewed:
            continue
        try:
            Source = bincode_dict[per_record["Source"]]
        except:
            continue
        BinCode = per_record["BinCode"]
        DefectImagePath = per_record["DefectImagePath"]
        aDCDataCollection = per_record["aDCDataCollection"]
        if len(aDCDataCollection) > 1 and adc_img_split:
            raise "is img split mode"
        if len(aDCDataCollection) == 0:
            continue
        adc_result = aDCDataCollection[0]["Imagedata"] if len(aDCDataCollection) != 0 else None
        # print(aDCDataCollection[0]["Imagedata"]["label"])
        ADCBin = bincode_str2int_dict[aDCDataCollection[0]["Imagedata"]["label"]]
        adc_img_path = aDCDataCollection[0]["Imagedata"]["image_name"]
        manual_result[adc_img_path] = {"ADCBin": ADCBin,"Source":Source ,"BinCode": BinCode, "adc_result": adc_result, "DefectImagePath": DefectImagePath}
    return manual_result


def stat_mode(update_time_int, recipe, lot, lot_path, FrontOrBack, camera):
    wafer_list = os.listdir(lot_path)
    ADC_stat = {"scratch": 0, "discolor": 0, "other": 0, "false": 0}
    ADC_wrong_stat = {"scratch": 0, "discolor": 0, "other": 0, "false": 0}
    manual_stat = {"scratch": 0, "discolor": 0, "other": 0, "false": 0}
    for wafer in wafer_list:
        wafer_path = os.path.join(lot_path, wafer)
        if not os.path.isdir(wafer_path):
            continue
        adc_stream = os.path.join(wafer_path, FrontOrBack, camera, "ADC", "ADCStream.json")
        adc_result = parse_ADCStream(adc_stream)

        manual_path = os.path.join(wafer_path, FrontOrBack, "ManualResult", "UserIdentifyADCResult.json")
        manual_result = parse_ManualResult(manual_path)
        #print(manual_result)
        if len(adc_result) == 0 or len(manual_result) == 0:
            continue
        # print(len(manual_result))
        for adc_img_path, per_manual_result in manual_result.items():
            DefectImagePath = per_manual_result["DefectImagePath"]
            Source = per_manual_result["Source"]
            if camera not in DefectImagePath:
                # print(camera, DefectImagePath)
                continue
            adc_label = per_manual_result["adc_result"]["label"]
            manual_label = bincode_dict[per_manual_result["BinCode"]]
            ADC_stat[adc_label] += 1
            manual_stat[manual_label] += 1
            print("m: {}, adc: {}".format(manual_label, adc_label))
            copy_flag = False
            save_path = os.path.join(save_img_path, str(update_time_int), recipe, lot, wafer, FrontOrBack, camera, manual_label)
            if Source != adc_label:
                # print("offline adc != adc server")
                raise "offline adc != adc server"
            if manual_label != adc_label:
                ADC_wrong_stat[adc_label] += 1
                if save_adc_wrong_img:
                    os.makedirs(save_path, exist_ok=True)
                    try:
                        shutil.copy2(adc_img_path, save_path)
                        basename = os.path.basename(adc_img_path)
                        dst_name = "{}@{}".format(adc_label, basename)
                        os.rename(os.path.join(save_path, basename), os.path.join(save_path, dst_name))
                        copy_flag = True
                    except:
                        continue
            if copy_all_img and not copy_flag:
                os.makedirs(save_path, exist_ok=True)
                try:
                    shutil.copy2(adc_img_path, save_path)
                    basename = os.path.basename(adc_img_path)
                    dst_name = "right@{}".format(basename)
                    os.rename(os.path.join(save_path, basename), os.path.join(save_path, dst_name))
                    copy_flag = True
                except:
                    continue

    return ADC_stat, ADC_wrong_stat, manual_stat


def parse_args():
    parser = argparse.ArgumentParser(description='get report')
    parser.add_argument('--imagedata', default=r'F:\ImageData', type=str)
    parser.add_argument('--save_img_path', default=r'D:\Solution\datas\get_report', type=str)
    parser.add_argument('--txt', default=r'D:\Solution\datas\get_report\report.txt', type=str)
    args = parser.parse_args()
    return args

args = parse_args()
imagedata = args.imagedata
save_img_path = args.save_img_path
result = dict()
txt = args.txt
f = open(txt, 'r', encoding='utf-8')
lines = f.readlines()
f.close()
recipe_dict = dict()
for line in lines:
    line = line.replace("\n", "")
    recipe, lot = line.split('\t')[0]+"_VSI_OM", line.split('\t')[1]
    print(recipe, lot)
    if recipe not in recipe_dict.keys():
        recipe_dict[recipe] = []
    recipe_dict[recipe].append(lot)

stat_unreviewed = False  # 统计未review的图片
adc_img_split = False  # 是否会切割小图
save_adc_wrong_img = True
copy_all_img = True
bincode_dict =         {"OTHER": "other", "SCRATCH": "scratch", "DISCOLOR": "discolor", "FALSE": "false"}
bincode_str2int_dict = {"other": "OTHER", "scratch": "SCRATCH", "discolor": "DISCOLOR", "false": "FALSE"}
# 获取文件信息
recipe_list = os.listdir(imagedata)
for recipe in recipe_list:
    if recipe not in recipe_dict.keys():
        continue
    recipe_path = os.path.join(imagedata, recipe)
    if not os.path.isdir(recipe_path):
        continue
    result[recipe] = dict()

    lot_list = os.listdir(recipe_path)
    target_lot_list = recipe_dict[recipe]
    for lot in lot_list:
        if lot not in target_lot_list:
            continue
        result[recipe][lot] = dict()
        lot_path = os.path.join(recipe_path, lot)
        lot_path_stat = os.stat(lot_path)
        # create_time = datetime.datetime.fromtimestamp(file_stat.st_ctime)
        update_time = datetime.datetime.fromtimestamp(lot_path_stat.st_mtime)
        update_time_int = int('{}{:02d}{:02d}'.format(update_time.year, update_time.month, update_time.day))
        if update_time_int < date_interval[0] or update_time_int > date_interval[1]:
            continue
        if not os.path.isdir(lot_path):
            continue

        ADC_stat, ADC_wrong_stat, manual_stat = stat_mode(update_time_int, recipe, lot, lot_path, "Front", "FrontCamera1_BrightField1")
        result[recipe][lot]["front"] = {"date":update_time_int, "ADC_stat": ADC_stat, "ADC_wrong_stat": ADC_wrong_stat, "manual_stat": manual_stat}
        ADC_stat, ADC_wrong_stat, manual_stat = stat_mode(update_time_int, recipe, lot, lot_path, "Back", "BackCamera1_BrightField1")
        result[recipe][lot]["backbright"] = {"date":update_time_int, "ADC_stat": ADC_stat, "ADC_wrong_stat": ADC_wrong_stat, "manual_stat": manual_stat}
        ADC_stat, ADC_wrong_stat, manual_stat = stat_mode(update_time_int, recipe, lot, lot_path, "Back", "BackCamera3_DarkField1")
        result[recipe][lot]["backdark"] = {"date":update_time_int, "ADC_stat": ADC_stat, "ADC_wrong_stat": ADC_wrong_stat, "manual_stat": manual_stat}

rows = []
for recipe, recipe_result in result.items():
    for lot, lot_result in recipe_result.items():
        for mode, mode_result in lot_result.items():
            date = mode_result["date"]
            ADC_stat = mode_result["ADC_stat"]
            ADC_wrong_stat = mode_result["ADC_wrong_stat"]
            manual_stat = mode_result["manual_stat"]
            rows.append([date, recipe, lot, mode, ADC_stat["scratch"], ADC_stat["discolor"], ADC_stat["other"], ADC_stat["false"],
                                            ADC_wrong_stat["scratch"], ADC_wrong_stat["discolor"], ADC_wrong_stat["other"], ADC_wrong_stat["false"],
                                            manual_stat["scratch"], manual_stat["discolor"], manual_stat["other"], manual_stat["false"]])

headers = ['date', 'recipe', 'lot', 'mode', 'ADC_scratch数量', 'ADC_discolor数量', 'ADC_other数量', 'ADC_alse数量',
                                    'Wrong_scratch数量', 'Wrong_discolor数量', 'Wrong_other数量', 'Wrong_false数量',
                                    'final_scratch数量', 'final_discolor数量', 'final_other数量', 'final_false数量']
write_csv(os.path.join(save_img_path, '{}_{}_{}_{}_{}_om_report.csv'.format(datetime.datetime.now().year, datetime.datetime.now().month,
                                                                         datetime.datetime.now().day, datetime.datetime.now().hour,
                                                                         datetime.datetime.now().minute)),
          rows, headers)
