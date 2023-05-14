import os, shutil, glob
import datetime
import json
import csv


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
        ADCBin = per_record["ADCBin"]
        BinCode = per_record["BinCode"]
        DefectImagePath = per_record["DefectImagePath"]
        aDCDataCollection = per_record["aDCDataCollection"]
        if len(aDCDataCollection) > 1 and not adc_img_split:
            raise "is img split mode"
        adc_result = aDCDataCollection[0]["Imagedata"] if len(aDCDataCollection) != 0 else None
        manual_result[DefectImagePath] = {"ADCBin": ADCBin, "BinCode": BinCode, "adc_result": adc_result}
    return manual_result


def stat_mode(update_time_int, recipe, lot_path, FrontOrBack, camera):
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
        if len(adc_result) == 0 or len(manual_result) == 0:
            continue
        for adc_img_path, per_adc_result in adc_result.items():
            adc_label = per_adc_result["label"]
            ADC_stat[adc_label] += 1

            defect_img = adc_img_path.replace("\\ADC", "")
            if defect_img not in manual_result.keys():
                raise "{} not in manual_result".format(defect_img)
            manual_label = bincode_dict[manual_result[defect_img]["BinCode"]]
            if manual_label != adc_label:
                ADC_wrong_stat[manual_label] += 1
                if save_adc_wrong_img:
                    save_path = os.path.join(save_img_path, str(update_time_int), recipe, FrontOrBack, camera, manual_label)
                    os.makedirs(save_path, exist_ok=True)
                    try:
                        shutil.copy2(adc_img_path, save_path)
                        basename = os.path.basename(adc_img_path)
                        dst_name = "{}@{}".format(adc_label, basename)
                        os.rename(os.path.join(save_path, basename), os.path.join(save_path, dst_name))
                    except:
                        continue
            manual_stat[manual_label] += 1
    return ADC_stat, ADC_wrong_stat, manual_stat


imagedata = r'D:\Solution\ADC_stat\ImageData'
save_img_path = r''
result = dict()
date_interval = [20230501, 20230530]

stat_unreviewed = False  # 统计未review的图片
adc_img_split = True  # 是否会切割小图
save_adc_wrong_img = True
bincode_dict = {0: "discolor", 1: "scratch", 2: "other"}
# 获取文件信息
recipe_list = os.listdir(imagedata)
for recipe in recipe_list:
    recipe_path = os.path.join(imagedata, recipe)
    if not os.path.isdir(recipe_path):
        continue
    result[recipe] = dict()

    lot_list = os.listdir(recipe_path)
    for lot in lot_list:
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

        ADC_stat, ADC_wrong_stat, manual_stat = stat_mode(update_time_int, recipe, lot_path, "Front", "FrontCamera1_BrightField1")
        result[recipe][lot]["front"] = {"ADC_stat": ADC_stat, "ADC_wrong_stat": ADC_wrong_stat, "manual_stat": manual_stat}
        ADC_stat, ADC_wrong_stat, manual_stat = stat_mode(update_time_int, recipe, lot_path, "Back", "BackCamera1_BrightField1")
        result[recipe][lot]["backbright"] = {"ADC_stat": ADC_stat, "ADC_wrong_stat": ADC_wrong_stat, "manual_stat": manual_stat}
        ADC_stat, ADC_wrong_stat, manual_stat = stat_mode(update_time_int, recipe, lot_path, "Back", "BackCamera3_DarkField1")
        result[recipe][lot]["backdark"] = {"ADC_stat": ADC_stat, "ADC_wrong_stat": ADC_wrong_stat, "manual_stat": manual_stat}

rows = []
for recipe, recipe_result in result.items():
    for lot, lot_result in recipe_result.items():
        for mode, mode_result in lot_result.items():
            ADC_stat = mode_result["ADC_stat"]
            ADC_wrong_stat = mode_result["ADC_wrong_stat"]
            manual_stat = mode_result["manual_stat"]
            rows.append([recipe, lot, mode, ADC_stat["scratch"], ADC_stat["discolor"], ADC_stat["other"], ADC_stat["false"],
                                            ADC_wrong_stat["scratch"], ADC_wrong_stat["discolor"], ADC_wrong_stat["other"], ADC_wrong_stat["false"],
                                            manual_stat["scratch"], manual_stat["discolor"], manual_stat["other"], manual_stat["false"]])

headers = ['recipe', 'lot', 'mode', 'ADC_scratch数量', 'ADC_discolor数量', 'ADC_other数量', 'ADC_alse数量',
                                    'Wrong_scratch数量', 'Wrong_discolor数量', 'Wrong_other数量', 'Wrong_false数量',
                                    'final_scratch数量', 'final_discolor数量', 'final_other数量', 'final_false数量']
write_csv(os.path.join(save_img_path, '{}_{}_{}_{}_{}_om_report.csv'.format(datetime.datetime.now().year, datetime.datetime.now().month,
                                                                         datetime.datetime.now().day, datetime.datetime.now().hour,
                                                                         datetime.datetime.now().minute)),
          rows, headers)


# 复盘时按下，Bincode字段是什么
# false类别在json文件中是怎么记录的
# 0 1 2分别对应哪个类别