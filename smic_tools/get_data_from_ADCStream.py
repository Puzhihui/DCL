import os, shutil, glob
import datetime
import json
import csv
import argparse
import requests
import random


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
        # ADCBin = per_record["ADCBin"]
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
        manual_result[adc_img_path] = {"ADCBin": ADCBin, "BinCode": BinCode, "adc_result": adc_result, "DefectImagePath": DefectImagePath}
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

        for adc_img_path, per_manual_result in manual_result.items():
            DefectImagePath = per_manual_result["DefectImagePath"]
            if camera not in DefectImagePath:
                continue
            adc_label = per_manual_result["adc_result"]["label"]
            manual_label = bincode_dict[per_manual_result["BinCode"]]
            ADC_stat[adc_label] += 1
            manual_stat[manual_label] += 1
            print("m: {}, adc: {}".format(manual_label, adc_label))
            copy_flag = False
            save_path = os.path.join(save_img_path, str(update_time_int), recipe, lot, wafer, FrontOrBack, camera, manual_label)
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
            
def copy_data_from_ADCStream(recipe, lot, FrontOrBack, camera):
	adc_stream = os.path.join(wafer_path, FrontOrBack, camera, "ADC", "ADCStream.json")
	adc_result = parse_ADCStream(adc_stream)
	for img_path, per_adc_result in adc_result.items():
	    label = per_adc_result["label"]
	    save_img_path = os.path.join(save_path, recipe, lot, FrontOrBack, camera, label)
	    os.makedirs(save_img_path, exist_ok=True)
	    shutil.copy2(img_path,save_img_path)

def copy_data_from_dir(wafer_path, recipe, lot, FrontOrBack, camera, mode):
    img_list = glob.glob(os.path.join(wafer_path, FrontOrBack, camera, "*", "*.bmp"))
    for img_path in img_list:
        label = img_path.split("\\")[-2]
        save_img_path = os.path.join(save_path, mode, label)
        os.makedirs(save_img_path, exist_ok=True)
        shutil.copy2(img_path, save_img_path)


def parse_args():
    parser = argparse.ArgumentParser(description='get report')
    parser.add_argument('--mode', default='Back', type=str)
    parser.add_argument('--imagedata', default=r'F:\ImageData', type=str)
    parser.add_argument('--save_img_path', default=r'D:\Solution\datas\get_report', type=str)
    parser.add_argument('--txt', default=r'D:\Solution\datas\get_report\report.txt', type=str)
    parser.add_argument('--sample', default=25, type=int)
    args = parser.parse_args()
    return args


def requests_data(recipe, lot, mode, requests_path):
    response = requests.post(url, data=requests_path)
    results = response.text
    results = json.loads(results)
    print(requests_path)
    if results["errorcode"] != 0:
        print("server error, errorcode:{}, msg: {}".format(results["errorcode"], results["msg"]))
        data_list = []
    else:
        data_list = results["data"]
    for data in data_list:
        image_name = data["image_name"]
        basename = image_name.split("\\")[-1]
        confidence = data["confidence"]
        label = data["label"]
        # print("{}:\tpred_label:{}\tconfidence:{}\t".format(basename, label, confidence))
        save_img_path = os.path.join(save_path, recipe, lot, mode, label)
        os.makedirs(save_img_path, exist_ok=True)
        try:
            shutil.copy2(image_name, save_img_path)
            basename = os.path.basename(image_name)
            os.rename(os.path.join(save_img_path, basename), os.path.join(save_img_path, "{}@{}".format(label, basename)))
        except:
            continue


args = parse_args()
sample = args.sample
print("每批次随机取{}片".format(sample))
from_path = args.imagedata
save_path = os.path.join(args.save_img_path, "{}_{}_{}_{}".format(datetime.datetime.now().year, datetime.datetime.now().month, datetime.datetime.now().day, datetime.datetime.now().hour))
txt = args.txt
mode = args.mode
url = "http://10.0.2.101:3081/ADC/"
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

def main():
    recipe_list = os.listdir(from_path)
    print("start ADC:")
    for recipe in recipe_list:
        if recipe not in recipe_dict.keys():
            continue
        recipe_path = os.path.join(from_path, recipe)
        if not os.path.isdir(recipe_path):
            continue

        lot_list = os.listdir(recipe_path)
        target_lot_list = recipe_dict[recipe]
        for lot in lot_list:
            if lot not in target_lot_list:
                continue
            print(recipe, lot)
            lot_path = os.path.join(recipe_path, lot)
            lot_path_stat = os.stat(lot_path)
            if not os.path.isdir(lot_path):
                continue

            wafer_list = os.listdir(lot_path)
            random.shuffle(wafer_list)
            for wafer in wafer_list[:sample]:
                wafer_path = os.path.join(lot_path, wafer)
                if not os.path.isdir(wafer_path):
                    continue
                for i in range(5):
                    FrontBright = os.path.join(recipe_path, lot, wafer, "Front", "FrontCamera1_BrightField{}".format(i), "ADC\\")
                    FrontDark = os.path.join(recipe_path, lot, wafer, "Front", "FrontCamera3_DarkField{}".format(i), "ADC\\")
                    BackBright = os.path.join(recipe_path, lot, wafer, "Back", "BackCamera1_BrightField{}".format(i), "ADC\\")
                    BackDark = os.path.join(recipe_path, lot, wafer, "Back", "BackCamera3_DarkField{}".format(i), "ADC\\")
                    if mode == "Front":
                        requests_data(recipe, lot, "Front", FrontBright)
                        requests_data(recipe, lot, "FrontDark", FrontDark)
                    elif mode == "Back":
                        requests_data(recipe, lot, "Back", BackBright)
                        requests_data(recipe, lot, "BackDark", BackDark)

main()
