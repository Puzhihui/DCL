import os
import glob
import shutil
import argparse
import requests
import json

from utils import get_recipe_lot


def parse_args():
    parser = argparse.ArgumentParser(description='get report')
    parser.add_argument('--mode', default='Back', type=str)
    parser.add_argument('--imagedata', default=r'F:\ImageData', type=str)
    parser.add_argument('--img_path', default=r'D:\Solution\datas\get_report', type=str)
    parser.add_argument('--is_all_recipe', action='store_true')
    # parser.add_argument('--reviewed_path', default=r'D:\Solution\datas\get_report\reviewed', type=str)
    # parser.add_argument('--txt', default=r'D:\Solution\datas\get_report\report.txt', type=str)
    # parser.add_argument('--sample', default=25, type=int)
    args = parser.parse_args()
    return args


def warning_message(args):
    print("请将{}中已经review的lot移动至{}".format(args.mode, os.path.join(args.img_path, args.mode, "reviewed")))
    user_input = "None"
    while(user_input.lower() != 'y'):
        user_input = input("移动完成后请输入y后回车:")


def requests_data(requests_path):
    response = requests.post(url, data=requests_path)
    results = response.text
    results = json.loads(results)
    print(requests_path)
    if results["errorcode"] != 0:
        print("server error, errorcode:{}, msg: {}".format(results["errorcode"], results["msg"]))
        data_list = []
    else:
        data_list = results["data"]
    img_result = dict()
    for data in data_list:
        image_name = data["image_name"]
        basename = image_name.split("\\")[-1]
        label = data["label"]
        img_result[basename] = label
    return img_result


def infer_by_server(args, pending_recipe_dict):
    pending_review_path = os.path.join(args.img_path, args.mode, "pending_review")
    for recipe in os.listdir(pending_review_path):
        if recipe not in pending_recipe_dict.keys():
            continue
        recipe_path = os.path.join(pending_review_path, recipe)
        target_lot_list = list(pending_recipe_dict[recipe])
        for lot in os.listdir(recipe_path):
            if lot not in target_lot_list:
                continue
            lot_path = os.path.join(recipe_path, lot)
            for old_label in os.listdir(lot_path):
                img_result = requests_data(os.path.join(lot_path, old_label))
                img_list = glob.glob(os.path.join(lot_path, old_label, "*.bmp"))
                for img in img_list:
                    basename = os.path.basename(img)
                    if basename in img_result:
                        new_label = img_result[basename]
                        if "@" in basename:
                            img_old_label = basename.split('@')[0]
                            basename_no_label = basename.replace("{}@".format(img_old_label), "")
                            dst_img_name = "{}@{}".format(new_label, basename_no_label)
                        else:
                            dst_img_name = "{}@{}".format(new_label, basename)
                        save_img_path = os.path.join(lot_path, new_label)
                        os.makedirs(save_img_path, exist_ok=True)
                        try:
                            shutil.copy2(img, save_img_path)
                            os.rename(os.path.join(save_img_path, basename), os.path.join(save_img_path, dst_img_name))
                        except:
                            continue
                    else:
                        save_img_path = os.path.join(args.img_path, args.mode, "pending_review", "Unknow", recipe, lot, old_label)
                        os.makedirs(save_img_path, exist_ok=True)
                        try:
                            shutil.copy2(img, save_img_path)
                        except:
                            continue


url = "http://10.0.2.101:3081/ADC/"
if __name__ == '__main__':
    args = parse_args()
    print("Start Reuse ADC classification data, PlEASE CHECK, model: {}, is_all_recipes: {}".format(args.mode, args.is_all_recipe))
    warning_message(args)
    recipe_dict = get_recipe_lot(args)
    infer_by_server(args, recipe_dict)


