import os
import shutil
import glob
import argparse
import random
import datetime


def parse_args():
    parser = argparse.ArgumentParser(description='get report')
    parser.add_argument('--mode', default='Back', type=str)
    parser.add_argument('--img_path', default=r'D:\Solution\datas\get_report', type=str)
    # parser.add_argument('--is_all_recipe', action='store_true')
    args = parser.parse_args()
    return args


def get_added_lot():
    added_lot = dict()
    if os.path.exists(added_txt):
        f = open(added_txt, "r", encoding='utf-8')
        lines = f.readlines()
        f.close()
        for line in lines:
            line = line.rstrip("\n")
            recipe, lot = line.split(',')[0], line.split(',')[1]
            if recipe not in added_lot.keys():
                added_lot[recipe] = set()
            added_lot[recipe].add(lot)
    return added_lot


def get_img_list(lot_path):
    img_list = glob.glob(os.path.join(lot_path, "*", "*.bmp"))
    defect_img_list = []
    overkill_img_list = []
    good_img_list = []
    for img in img_list:
        basename = os.path.basename(img)
        op_label = img.split('\\')[-2]
        try:
            adc_label = basename.split('@')[0]
        except:
            continue

        if op_label == false_string:
            if adc_label != false_string:
                overkill_img_list.append([img, false_string])
            else:
                good_img_list.append([img, false_string])
        elif op_label in [scratch_string, discolor_string, other_string, pasd_string, sinr_string, cscratch_string]:
            defect_img_list.append([img, op_label])
        else:
            continue
    random.shuffle(overkill_img_list)
    random.shuffle(good_img_list)
    return defect_img_list, overkill_img_list, good_img_list


def move_img_list(save_path, recipe, defect_img_list, overkill_imgs, false_img_list):
    defect_num = len(defect_img_list)
    if defect_num == 0:
        return False

    for defect_img in defect_img_list:
        img_path, op_label = defect_img[0], defect_img[1]
        save_img_path = os.path.join(save_path, recipe, op_label)
        os.makedirs(save_img_path, exist_ok=True)
        shutil.copy2(img_path, save_img_path)
        # os.remove(img_path)

    for overkill in overkill_imgs:
        img_path, op_label = overkill[0], overkill[1]
        save_img_path = os.path.join(save_path, recipe, op_label)
        os.makedirs(save_img_path, exist_ok=True)
        shutil.copy2(img_path, save_img_path)
        defect_num -= 1

    if defect_num > 0:
        for img in false_img_list[:defect_num]:
            img_path, op_label = img[0], img[1]
            save_img_path = os.path.join(save_path, today, "Front_{}".format(today), recipe, op_label)
            os.makedirs(save_img_path, exist_ok=True)
            shutil.copy2(img_path, save_img_path)
            # os.remove(img_path)
    return True


def get_train_imgs(reviewed_path, save_path, added_lot_dict):
    added_this_time = dict()
    for recipe in os.listdir(reviewed_path):
        recipe_path = os.path.join(reviewed_path, recipe)
        if not os.path.isdir(recipe_path):
            continue
        added_lot_list = []
        if recipe in added_lot_dict.keys():
            added_lot_list = added_lot_dict[recipe]
        for lot in os.listdir(recipe_path):
            lot_path = os.path.join(recipe_path, lot)
            if not os.path.isdir(lot_path) or lot in added_lot_list:
                continue
            bright_defect_img_list, bright_overkill_imgs, bright_false_img_list = get_img_list(lot_path)
            flag = move_img_list(save_path, recipe, bright_defect_img_list, bright_overkill_imgs, bright_false_img_list)
            if flag:
                if recipe not in added_this_time.keys():
                    added_this_time[recipe] = set()
                added_this_time[recipe].add(lot)
    return added_this_time


def write_txt(added_this_time):
    with open(added_txt, "a", encoding='utf-8') as file:
        for recipe, lot_set in added_this_time.items():
            for lot in list(lot_set):
                file.write("{},{}\n".format(recipe, lot))


def warning_message(save_path, added_this_time):
    if len(added_this_time.keys()) > 0:
        print("请调整{}中的数据, 以便于训练".format(save_path))
        print("若需要增加其他数据，请在此阶段加入")
        user_input = "None"
        while(user_input.lower() != 'y'):
            user_input = input("调整或加入完成后请输入y后回车:")
    else:
        print("本次从reviewed文件夹中未获取到数据，直接开始训练")


def copy_remove_imgs(save_train_path, recipe, category, img_list):
    for img in img_list:
        save_img_path = os.path.join(save_train_path, recipe, category)
        os.makedirs(save_img_path, exist_ok=True)
        try:
            shutil.copy2(img, save_img_path)
            os.remove(img)
        except:
            print("copy ERROR: {}".format(img))
            continue

def move_split_imgs(from_path, dataset_path, val_ratio=0.1):
    global val_num
    for recipe in os.listdir(from_path):
        recipe_path = os.path.join(from_path, recipe)
        if not os.path.isdir(recipe_path):
            continue
        for lot in os.listdir(recipe_path):
            lot_path = os.path.join(recipe_path, lot)
            if not os.path.isdir(lot_path):
                continue
            for category in os.listdir(lot_path):
                category_path = os.path.join(lot_path, category)
                if not os.path.isdir(category_path):
                    continue
                img_list = glob.glob(os.path.join(category_path, "*.bmp"))
                random.shuffle(img_list)
                if len(img_list) < 3:
                    val_num = 0
                elif len(img_list) <= 10 and len(img_list) >= 3:
                    val_num = 1
                elif len(img_list) > 10:
                    val_num = int(len(img_list) * val_ratio)
                copy_remove_imgs(os.path.join(dataset_path, "val"), recipe, category, img_list[:val_num])
                copy_remove_imgs(os.path.join(dataset_path, "train"), recipe, category, img_list[val_num:])


false_string = 'false'
scratch_string = 'scratch'
cscratch_string = 'cScratch'
discolor_string = 'discolor'
other_string = 'other'
pasd_string = 'PASD'
sinr_string = 'SINR'

added_txt = "added_lot.txt"
if __name__ == '__main__':
    args = parse_args()
    added_lot_dict = get_added_lot()
    today = datetime.datetime.now().strftime('%Y%m%d')
    save_path = os.path.join(args.img_path, args.mode, "train_data", today)
    added_this_time = get_train_imgs(os.path.join(args.img_path, args.mode, "reviewed"),
                                     save_path,
                                     added_lot_dict)
    write_txt(added_this_time)
    warning_message(save_path, added_this_time)
    # 移动数据至训练集和验证集 在每个文件夹下，如果图片数<3张，则全部用于训练集，如果>=3并<=10，则随机取1张为验证集，如果>10张取10%
    if args.mode == "Front":
        dataset_path = "D:\Solution\datas\smic_om_back_5"
    elif args.mode == "Back":
        dataset_path = "D:\Solution\datas\smic_om_front_by_recipe"
    else:
        raise "不存在这个数据集:{}".format(args.mode)
    move_split_imgs(save_path, dataset_path)

