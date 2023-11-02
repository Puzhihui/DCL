import os
import random
import shutil
import glob
import argparse
import datetime


def parse_args():
    parser = argparse.ArgumentParser(description='replace online model')
    parser.add_argument('--mode', default='Front', type=str)
    args = parser.parse_args()
    return args


def get_img_list(mode_path):
    img_list = glob.glob(os.path.join(mode_path, "*", "*.bmp"))
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
        elif op_label in [scratch_string, discolor_string, other_string, pasd_string, sinr_string]:
            defect_img_list.append([img, op_label])
        else:
            continue
    random.shuffle(overkill_img_list)
    random.shuffle(good_img_list)
    return defect_img_list, overkill_img_list, good_img_list


def move_img_list(recipe, defect_img_list, overkill_imgs, false_img_list):
    defect_num = len(defect_img_list)
    # if defect_num == 0:
    #     return first_false_img_list, second_false_img_list

    today = datetime.datetime.now().strftime('%Y%m%d')
    for defect_img in defect_img_list:
        img_path, op_label = defect_img[0], defect_img[1]
        save_img_path = os.path.join(save_path, today, "Front_{}".format(today), recipe, op_label)
        os.makedirs(save_img_path, exist_ok=True)
        shutil.copy2(img_path, save_img_path)
        # os.remove(img_path)

    for overkill in overkill_imgs:
        img_path, op_label = overkill[0], overkill[1]
        save_img_path = os.path.join(save_path, today, "Front_{}".format(today), recipe, op_label)
        os.makedirs(save_img_path, exist_ok=True)
        shutil.copy2(img_path, save_img_path)
        defect_num -= 1

    if defect_num > 0:
        for img in false_img_list[:defect_num]:
            img_path, op_label = img[0], img[1]
            save_img_path = os.path.join(save_path, today, "Front_{}".format(today), recipe, op_label)
            os.makedirs(save_img_path, exist_ok=True)
            shutil.copy2(img_path, save_img_path)
            os.remove(img_path)
    return True


args = parse_args()
mode = args.mode
save_root = r"D:\Solution\datas\smic_data"
report_root = r'D:\Solution\datas\get_report'
if mode == "Back":
    report_data = os.path.join(report_root, "Back")
    save_path = os.path.join(save_root, "Back")
elif mode == "Front":
    report_data = os.path.join(report_root, "Front")
    save_path = os.path.join(save_root, "Front")
else:
    raise "Mode error!!!"

false_string = 'false'
scratch_string = 'scratch'
discolor_string = 'discolor'
other_string = 'other'
pasd_string = 'PASD'
sinr_string = 'SINR'


def main():
    for date_dir in os.listdir(report_data):
        date_path = os.path.join(report_data, date_dir)
        if not os.path.isdir(date_path) or date_dir == 'underkill':
            continue
        try:
            date_int = int(date_dir)
        except:
            continue

        for recipe in os.listdir(date_path):
            recipe_path = os.path.join(date_path, recipe)
            if not os.path.isdir(recipe_path):
                continue

            for lot in os.listdir(recipe_path):
                lot_path = os.path.join(recipe_path, lot)
                if not os.path.isdir(lot_path):
                    continue
                print(date_int, recipe, lot)
                # frontDark = os.path.join(lot_path, "FrontDark")
                bright_defect_img_list, bright_overkill_imgs, bright_false_img_list = get_img_list(lot_path)
                # dark_defect_img_list, dark_overkill_imgs, dark_false_img_list = get_img_list(frontDark)
                move_img_list(recipe, bright_defect_img_list, bright_overkill_imgs, bright_false_img_list)
                # move_img_list("FrontDark", dark_defect_img_list, dark_overkill_imgs, dark_false_img_list)


if __name__ == '__main__':
    main()
