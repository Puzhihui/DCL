import glob
import os
import shutil
import csv


def write2csv(save_csv_path, write_type, headers, rows):
    with open(save_csv_path, write_type, newline='', encoding='utf-8')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(rows)


def database_by_recipe(data_path, save_path):
    img_list = glob.glob(os.path.join(data_path, "*", "*", "*", "*.bmp"))
    fined_dict = dict()
    fined_dict_categories = dict()
    for img in img_list:
        basename = os.path.basename(img)
        recipe = basename.split("_")[1]
        mode = img.split('\\')[-3]
        class_ = img.split('\\')[-2]
        save_img_path = os.path.join(save_path, recipe, mode, class_)
        os.makedirs(save_img_path, exist_ok=True)
        shutil.copy2(img, save_img_path)

        lot_wafer = basename.split("_")[4]  # Front_00EX_VSI_OM_EMK066-02_F00006Row00Col00.bmp
        lot_id = lot_wafer.split('-')[0]
        try:
            wafer_id = img.split('-')[1].split('_')[0]
        except:
            wafer_id = 'false'
        if recipe not in fined_dict.keys():
            fined_dict[recipe] = dict()
        if lot_id not in fined_dict[recipe].keys():
            fined_dict[recipe][lot_id] = set()
        fined_dict[recipe][lot_id].add(wafer_id)

        if recipe not in fined_dict_categories.keys():
            fined_dict_categories[recipe] = dict()
        if class_ not in fined_dict_categories[recipe].keys():
            fined_dict_categories[recipe][class_] = 0
        fined_dict_categories[recipe][class_] += 1

    rows = []
    for recipe, lot_info in fined_dict.items():
        for lot_id, wafer_list in lot_info.items():
            wafer_string = ', '.join(wafer_list)
            rows.append([recipe, lot_id, wafer_string])

    headers = ['recipe', 'lot', 'wafer']
    write2csv(os.path.join(save_path, "defect.csv"), 'w', headers, rows)

    rows_num = []
    for recipe, category_info in fined_dict_categories.items():
        for category, num in category_info.items():
            rows_num.append([recipe, category, num])
    headers = ['recipe', 'category', 'num']
    write2csv(os.path.join(save_path, "defect_num.csv"), 'w', headers, rows_num)
    return rows


front_path = r'D:\Solution\datas\smic_om_front'
by_recipe_path = r'D:\Solution\datas\smic_om_front_by_recipe'
front_train_save_path = os.path.join(by_recipe_path, 'train')
front_val_save_path = os.path.join(by_recipe_path, 'val')
if os.path.exists(front_train_save_path):
    shutil.rmtree(front_train_save_path)
if os.path.exists(front_val_save_path):
    shutil.rmtree(front_val_save_path)
database_by_recipe(os.path.join(front_path, "train"), front_train_save_path)
database_by_recipe(os.path.join(front_path, "val"), front_val_save_path)
