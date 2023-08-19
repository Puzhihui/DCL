import os, shutil, glob
import datetime

front_img_path = r"D:\Solution\datas\front_full_img"
txt = os.path.join(front_img_path, "front.txt")

ImageData = r"F:\ImageData"

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

recipe_list = os.listdir(ImageData)
for recipe in recipe_list:
    if recipe not in recipe_dict.keys():
        continue
    recipe_path = os.path.join(ImageData, recipe)
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
        full_img_list = glob.glob(os.path.join(lot_path, "*", "*", "FrontCamera1_BrightField1", "*.jpg"))
        img_list = glob.glob(os.path.join(lot_path, "*", "*", "FrontCamera1_BrightField1", "ADC","*.bmp"))
        save_full_img_path = os.path.join(front_img_path, "{}_{}_{}_{}".format(datetime.datetime.now().year, datetime.datetime.now().month, datetime.datetime.now().day, datetime.datetime.now().hour))
        os.makedirs(save_full_img_path, exist_ok=True)
        for img in full_img_list:
            wafer_name = img.split("\\")[-4]
            try:
                shutil.copy2(img, save_full_img_path)
                basename = os.path.basename(img)
                os.rename(os.path.join(save_full_img_path, basename), os.path.join(save_full_img_path, "{}@{}@{}".format(recipe, wafer_name, basename)))
            except:
                continue
        for img in img_list:
            wafer_name = img.split("\\")[-5]
            save_adc_img_path = os.path.join(save_full_img_path, recipe, wafer_name)
            os.makedirs(save_adc_img_path, exist_ok=True)
            try:
                shutil.copy2(img, save_adc_img_path)
            except:
                continue
                
