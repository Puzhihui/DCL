import os, glob, shutil
import random

train_path = r'D:\Solution\datas\smic_om_front_by_recipe\train'
val_path = r'D:\Solution\datas\smic_om_front_by_recipe\val'

for recipe in os.listdir(train_path):
    # img_list = glob.glob(os.path.join(train_path, recipe, 'Front', '*', '*.bmp'))
    categories = os.listdir(os.path.join(train_path, recipe, 'Front'))
    for category in categories:
        img_list = glob.glob(os.path.join(train_path, recipe, 'Front', category, '*.bmp'))
        if len(img_list) >= 3 and len(img_list) < 10:
            val = 1
        else:
            val = int(len(img_list) * 0.1)

        random.shuffle(img_list)
        for img in img_list[:val]:
            save_img_path = os.path.join(val_path, recipe, 'Front', category)
            os.makedirs(save_img_path, exist_ok=True)
            shutil.copy2(img, save_img_path)
            # os.remove(img)
