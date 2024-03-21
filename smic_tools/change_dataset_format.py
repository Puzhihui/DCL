import os
import glob
import shutil

path = r''
save_path = r''
save_dark_path = r''
save_front_other = r''
category2bincode = {'discolor': 'PADC', 'other': "PAOH", 'scratch': "PASC"}
# category2bincode = {'discolor': "BSDC", 'other': "BSOH", "cScratch": "BSCS"}
img_list = glob.glob(os.path.join(path, "*", "*", "*", "*.bmp"))
for img in img_list:
    recipe, field, label = img.split("\\")[-4], img.split("\\")[-3], img.split("\\")[-2]
    bincode = category2bincode[label] if label in category2bincode.keys() else label
    if label == 'other':
        shutil.copy2(img, save_front_other)
        basename = os.path.basename(img)
        os.rename(os.path.join(save_front_other, basename), os.path.join(save_front_other, "{}&{}&{}".format(recipe, field, basename)))
        continue

    save_img = os.path.join(save_path, recipe, bincode)
    if "Dark" in field:
        save_img = os.path.join(save_dark_path, recipe, bincode)
    os.makedirs(save_img, exist_ok=True)
    shutil.copy2(img, save_img)


img_list = glob.glob(os.path.join(save_front_other, "*", "*.bmp"))
for img in img_list:
    basename = os.path.basename(img)
    label = img.split("\\")[-2]
    recipe, field, save_basename = basename.split("&")[0], basename.split("&")[1], basename.split("&")[2]
    save_img = os.path.join(save_path, recipe, label)
    if "Dark" in field:
        save_img = os.path.join(save_dark_path, recipe, label)
    os.makedirs(save_img, exist_ok=True)
    os.rename(os.path.join(save_img, basename),
                  os.path.join(save_img, save_basename))

