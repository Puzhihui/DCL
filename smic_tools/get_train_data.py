import sys
sys.path.insert(0, '../')
import os
import shutil
import glob
import argparse
from config import smic_back_online, smic_front_online

def parse_args():
    parser = argparse.ArgumentParser(description='replace online model')
    parser.add_argument('--mode', default='Back', type=str)
    args = parser.parse_args()
    return args

args = parse_args()
mode = args.mode
if mode == "Back":
    cfg_mode = smic_back_online()
elif mode == "Front":
    cfg_mode = smic_front_online()
else:
    raise "Mode error!!!"

# support_categories = ['discolor', 'other', 'scratch']
support_images = ['.bmp', '.BMP']


def split_and_move(recipe, camera, category, img_list, save_path, ratio_val):
    val_num = int(len(img_list) * ratio_val)
    for img in img_list[:val_num]:
        save_img_path = os.path.join(save_path, 'val', recipe, camera, category)
        os.makedirs(save_img_path, exist_ok=True)
        if os.path.exists(os.path.join(save_img_path, os.path.basename(img))):
            os.remove(os.path.join(save_img_path, os.path.basename(img)))
        shutil.move(img, save_img_path)

    for img in img_list[val_num:]:
        save_img_path = os.path.join(save_path, 'train', recipe, camera, category)
        os.makedirs(save_img_path, exist_ok=True)
        if os.path.exists(os.path.join(save_img_path, os.path.basename(img))):
            os.remove(os.path.join(save_img_path, os.path.basename(img)))
        shutil.move(img, save_img_path)


def get_train_data(data_path, save_path, ratio_val):
    categories = os.listdir(data_path)
    train_images = dict()

    for category in categories:
        img_list = glob.glob(os.path.join(data_path, category, '*'))

        new_img_list = []
        for img in img_list:
            if os.path.splitext(img)[-1] not in support_images:
                continue
            new_img_list.append(img)
        split_and_move(category, new_img_list, save_path, ratio_val)
        train_images[category] = len(new_img_list)

    return train_images


if __name__ == "__main__":
    print('-------------------------------------------------')
    print('1.start:开始移动数据')
    val_ratio = 0.1
    save_dir = cfg_mode.train_data_path
    os.makedirs(save_dir, exist_ok=True)
    smic_data_root = r"D:\Solution\datas\smic_data"
    if mode == "Back":
        smic_data = os.path.join(smic_data_root, "Back")
    elif mode == "Front":
        smic_data = os.path.join(smic_data_root, "Front")
    date_list = os.listdir(smic_data)
    all_move_num = dict()
    all_move_num['discolor'], all_move_num['other'], all_move_num['scratch'], all_move_num['false'] = 0, 0, 0, 0
    for per_date in date_list:
        date_path = os.path.join(smic_data, per_date)
        if not os.path.isdir(date_path): continue
        recipe_list = os.listdir(date_path)
        for recipe in recipe_list:
            move_num = dict()
            move_num['discolor'], move_num['other'], move_num['scratch'], move_num['false'] = 0, 0, 0, 0
            recipe_path = os.path.join(date_path, recipe)
            if not os.path.isdir(recipe_path): continue
            camera_list = os.listdir(recipe_path)
            for camera in camera_list:
                camera_path = os.path.join(recipe_path, camera)
                if not os.path.isdir(camera_path): continue
                categories = os.listdir(camera_path)
                for category in categories:
                    category_path = os.path.join(camera_path, category)
                    if not os.path.isdir(category_path): continue
                    img_list = glob.glob(os.path.join(category_path, "*.bmp"))
                    split_and_move(recipe, camera, category, img_list, save_dir, val_ratio)
                    move_num[category] += len(img_list)
                    if category not in all_move_num:
                        all_move_num[category] = 0
                    all_move_num[category] += len(img_list)
            print('{}: discolor:{}张, other:{}张, scratch:{}张, false:{}张'.format(recipe, move_num['discolor'], move_num['other'], move_num['scratch'], move_num['false']))
    print('合计: discolor:{}张, other:{}张, scratch:{}张, false:{}张'.format(all_move_num['discolor'], all_move_num['other'], all_move_num['scratch'], all_move_num['false']))
    print('-------------------------------------------------')
