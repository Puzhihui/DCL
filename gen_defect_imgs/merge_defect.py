import glob
import math
import os
import shutil

import cv2
import random
import argparse
import datetime


def get_roi_from_background(background, mask_shape):
    height, width, channel = background.shape
    mask_h, mask_w = mask_shape
    if height < mask_h or width < mask_w:
        raise "背景图长宽小于mask图"
    x_start = random.randint(0, width - mask_w)
    y_start = random.randint(0, height - mask_h)
    roi_coord = [x_start, y_start, x_start+mask_w, y_start+mask_h]

    background_roi = background[roi_coord[1]:roi_coord[3], roi_coord[0]:roi_coord[2]]
    return background_roi, roi_coord


def gen_defect_img(image_path, mask_path, background_path, output_path):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    background = cv2.imread(background_path)

    background_roi, roi_coord = get_roi_from_background(background, mask_shape=mask.shape)
    print("ROI 坐标", roi_coord)
    background_roi[mask > 0] = image[mask > 0]
    background[roi_coord[1]:roi_coord[3], roi_coord[0]:roi_coord[2]] = background_roi
    cv2.imwrite(output_path, background)


def category_bright_dark(false_img_list):
    bright_imgs = []
    dark_imgs = []
    for img in false_img_list:
        mode = os.path.basename(img).split("_")[0]
        if mode == "Front":
            bright_imgs.append(img)
        elif mode == "FrontDark":
            dark_imgs.append(img)
        else:
            raise "请重命名该图片，格式为 Front_文件名，若为暗场则 FrontDark_文件名：{}".format(img)
    random.shuffle(bright_imgs)
    random.shuffle(dark_imgs)
    return bright_imgs, dark_imgs


def expansion_imgs(img_list, gen_num):
    if len(img_list) >= gen_num or len(img_list) == 0:
        return img_list
    else:
        need_num = gen_num - len(img_list)
        need_img_list = []
        while len(need_img_list) < need_num:
            for img in img_list:
                need_img_list.append(img)
                if len(need_img_list) >= need_num:
                    break
        img_list.extend(need_img_list)
        return img_list


def gen_defect_by_mode(false_imgs, defect_path, category_dict, save_front_path):
    for label, gen_num in category_dict.items():
        if gen_num == 0:
            continue
        mask_img_list = glob.glob(os.path.join(defect_path, label, "*_mask.png"))
        if len(mask_img_list) == 0 and gen_num > 0:
            raise "{}路径未找到mask图".format(os.path.join(defect_path, label))
        random.shuffle(mask_img_list)
        mask_img_list = expansion_imgs(mask_img_list, gen_num)
        false_imgs = expansion_imgs(false_imgs, gen_num)
        for i, mask_path in enumerate(mask_img_list[:gen_num]):
            defect_img_path = mask_path.replace("_mask.png", ".bmp")
            background_path = false_imgs[i]
            if not os.path.exists(defect_img_path):
                raise "不存在该mask{}对应的原缺陷图{}".format(mask_path, defect_img_path)
            synthetic_defect_basename = '{}_{}_{}{}.bmp'.format(os.path.splitext(os.path.basename(background_path))[0],
                                                                label, random.randint(0, 100), i)
            save_img_path = os.path.join(save_front_path, label)
            os.makedirs(save_img_path, exist_ok=True)
            gen_defect_img(defect_img_path, mask_path, background_path, os.path.join(save_img_path, synthetic_defect_basename))


def copy_imgs(save_path, img_list, remove_org=False):
    os.makedirs(save_path, exist_ok=True)
    success_num = 0
    for img in img_list:
        try:
            shutil.copy2(img , save_path)
            success_num += 1
            if remove_org:
                os.remove(img)
        except:
            continue
    return success_num

def get_mode_false_imgs(recipe_path, need_num, mode):
    false_imgs = []
    for lot in os.listdir(recipe_path):
        lot_path = os.path.join(recipe_path, lot)
        if not os.path.isdir(lot_path):
            continue
        for wafer in os.listdir(lot_path):
            wafer_path = os.path.join(lot_path, wafer)
            if not os.path.isdir(wafer_path):
                continue
            img_list = glob.glob(os.path.join(wafer_path, "Front", mode, "ADC", "*.bmp"))
            random.shuffle(img_list)
            false_imgs.extend(img_list[: max(0, need_num - len(false_imgs))])
            if len(false_imgs) >= need_num:
                return false_imgs

    return false_imgs


def move_to_dataset(recipe, from_path, dataset_path, bright_category_dict, dark_category_dict):
    recipe_generate = "_{}generate".format(recipe)
    for label in os.listdir(from_path):
        label_path = os.path.join(from_path, label)
        if not os.path.isdir(label_path) or label not in bright_category_dict.keys() or label not in dark_category_dict.keys():
            continue
        img_list = glob.glob(os.path.join(label_path, "*.bmp"))
        img_list_num = len(img_list)

        if img_list_num <= 6 and img_list_num > 0:
            val_num = 1
        elif img_list_num <= 10 and img_list_num > 6:
            val_num = 2
        elif img_list_num > 10:
            val_num = math.ceil(img_list_num * 0.2)
        else:
            continue
        copy_val_num = copy_imgs(os.path.join(dataset_path, "val", recipe_generate, label), img_list[:val_num], remove_org=True)
        copy_train_num = copy_imgs(os.path.join(dataset_path, "train", recipe_generate, label), img_list[val_num:], remove_org=True)
        print("{} {}: 训练集添加{}张, 验证集添加{}张".format(recipe_generate, label, copy_train_num, copy_val_num))

def main(args, save_path, defect_path, bright_category_dict, dark_category_dict):
    imagedata = args.imagedata
    bright_num = sum(bright_category_dict.values())
    dark_num = sum(dark_category_dict.values())
    print("合成缺陷背景图会在{}文件夹中进行捞取，RecipeName请按照该文件夹中的命名输入,例如00EX_VSI_OM 04NQ_VSI_OM-Test".format(imagedata))
    user_input = input("请输入RecipeName,多个RecipeName用空格分开:")
    recipe_list = user_input.split(" ")
    for recipe in recipe_list:
        print("start:==========={}===========".format(recipe))
        recipe_path = os.path.join(imagedata, recipe)
        if not os.path.exists(recipe_path):
            print("{}：不存在该recipe".format(recipe))

        bright_img_list = get_mode_false_imgs(recipe_path, int(bright_num*1.2), mode="FrontCamera1_BrightField*")
        save_front_path = os.path.join(save_path, recipe, "Front")
        success_num = copy_imgs(os.path.join(save_front_path, "false"), bright_img_list)
        print("明场获取到{}张背景图, 成功复制{}张".format(len(bright_img_list), success_num))

        dark_img_list = get_mode_false_imgs(recipe_path, int(dark_num*1.2), mode="FrontCamera3_DarkField*")
        save_frontdark_path = os.path.join(save_path, recipe, "FrontDark")
        success_num = copy_imgs(os.path.join(save_frontdark_path, "false"), dark_img_list)
        print("暗场获取到{}张背景图, 成功复制{}张".format(len(dark_img_list), success_num))

        print("Warning: 请打开以下两个文件夹，删除缺陷数据")
        print("1.{}".format(os.path.join(save_front_path, "false")))
        print("2.{}".format(os.path.join(save_frontdark_path, "false")))
        user_input1 = input("完成后输入任意字符回车:")

        # 生成缺陷图
        bright_imgs = glob.glob(os.path.join(save_front_path, "false", "*.bmp"))
        dark_imgs = glob.glob(os.path.join(save_frontdark_path, "false", "*.bmp"))
        gen_defect_by_mode(bright_imgs, os.path.join(defect_path, "Front"), bright_category_dict, save_front_path)
        gen_defect_by_mode(dark_imgs, os.path.join(defect_path, "FrontDark"), dark_category_dict, save_frontdark_path)
        user_input1 = input("缺陷数据已生成,请检查数据，完成后输入任意字符回车:")

        # 转移到训练集
        dataset_path = args.dataset_path
        move_to_dataset(recipe, save_front_path, dataset_path, bright_category_dict, dark_category_dict)


def parse_args():
    parser = argparse.ArgumentParser(description='gen defect imgs')
    parser.add_argument('--save_path', default=r'D:\Solution\datas\mix_data\mix', type=str)
    parser.add_argument('--defect_path', default=r'D:\Solution\datas\mix_data\defect', type=str)
    parser.add_argument('--imagedata', default=r'F:\ImageData', type=str)
    parser.add_argument('--dataset_path', default=r'D:\Solution\datas\smic_om_front_by_recipe', type=str)

    parser.add_argument('--scratch_bright', default=0, type=int)
    parser.add_argument('--discolor_bright', default=0, type=int)
    parser.add_argument('--other_bright', default=47, type=int)
    parser.add_argument('--SINR_bright', default=0, type=int)
    parser.add_argument('--PASD_bright', default=0, type=int)

    parser.add_argument('--scratch_dark', default=2, type=int)
    parser.add_argument('--other_dark', default=32, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    bright_category_dict = {
        "scratch": args.scratch_bright, "discolor": args.discolor_bright, "other": args.other_bright,
        "SINR": args.SINR_bright, "PASD": args.PASD_bright}
    dark_category_dict = {"scratch": args.scratch_dark, "other": args.other_dark}
    save_path = os.path.join(args.save_path, datetime.now().date().strftime("%Y%m%d"))
    os.makedirs(save_path)
    defect_path = args.defect_path
    main(args, save_path, defect_path, bright_category_dict, dark_category_dict)
