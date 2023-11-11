import cv2
import shutil
import glob
import os


def crop_img_list(img_list, save_path, shape=["h", "w"]):
    for image_path in img_list:
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        left = (width - shape[1]) // 2
        top = (height - shape[0]) // 2
        cropped = image[top:top + shape[0], left:left + shape[1]]

        recipe, label = image_path.split("\\")[-4], image_path.split("\\")[-2]
        basename_with_label = os.path.basename(image_path)
        if "@" in basename_with_label:
            basename = basename_with_label.replace("{}@".format(basename_with_label.split('@')[0]), "")
        else:
            basename = basename_with_label
        mode = basename.split('_')[0]
        if mode not in ["Front", "FrontDark"]:
            raise "mode ERROR:{}".format(image_path)
        save_img_path = os.path.join(save_path, mode, label)
        os.makedirs(save_img_path, exist_ok=True)
        img_cropped_path = os.path.join(save_img_path, basename)
        # shutil.copy2(image_path, img_cropped_path)
        cv2.imwrite(img_cropped_path, cropped)

if __name__ == '__main__':
    save_path = r'D:\Solution\datas\mix_data\defect_1'
    image_list = glob.glob(os.path.join(r'D:\Solution\datas\mix_data\defect-1', "*", "*", "*.bmp"))
    crop_img_list(image_list, save_path, shape=[250, 250])
