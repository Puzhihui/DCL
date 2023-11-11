import glob
import json
import cv2
import numpy as np
import os


def json_to_mask(json_path, image_shape):
    with open(json_path, 'r') as f:
        data = json.load(f)

    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    for shape in data['shapes']:
        label = shape['label']
        points = shape['points']
        polygon = np.array(points, np.int32)
        cv2.fillPoly(mask, [polygon], 255)

    return mask

if __name__ == '__main__':
    defect_path = r'D:\Solution\datas\mix_data\defect'
    json_list = glob.glob(os.path.join(defect_path, "*", "*", "*.json"))
    for json_path in json_list:
        label = json_path.split('\\')[-2]
        img_path = json_path.replace(".json", ".bmp")
        if not os.path.exists(img_path):
            raise "json{}对应的图片{}不存在:".format(json_path, img_path)
        image = cv2.imread(img_path)
        h, w, c = image.shape
        image_shape = (h, w)
        mask = json_to_mask(json_path, image_shape)
        save_mask_path = img_path.replace(".bmp", "_mask.png")
        cv2.imwrite(save_mask_path, mask)
