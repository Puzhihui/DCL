import glob
import os
import shutil

path = "D:\Solution\datas\smic_om_3"
dst_path = "D:\Solution\datas\smic_om_3_rename"
if os.path.exists(dst_path):
    shutil.rmtree(dst_path)


def copy_img(from_path, save_path):
    img_list = glob.glob(os.path.join(from_path, "*.bmp"))
    i = 0
    for img in img_list:
        basename = os.path.basename(img)
        camera = basename.split('_')[0]
        os.makedirs(save_path, exist_ok=True)
        shutil.copy2(img, save_path)
        os.replace(os.path.join(save_path, basename), os.path.join(save_path, '{}_{}.bmp'.format(camera, str(i))))
        i += 1

def rename_and_copy(from_path, save_path, trainORval):
    discolor = os.path.join(from_path, "discolor")
    copy_img(discolor, os.path.join(save_path, trainORval, "discolor"))
    other = os.path.join(from_path, "other")
    copy_img(other, os.path.join(save_path, trainORval, "other"))
    scratch = os.path.join(from_path, "scratch")
    copy_img(scratch, os.path.join(save_path, trainORval, "scratch"))

train = os.path.join(path, "train")
rename_and_copy(train, dst_path, "train")
val = os.path.join(path, "val")
rename_and_copy(val, dst_path, "val")
