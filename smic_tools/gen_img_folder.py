import os, shutil, glob
import datetime
import tkinter as tk
from tkinter import filedialog
import tkinter
import tkinter.messagebox

root = tk.Tk()
root.withdraw()


def info(info_title, info_messg):
    result = tkinter.messagebox.showinfo(title=info_title, message=info_messg)
    # 返回值为：ok


info('提示', '自动按照所需目录层级生成文件夹，请保持文件夹内的图片数据为同一类别，点击确定后选择图片所在文件夹')
from_path = filedialog.askdirectory()
category = input("请输入图片类别（scratch或discolor或other）,然后按回车键:")
while(True):
    if category not in ["scratch", "discolor", "other"]:
        category = input("类别输入不正确，请输入scratch、discolor、other其中一种：")
    else:
        break
now = datetime.datetime.now()
now_str = '{}{:02d}{:02d}'.format(now.year, now.month, now.day)
save_path = os.path.join(r"D:\Solution\datas\smic_data", now_str)

img_list = glob.glob(os.path.join(from_path, "*.bmp"))
copy_num = 0
for img in img_list:
    basename = os.path.basename(img)
    camera = basename.split("_")[0]
    recipe = basename.split("_")[1]
    save_img_path = os.path.join(save_path, recipe, camera, category)
    os.makedirs(save_img_path, exist_ok=True)
    if os.path.exists(os.path.join(save_img_path, basename)):
        print("{}已存在，不再进行复制".format(img))
    else:
        shutil.copy2(img, save_img_path)
        copy_num += 1
print("数据已复制到{},共复制{}张图，类别为{}".format(save_path, copy_num, category))
print("Complete！")

