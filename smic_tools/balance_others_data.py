# 根据划伤和色差的图片数，随机筛选出others类别的图片
import glob
import os
import random
import shutil
import tkinter as tk
from tkinter import filedialog
import tkinter
import tkinter.messagebox

root = tk.Tk()
root.withdraw()

def info(info_title, info_messg):
    result = tkinter.messagebox.showinfo(title = info_title, message=info_messg)
    # 返回值为：ok

info('提示', '自动删除others多余的数据，保留的数量取色差和划伤中的最大值，确保色差和划伤数据已整理')
info('提示', r'点击确定后选择日期文件夹，在D:\Solution\datas\review_data')
date_path = filedialog.askdirectory()
recipe_list = os.listdir(date_path)
for recipe in recipe_list:
    recipe_path = os.path.join(date_path, recipe)
    camera_list = os.listdir(recipe_path)
    for camera in camera_list:
        discolor = os.path.join(recipe_path, camera, "discolor")
        scratch = os.path.join(recipe_path, camera, "scratch")
        others = os.path.join(recipe_path, camera, "others")

        num = max(len(glob.glob(os.path.join(discolor, "*.bmp"))), len(glob.glob(os.path.join(scratch, "*.bmp"))))
        if len(glob.glob(os.path.join(discolor, "*.bmp"))) > len(glob.glob(os.path.join(scratch, "*.bmp"))):
            index = discolor
        else:
            index = scratch

        index_num = len(glob.glob(os.path.join(index, ".bmp")))

        others_img_list = glob.glob(os.path.join(others, ".bmp"))
        need_num = int(index_num * 1.5)
        for img in others_img_list[need_num:]:
            os.remove(img)
