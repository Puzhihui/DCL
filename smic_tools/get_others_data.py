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

info('提示', '均衡数据，选择已挑选的色差和划伤文件夹，随机在源数据中挑选等量的others类别数据')
info('提示', '点击确定后选择色差文件夹')
discolor = filedialog.askdirectory() #获得选择好的文件夹
info('提示', '点击确定后选择划伤文件夹')
scratch = filedialog.askdirectory() #获得选择好的文件夹

info('提示', '点击确定后选择others源数据')
from_ohers = filedialog.askdirectory() #获得选择好的文件夹
info('提示', '点击确定后选择数据保存文件夹')
dst_ohers = filedialog.askdirectory() #获得选择好的文件夹

num = max(len(glob.glob(os.path.join(discolor, "*.bmp"))), len(glob.glob(os.path.join(scratch, "*.bmp"))))
if len(glob.glob(os.path.join(discolor, "*.bmp"))) > len(glob.glob(os.path.join(scratch, "*.bmp"))):
    index = discolor
else:
    index = scratch
dark_num = 0
for img in glob.glob(os.path.join(index, ".bmp")):
    basename = os.path.basename(img)
    camera = basename.split("_")[0]
    if "Dark" in camera:
        dark_num += 1
bright_num = len(glob.glob(os.path.join(index, ".bmp"))) - dark_num

from_others_img_list = glob.glob(os.path.join(from_ohers, "*.bmp"))
random.shuffle(from_others_img_list)

need_bright_num, need_dark_num = int(bright_num*1.5), int(dark_num*1.5)
copy_bright_num, copy_dark_num = 0, 0
for img in from_others_img_list:
    basename = os.path.basename(img)
    camera = basename.split("_")[0]
    if "Dark" in camera and copy_dark_num < need_dark_num:
        shutil.copy2(img, dst_ohers)
        copy_dark_num += 1
        os.remove(img)
    elif copy_bright_num < need_bright_num:
        shutil.copy2(img, dst_ohers)
        copy_bright_num += 1
        os.remove(img)
