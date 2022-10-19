import os
import glob


train_data_path = r'D:\Solution\datas\smic_om_3\train'
val_data_path = r'D:\Solution\datas\smic_om_3\val'

txt_root_path = r'D:\Solution\code\DCL\datasets\smic_om_3'
os.makedirs(txt_root_path, exist_ok=True)
f_train = open(os.path.join(txt_root_path, 'train.txt'), 'w', encoding='utf-8')
f_val = open(os.path.join(txt_root_path, 'val.txt'), 'w', encoding='utf-8')


def img2txt(img_list, label, f_txt):
    copy_num = 0
    for per_img in img_list:
        img_folder = per_img.split('\\')[4:]
        dst_img_name = '\\'.join(img_folder)
        f_txt.write(dst_img_name + ', ' + label + '\n')
        copy_num += 1
    return copy_num


print('2.start:开始生成数据集txt文件')

all_train_num, all_val_num = 0, 0
train_categories = os.listdir(train_data_path)
for i, category in enumerate(train_categories):
    # train
    img_list_train = glob.glob(os.path.join(train_data_path, category, '*'))
    train_num = img2txt(img_list_train, str(i), f_train)
    all_train_num += train_num
    # val
    img_list_val = glob.glob(os.path.join(val_data_path, category, '*'))
    val_num = img2txt(img_list_val, str(i), f_val)
    all_val_num += val_num

    print('{}){} 已加入数据集，\t其中训练集{}张、验证集{}张'.format(i, category, train_num, val_num))

f_train.close()
f_val.close()
print('训练集文件为{}\n验证集文件为{}'.format(os.path.join(txt_root_path, 'train.txt'), os.path.join(txt_root_path, 'val.txt')))
print('2.end:生成数据集txt成功！训练集{}张，验证集{}张'.format(all_train_num, all_val_num))
print('-------------------------------------------------')
