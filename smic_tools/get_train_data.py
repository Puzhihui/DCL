import os
import shutil
import glob

support_categories = ['discolor', 'other', 'scratch']
support_images = ['.bmp', '.BMP']


def split_and_move(category, img_list, save_path, ratio_val):
    val_num = int(len(img_list) * ratio_val)
    for img in img_list[:val_num]:
        save_img_path = os.path.join(save_path, 'val', category)
        os.makedirs(save_img_path, exist_ok=True)
        if os.path.exists(os.path.join(save_img_path, os.path.basename(img))):
            os.remove(os.path.join(save_img_path, os.path.basename(img)))
        shutil.move(img, save_img_path)

    for img in img_list[val_num:]:
        save_img_path = os.path.join(save_path, 'train', category)
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
    val_ratio = 0.1
    save_dir = r'D:\Solution\datas\smic_om_3'
    os.makedirs(save_dir, exist_ok=True)

    from_front_data = r'D:\Solution\datas\smic_data\front'
    from_back_data = r'D:\Solution\datas\smic_data\back'

    print('1.start:开始移动数据....')
    train_images = get_train_data(from_front_data, save_dir, val_ratio)
    print('移动正检数据, discolor:{}张, other:{}张, scratch:{}张'.format(train_images['discolor'], train_images['other'], train_images['scratch']))
    train_images = get_train_data(from_back_data, save_dir, val_ratio)
    print('移动背检数据, discolor:{}张, other:{}张, scratch:{}张'.format(train_images['discolor'], train_images['other'], train_images['scratch']))
    print('1.end:数据移动完成！数据已移动至{}'.format(save_dir))
    print('-------------------------------------------------')
