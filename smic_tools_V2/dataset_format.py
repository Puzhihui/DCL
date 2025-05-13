import glob
import os
import shutil

from tqdm import tqdm

client = 'M6'

def rename_recipeName(data_path, extension='_VSI_OM'):
    print('正在添加_VSI_OM', data_path)
    for recipe in tqdm(os.listdir(data_path)):
        recipe_path = os.path.join(data_path, recipe)
        if not os.path.isdir(recipe_path):
            continue
        if extension not in recipe:
            os.rename(recipe_path, os.path.join(data_path, '{}{}'.format(recipe, extension)))


def generate(data_path, extension='_generate'):
    print('正在合并generate', data_path)
    for recipe in tqdm(os.listdir(data_path)):
        recipe_path = os.path.join(data_path, recipe)
        if not os.path.isdir(recipe_path):
            continue

        if extension in recipe:
            recipe_short = recipe.replace(extension, "")
            for label in os.listdir(recipe_path):
                img_list = glob.glob(os.path.join(recipe_path, label, "*"))
                for img in img_list:
                    save_img_path = os.path.join(data_path, recipe_short, label)
                    os.makedirs(save_img_path, exist_ok=True)
                    shutil.copy2(img, save_img_path)
                    os.remove(img)
            # shutil.rmtree(recipe_path)


def main():
    front_path = 'D:\Solution\datas\Front_{}'.format(client)
    front_path_val = 'D:\Solution\datas\Front_{}_val'.format(client)
    frontDark_path = 'D:\Solution\datas\FrontDark_{}'.format(client)
    frontDark_path_val = 'D:\Solution\datas\FrontDark_{}_val'.format(client)
    # back_path = 'D:\Solution\datas\Back_{}'.format(client)
    # back_path_val = 'D:\Solution\datas\Back_{}_val'.format(client)

    generate(front_path)
    generate(front_path_val)
    generate(frontDark_path)
    generate(frontDark_path_val)

    rename_recipeName(front_path)
    rename_recipeName(front_path_val)
    rename_recipeName(frontDark_path)
    rename_recipeName(frontDark_path_val)

if __name__ == '__main__':
    main()

