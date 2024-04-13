import os

dataset_path = r'D:\Solution\datas\Front_M6'
category_org = "false"
category_dst = "FALSE"

for recipe in os.listdir(dataset_path):
    category_org_path = os.path.join(dataset_path, recipe, category_org)
    if not os.path.exists(category_org_path):
        continue
    os.rename(category_org_path, os.path.join(dataset_path, recipe, category_dst))
