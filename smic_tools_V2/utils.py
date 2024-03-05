import os
import csv


def get_recipe_lot(args, path=None):
    if path is None:
        path = os.path.join(args.img_path, args.mode, "pending_review")
    # pending_review_path = os.path.join(args.img_path, args.mode, "pending_review")
    pending_recipe_dict = dict()
    if args.is_all_recipe:
        for recipe in os.listdir(path):
            recipe_path = os.path.join(path, recipe)
            if recipe in ["underkill", "reviewed"] or not os.path.isdir(recipe_path):
                continue
            if recipe not in pending_recipe_dict.keys():
                pending_recipe_dict[recipe] = set()
            for lot in os.listdir(recipe_path):
                if os.path.isdir(os.path.join(recipe_path, lot)):
                    pending_recipe_dict[recipe].add(lot)
    else:
        report_txt = os.path.join(args.img_path, args.mode, 'report.txt')
        f = open(report_txt, 'r', encoding='utf-8')
        lines = f.readlines()
        f.close()
        for line in lines:
            line = line.replace("\n", "")
            recipe, lot = line.split('\t')[0], line.split('\t')[1]
            if recipe not in pending_recipe_dict.keys():
                pending_recipe_dict[recipe] = set()
            pending_recipe_dict[recipe].add(lot)

    return pending_recipe_dict


def write_csv(csv_path, rows, headers):
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(rows)


def get_added_lot(added_txt):
    added_lot = dict()
    if os.path.exists(added_txt):
        f = open(added_txt, "r", encoding='utf-8')
        lines = f.readlines()
        f.close()
        for line in lines:
            line = line.rstrip("\n")
            recipe, lot = line.split(',')[0], line.split(',')[1]
            if recipe not in added_lot.keys():
                added_lot[recipe] = set()
            added_lot[recipe].add(lot)
    return added_lot
