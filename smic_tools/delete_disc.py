import os
import datetime
import argparse
import sys
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description='delete_disc')
    parser.add_argument('--imagedata', default=r'F:\ImageData', type=str)
    parser.add_argument('--days_thres', default=15, type=int)
    args = parser.parse_args()
    return args


def delete_lot(imagedata, days_thres):
    end_time = datetime.datetime.now() + datetime.timedelta(days=-days_thres)
    for recipe in os.listdir(imagedata):
        recipe_path = os.path.join(imagedata, recipe)
        if not os.path.isdir(recipe_path):
            continue

        for lot in os.listdir(recipe_path):
            lot_path = os.path.join(recipe_path, lot)
            if not os.path.isdir(lot_path):
                continue

            file_creation_time = datetime.datetime.fromtimestamp(os.path.getctime(lot_path))
            if file_creation_time < end_time:
                print("deleting: {}".format(lot_path))
                try:
                    shutil.rmtree(lot_path)
                except:
                    print("delete eror: {}, continue".format(lot_path))
                    continue
            else:
                print("{}: {}".format(lot_path, file_creation_time))
                continue


if __name__ == '__main__':
    args = parse_args()
    imagedata = args.imagedata
    days_thres = args.days_thres
    if days_thres <= 7:
        print("days_thres must be greater than 7.")
        sys.exit(1)
    delete_lot(imagedata, days_thres)

