import sys
sys.path.insert(0, '../')
import glob
import os
import shutil
import csv
import time
import argparse
import torch
from utils.dataset_DCL import collate_fn4train, collate_fn4val, collate_fn4test, collate_fn4backbone, dataset
from config import LoadConfig, load_data_transformers
from models.LoadModel import MainModel
from torch.autograd import Variable
from transformers import BertTokenizer
import numpy as np
from collections import defaultdict
from tqdm import tqdm

os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def write_csv(csv_path, rows, headers):
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(rows)


def test_imgs_by_model(val_anno):
    dataloader = {}
    transformers = load_data_transformers(args.resize_resolution, args.crop_resolution, args.swap_num)

    val_set = dataset(Config=Config, swap_size=args.swap_num, anno=val_anno, common_aug=transformers["None"], \
                      swap=transformers["None"], totensor=transformers["adc_val_totensor"], \
                      # val_resize_totensor = transformers["adc_val_resize_totensor"],\
                      val_resize_totensor=transformers["adc_val_oi_resize_totensor"], test=True)
    dataloader['val'] = torch.utils.data.DataLoader(val_set, batch_size=args.val_batch, shuffle=False, \
                                                    num_workers=args.val_num_workers, \
                                                    collate_fn=collate_fn4test if not Config.use_backbone else collate_fn4backbone,
                                                    drop_last=True if Config.use_backbone else False, pin_memory=True)
    setattr(dataloader['val'], 'num_cls', Config.numcls)
    num_cls = dataloader['val'].num_cls
    val_corrects1, val_corrects2 = 0, 0
    with torch.no_grad():
        for batch_cnt_val, data_val in enumerate(dataloader['val']):
            if Config.use_language:
                recipe_list = []
                for img_path in data_val[2]:
                    # recipe = img_path.split('/')[-3]
                    # recipe = recipe.split("@")[0]
                    # recipe = recipe.split("_")[0]
                    # recipe_list.append(recipe)

                    recipe = img_path.split('/')[-3]
                    recipe = recipe.split("@")[0]
                    recipe = recipe.split("_")[0]
                    customer_name = recipe[3:7]
                    product_code = recipe[7:11]
                    layer_name = recipe[11:]
                    recipe_list.append('Client: {}, Product: {}, Process: {}'.format(customer_name, product_code, layer_name))
                text_inputs = bert_tokenizer(recipe_list, padding=True, truncation=True, return_tensors='pt')
                for key in text_inputs.keys():
                    text_inputs[key] = text_inputs[key].cuda()
                input_ids = text_inputs['input_ids']
                attention_mask = text_inputs['attention_mask']
            inputs = Variable(data_val[0].cuda())
            labels = Variable(torch.from_numpy(np.array(data_val[1])).long().cuda())
            if Config.use_language:
                outputs = model(inputs, input_ids, attention_mask)
            else:
                outputs = model(inputs)

            if Config.use_dcl and Config.cls_2xmul:
                outputs_pred = outputs[0] + outputs[1][:, 0:num_cls] + outputs[1][:, num_cls:2*num_cls]
            else:
                outputs_pred = outputs[0]
            top2_val, top2_pos = torch.topk(outputs_pred, 2)
            batch_corrects1 = torch.sum((top2_pos[:, 0] == labels)).data.item()
            val_corrects1 += batch_corrects1
            batch_corrects2 = torch.sum((top2_pos[:, 1] == labels)).data.item()
            val_corrects2 += (batch_corrects2 + batch_corrects1)
    return val_corrects1, val_corrects2


def test_category_dataset(result, recipe_path, recipe_name, diff_save_path):
    bad_imgs = glob.glob(os.path.join(recipe_path, 'Bad', '*.jpg'))
    good_imgs = glob.glob(os.path.join(recipe_path, 'Good', '*.jpg'))
    all_imgs_size = len(bad_imgs) + len(good_imgs)
    # good_imgs = good_imgs[:1]
    # Bad数据推理+统计
    anno_bad = dict()
    anno_bad['img_name'] = bad_imgs
    anno_bad['label'] = [0 for _ in range(len(bad_imgs))]
    corrects1_bad, _ = test_imgs_by_model(anno_bad)
    # Good类推理+统计
    anno_good = dict()
    anno_good['img_name'] = good_imgs
    anno_good['label'] = [1 for _ in range(len(good_imgs))]
    corrects1_good, _ = test_imgs_by_model(anno_good)
    result[recipe_name] = {
        'all': len(bad_imgs) + len(good_imgs),
        'Bad': len(bad_imgs),
        'Good': len(good_imgs),
        'underkill': len(bad_imgs) - corrects1_bad,
        'overkill': len(good_imgs) - corrects1_good,
        'underkill_rate': (len(bad_imgs) - corrects1_bad) / all_imgs_size if all_imgs_size != 0 else 0,
        'overkill_rate': (len(good_imgs) - corrects1_good) / all_imgs_size if all_imgs_size != 0 else 0,
        'accuracy': (corrects1_bad + corrects1_good) / all_imgs_size if all_imgs_size != 0 else 0
    }
    print('{}: {}'.format(recipe_name, result[recipe_name]))
    return result


def test_server_data(data_path, save_diff_path, only_test_recipe=None):
    diff_save_path = os.path.join(save_diff_path, 'diff_data')
    os.makedirs(diff_save_path, exist_ok=True)

    # csv
    test_result_csv_path = os.path.join(save_diff_path, 'test_result.csv')
    test_result_headers = ['Recipe', 'all', 'Bad', 'Good', 'underkill', 'overkill', 'underkill_rate', 'overkill_rate', 'accuracy']

    result = defaultdict(dict)
    for per_recipe in tqdm(os.listdir(data_path)):
        recipe_path = os.path.join(data_path, per_recipe)
        if len(test_recipe) != 0 and per_recipe not in only_test_recipe:
            print('测试单独recipe:', per_recipe)
            continue
        if not os.path.isdir(recipe_path):
            continue

        result = test_category_dataset(result, recipe_path, per_recipe, diff_save_path)

    # 写入表格
    rows = []
    for recipe_name, value in result.items():
        rows.append([recipe_name, value['all'], value['Bad'], value['Good'], value['underkill'], value['overkill'],
                                  value['underkill_rate'], value['overkill_rate'], value['accuracy']])
    write_csv(test_result_csv_path, rows, test_result_headers)


def parse_args():
    parser = argparse.ArgumentParser(description='dcl parameters')
    parser.add_argument('--data', dest='dataset', default='CUB', type=str)
    parser.add_argument('--save', dest='resume', default=None, type=str)
    parser.add_argument('--save_dir', dest='save_dir', default='./net_model', type=str)
    parser.add_argument('--backbone', dest='backbone', default='resnet50', type=str)
    parser.add_argument('--resume_checkpoint', default='False', type=str)
    parser.add_argument('--replace_online_model', default='False', type=str)
    parser.add_argument('--auto_resume', dest='auto_resume', action='store_true')
    parser.add_argument('--epoch', dest='epoch', default=360, type=int)
    parser.add_argument('--log_interval', default=1, type=int)
    parser.add_argument('--tb', dest='train_batch', default=32, type=int)
    parser.add_argument('--vb', dest='val_batch', default=2, type=int)
    parser.add_argument('--sp', dest='save_point', default=5000, type=int)
    parser.add_argument('--cp', dest='check_point', default=5000, type=int)
    parser.add_argument('--lr', dest='base_lr', default=0.0008, type=float)
    parser.add_argument('--lr_step', dest='decay_step', default=20, type=int)
    parser.add_argument('--cls_lr_ratio', dest='cls_lr_ratio', default=10.0, type=float)
    parser.add_argument('--start_epoch', dest='start_epoch', default=0,  type=int)
    parser.add_argument('--tnw', dest='train_num_workers', default=12, type=int)
    parser.add_argument('--vnw', dest='val_num_workers', default=12, type=int)
    parser.add_argument('--detail', dest='discribe', default='', type=str)
    parser.add_argument('--size', dest='resize_resolution', default=512, type=int)
    parser.add_argument('--crop', dest='crop_resolution', default=448, type=int)
    parser.add_argument('--cls_2', dest='cls_2', action='store_true')
    parser.add_argument('--cls_mul', dest='cls_mul', action='store_true')
    parser.add_argument('--use_sagan', dest='use_sagan', action='store_true')
    parser.add_argument('--swap_num', default=[7, 7], nargs=2, metavar=('swap1', 'swap2'), type=int, help='specify a range')
    parser.add_argument('--use_language', action='store_true')

    # 本脚本新增
    parser.add_argument('--test_path',  default='/data1/pzh/data/jssi/aoi/val/', type=str)
    parser.add_argument('--test_save_path', default='/data1/pzh/data/jssi/diff', type=str)

    args = parser.parse_args()
    return args

copy_diff = True
remove_diff = True

if __name__ == '__main__':
    args = parse_args()
    Config = LoadConfig(args, 'val')
    Config.use_sagan = args.use_sagan
    Config.use_language = args.use_language
    Config.cls_2 = args.cls_2
    Config.cls_2xmul = args.cls_mul
    Config.replace_online_model = args.replace_online_model
    Config.log_interval = args.log_interval
    assert Config.cls_2 ^ Config.cls_2xmul

    model = MainModel(Config)
    model_dict = model.state_dict()
    pretrained_dict = torch.load(args.resume)
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.cuda()
    model.train(False)

    bert_tokenizer = BertTokenizer.from_pretrained('/data1/pzh/project/local/hugging_face/bert-base-uncased')

    test_path = args.test_path
    test_recipe = []  # 测试某几个recipe，为空时不测
    save_path = args.test_save_path
    test_server_data(test_path, save_path, test_recipe)
    print('diff data save path is:{}'.format(save_path))
