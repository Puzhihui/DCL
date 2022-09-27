#coding=utf-8
import os
import json
import csv
import argparse
import pandas as pd
import numpy as np
from math import ceil
from tqdm import tqdm
import pickle
import shutil

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torchvision import datasets, models
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from transforms import transforms
from models.LoadModel import MainModel
from utils.dataset_DCL import collate_fn4train, collate_fn4test, collate_fn4val, dataset
from config import LoadConfig, load_data_transformers
from utils.test_tool import set_text, save_multi_img, cls_base_acc

import pdb

os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# python test_my.py --data jssi_photo --use_center --use_resize  --test_txt ./test.txt --save ./model.pth

def parse_args():
    parser = argparse.ArgumentParser(description='dcl parameters')
    parser.add_argument('--data', dest='dataset',
                        default='jssi_photo', type=str)
    parser.add_argument('--backbone', dest='backbone',
                        default='efficientnet-b4', type=str)
    parser.add_argument('--test_txt', dest='test_txt',
                        default='', type=str)
    parser.add_argument('--use_center', dest='use_center',
                        default=False,  action='store_true')
    parser.add_argument('--use_resize', dest='use_resize',
                        default=False,  action='store_true')
    parser.add_argument('--b', dest='batch_size',
                        default=64, type=int)
    parser.add_argument('--nw', dest='num_workers',
                        default=12, type=int)
    parser.add_argument('--ver', dest='version',
                        default='val', type=str)
    parser.add_argument('--save', dest='resume',
                        default=None, type=str)
    parser.add_argument('--size', dest='resize_resolution',
                        default=512, type=int)
    parser.add_argument('--crop', dest='crop_resolution',
                        default=448, type=int)
    parser.add_argument('--ss', dest='save_suffix',
                        default=None, type=str)
    parser.add_argument('--acc_report', dest='acc_report',
                        action='store_true')
    parser.add_argument('--swap_num', default=[7, 7],
                    nargs=2, metavar=('swap1', 'swap2'),
                    type=int, help='specify a range')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    resume = args.resume
    print(args)
    # if args.submit:
    #     args.version = 'test'
    #     if args.save_suffix == '':
    #         raise Exception('**** miss --ss save suffix is needed. ')

    Config = LoadConfig(args, args.version)
    Config.cls_2xmul = True
    args.acc_report = True
    transformers = load_data_transformers(args.resize_resolution, args.crop_resolution, args.swap_num)
    test_anno = pd.read_csv(args.test_txt, sep=", ", header=None, names=['ImageName', 'label'], engine='python')
    data_set = dataset(Config, \
                       # anno=Config.val_anno if args.version == 'val' else Config.test_anno, \
                       anno=test_anno, \
                       swap=transformers["None"], \
                       # totensor=transformers['adc_val_totensor'],\
                       totensor=transformers['adc_val_resize_totensor'], \
                       test=True)

    dataloader = torch.utils.data.DataLoader(data_set, \
                                             batch_size=args.batch_size, \
                                             shuffle=False, \
                                             num_workers=args.num_workers, \
                                             collate_fn=collate_fn4test)

    setattr(dataloader, 'total_item_len', len(data_set))

    cudnn.benchmark = True

    model = MainModel(Config)
    model_dict = model.state_dict()
    pretrained_dict = torch.load(resume)
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.cuda()
    # model = nn.DataParallel(model)

    model.train(False)
    with torch.no_grad():
        val_corrects1 = 0
        val_corrects2 = 0
        val_corrects3 = 0
        val_size = ceil(len(data_set) / dataloader.batch_size)
        result_gather = {}
        count_bar = tqdm(total=dataloader.__len__())
        for batch_cnt_val, data_val in enumerate(dataloader):
            count_bar.update(1)
            inputs, labels, img_name = data_val
            inputs = Variable(inputs.cuda())
            labels = Variable(torch.from_numpy(np.array(labels)).long().cuda())

            outputs = model(inputs)
            outputs_pred = outputs[0] + outputs[1][:, 0:Config.numcls] + outputs[1][:, Config.numcls:2 * Config.numcls]
            # outputs_pred = outputs[0]

            # top3_val, top3_pos = torch.topk(outputs_pred, 3)
            top3_val, top3_pos = torch.topk(outputs_pred, 2)

            if args.version == 'val' or args.version == 'test':
                batch_corrects1 = torch.sum((top3_pos[:, 0] == labels)).data.item()
                val_corrects1 += batch_corrects1
                batch_corrects2 = torch.sum((top3_pos[:, 1] == labels)).data.item()
                val_corrects2 += (batch_corrects2 + batch_corrects1)
                # batch_corrects3 = torch.sum((top3_pos[:, 2] == labels)).data.item()
                # val_corrects3 += (batch_corrects3 + batch_corrects2 + batch_corrects1)
                val_corrects3 += (batch_corrects2 + batch_corrects1)

            if args.acc_report:
                for sub_name, sub_cat, sub_val, sub_label in zip(img_name, top3_pos.tolist(), top3_val.tolist(),
                                                                 labels.tolist()):
                    result_gather[sub_name] = {'top1_cat': sub_cat[0], 'top2_cat': sub_cat[1], 'top3_cat': sub_cat[1],
                                               'top1_val': sub_val[0], 'top2_val': sub_val[1], 'top3_val': sub_val[1],
                                               'label': sub_label}
                    # print(result_gather[sub_name])
    if args.acc_report:
        torch.save(result_gather, 'result_gather_%s' % resume.split('/')[-1][:-4] + '.pt')

    count_bar.close()

    if args.acc_report:
        val_acc1 = val_corrects1 / len(data_set)
        val_acc2 = val_corrects2 / len(data_set)
        # val_acc3 = val_corrects3 / len(data_set)
        # print('%sacc1 %f%s\n%sacc2 %f%s\n%sacc3 %f%s\n'%(8*'-', val_acc1, 8*'-', 8*'-', val_acc2, 8*'-', 8*'-',  val_acc3, 8*'-'))
        print('%sacc1 %f%s\n%sacc2 %f%s\n' % (8 * '-', val_acc1, 8 * '-', 8 * '-', val_acc2, 8 * '-'))

        cls_top1, cls_top3, cls_count = cls_base_acc(result_gather)

        acc_report_io = open('acc_report_%s_%s.json' % (args.save_suffix, resume.split('/')[-1]), 'w')
        json.dump({'val_acc1': val_acc1,
                   'val_acc2': val_acc2,
                   'cls_top1': cls_top1,
                   'cls_count': cls_count}, acc_report_io)
        acc_report_io.close()


def main_CenterAndResize():
    args = parse_args()
    use_center = args.use_center
    use_resize = args.use_resize
    resume = args.resume
    print(args)
    # if args.submit:
    #     args.version = 'test'
    #     if args.save_suffix == '':
    #         raise Exception('**** miss --ss save suffix is needed. ')

    Config = LoadConfig(args, args.version)
    Config.cls_2xmul = True
    args.acc_report = True
    transformers = load_data_transformers(args.resize_resolution, args.crop_resolution, args.swap_num)
    test_anno = pd.read_csv(args.test_txt, sep=", ", header=None, names=['ImageName', 'label'], engine='python')
    data_set_center = dataset(Config, \
                       # anno=Config.val_anno if args.version == 'val' else Config.test_anno, \
                       anno=test_anno, \
                       swap=transformers["None"], \
                       totensor=transformers['adc_val_totensor'],\
                       test=True)
    data_set_resize = dataset(Config, \
                       anno=test_anno, \
                       swap=transformers["None"], \
                       totensor=transformers['adc_val_resize_totensor'], \
                       test=True)

    dataloader_center = torch.utils.data.DataLoader(data_set_center, \
                                                    batch_size=args.batch_size, \
                                                    shuffle=False, \
                                                    num_workers=args.num_workers, \
                                                    collate_fn=collate_fn4test)
    dataloader_resize = torch.utils.data.DataLoader(data_set_resize, \
                                                    batch_size=args.batch_size, \
                                                    shuffle=False, \
                                                    num_workers=args.num_workers, \
                                                    collate_fn=collate_fn4test)

    setattr(dataloader_center, 'total_item_len', len(dataloader_center))

    cudnn.benchmark = True

    model = MainModel(Config)
    model_dict = model.state_dict()
    pretrained_dict = torch.load(resume)
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.cuda()
    # model = nn.DataParallel(model)

    model.train(False)
    with torch.no_grad():
        val_corrects1 = 0
        val_corrects2 = 0
        # val_size = ceil(len(data_set) / dataloader.batch_size)
        result_gather = {}
        count_bar = tqdm(total=dataloader_center.__len__())
        for batch_cnt_val, (data_val_center, data_val_resize) in enumerate(zip(dataloader_center, dataloader_resize)):
            count_bar.update(1)
            batch_corrects1_result, batch_corrects2_result = [], []

            # center
            if use_center:
                inputs_center, labels_center, img_name_center = data_val_center
                inputs_center = Variable(inputs_center.cuda())
                labels_center = Variable(torch.from_numpy(np.array(labels_center)).long().cuda())
                outputs_center = model(inputs_center)
                outputs_pred_center = outputs_center[0] + outputs_center[1][:, 0:Config.numcls] + outputs_center[1][:, Config.numcls:2 * Config.numcls]
                # outputs_pred_center = outputs_center[0]
                top3_val_center, top3_pos_center = torch.topk(outputs_pred_center, 2)

                if args.version == 'val' or args.version == 'test':
                    batch_corrects1_result_center = (top3_pos_center[:, 0] == labels_center).cpu().numpy().tolist()
                    batch_corrects2_result_center = (top3_pos_center[:, 1] == labels_center).cpu().numpy().tolist()
            else:
                batch_corrects1_result_center, batch_corrects2_result_center = None, None


            # resize
            if use_resize:
                inputs_resize, labels_resize, img_name_resize = data_val_resize
                inputs_resize = Variable(inputs_resize.cuda())
                labels_resize = Variable(torch.from_numpy(np.array(labels_resize)).long().cuda())
                outputs_resize = model(inputs_resize)
                outputs_pred_resize = outputs_resize[0] + outputs_resize[1][:, 0:Config.numcls] + outputs_resize[1][:, Config.numcls:2 * Config.numcls]
                # outputs_pred_resize = outputs_resize[0]
                top3_val_resize, top3_pos_resize = torch.topk(outputs_pred_resize, 2)
                if args.version == 'val' or args.version == 'test':
                    batch_corrects1_result_resize = (top3_pos_resize[:, 0] == labels_resize).cpu().numpy().tolist()
                    batch_corrects2_result_resize = (top3_pos_resize[:, 1] == labels_resize).cpu().numpy().tolist()
            else:
                batch_corrects1_result_resize, batch_corrects2_result_resize = None, None

            if use_center and not use_resize:
                batch_corrects1_result = batch_corrects1_result_center
                batch_corrects2_result = batch_corrects2_result_center
                val_corrects1 += batch_corrects1_result.count(True)
                val_corrects2 += (batch_corrects2_result.count(True) + batch_corrects1_result.count(True))
                for sub_name, sub_cat, sub_val, sub_label in zip(img_name_center, top3_pos_center.tolist(), top3_val_center.tolist(), labels_center.tolist()):
                    result_gather[sub_name] = {'top1_cat': sub_cat[0], 'top2_cat': sub_cat[1], 'top3_cat': sub_cat[1],
                                               'top1_val': sub_val[0], 'top2_val': sub_val[1], 'top3_val': sub_val[1],
                                               'label': sub_label}
            elif use_resize and not use_center:
                batch_corrects1_result = batch_corrects1_result_resize
                batch_corrects2_result = batch_corrects2_result_resize
                val_corrects1 += batch_corrects1_result.count(True)
                val_corrects2 += (batch_corrects2_result.count(True) + batch_corrects1_result.count(True))
                for sub_name, sub_cat, sub_val, sub_label in zip(img_name_resize, top3_pos_resize.tolist(), top3_val_resize.tolist(), labels_resize.tolist()):
                    result_gather[sub_name] = {'top1_cat': sub_cat[0], 'top2_cat': sub_cat[1], 'top3_cat': sub_cat[1],
                                               'top1_val': sub_val[0], 'top2_val': sub_val[1], 'top3_val': sub_val[1],
                                               'label': sub_label}
            elif use_resize and use_center:
                for center_1, resize_1, center_2, resize_2, per_label in zip(batch_corrects1_result_center, batch_corrects1_result_resize,
                                                                             batch_corrects2_result_center, batch_corrects2_result_resize,
                                                                             labels_center):
                    top1_center_resize = center_1 | resize_1 if per_label == 0 else center_1 & resize_1
                    top2_center_resize = per_label
                    batch_corrects1_result.append(top1_center_resize)
                    batch_corrects2_result.append(top2_center_resize)
                val_corrects1 += batch_corrects1_result.count(True)
                val_corrects2 += (batch_corrects2_result.count(True) + batch_corrects1_result.count(True))

                for sub_name, sub_label, top1_res, top2_res in zip(img_name_center, labels_center.tolist(), batch_corrects1_result, batch_corrects2_result):
                    sub_cat, sub_val = [-1, -1], [-1, -1]
                    sub_cat[0] = sub_label if top1_res else (1 if sub_label == 0 else 0)
                    sub_cat[1] = sub_label
                    sub_val[0], sub_val[1] = 0, 0
                    result_gather[sub_name] = {'top1_cat': sub_cat[0], 'top2_cat': sub_cat[1], 'top3_cat': sub_cat[1],
                                               'top1_val': sub_val[0], 'top2_val': sub_val[1], 'top3_val': sub_val[1],
                                               'label': sub_label}

    count_bar.close()

    if args.acc_report:
        val_acc1 = val_corrects1 / len(data_set_center)
        val_acc2 = val_corrects2 / len(data_set_center)
        print('%sacc1 %f%s\n%sacc2 %f%s\n' % (8 * '-', val_acc1, 8 * '-', 8 * '-', val_acc2, 8 * '-'))

        cls_top1, cls_top3, cls_count = cls_base_acc(result_gather)

        # acc_report_io = open('acc_report_%s_%s.json' % (args.save_suffix, resume.split('/')[-1]), 'w')
        # json.dump({'val_acc1': val_acc1,
        #            'val_acc2': val_acc2,
        #            'cls_top1': cls_top1,
        #            'cls_count': cls_count}, acc_report_io)
        # acc_report_io.close()


if __name__ == '__main__':
    # main()
    main_CenterAndResize()



