#coding=utf-8
import os
import datetime
import argparse
import logging
import pandas as pd
import platform

import torch
import torch.nn as nn
from  torch.nn import CrossEntropyLoss
import torch.utils.data as torchdata
from torchvision import datasets, models
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

from transforms import transforms
from utils.train_model import train
from models.LoadModel import MainModel
from config import LoadConfig, load_data_transformers, smic_front_online, smic_back_online
from utils.dataset_DCL import collate_fn4train, collate_fn4val, collate_fn4test, collate_fn4backbone, dataset
from logserver import LogServer

import pdb

os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

# ========================================================日志模块========================================================
def get_str_datetime():
    return str(datetime.datetime.now().year) + '_' + str(datetime.datetime.now().month) + '_' + str(
        datetime.datetime.now().day) + '_' + str(datetime.datetime.now().hour)

# nohup python train_smic.py --tb 32 >nohup.log 2>&1 &

# parameters setting
def parse_args():
    parser = argparse.ArgumentParser(description='dcl parameters')
    parser.add_argument('--mode', default='Back', type=str)
    parser.add_argument('--client', default='M47', type=str)
    parser.add_argument('--data', dest='dataset',
                        default='smic_om_3', type=str)
    parser.add_argument('--save', dest='resume',
                        default=None, type=str)
    parser.add_argument('--save_dir', dest='save_dir',
                        default=r'D:\Solution\code\smic\DCL\net_model', type=str)
    parser.add_argument('--backbone', dest='backbone',
                        default='resnet50', type=str)
    parser.add_argument('--auto_resume', dest='auto_resume',
                        action='store_true')
    parser.add_argument('--epoch', dest='epoch',
                        default=50, type=int)
    parser.add_argument('--tb', dest='train_batch',
                        default=16, type=int)
    parser.add_argument('--vb', dest='val_batch',
                        default=16, type=int)
    parser.add_argument('--sp', dest='save_point',
                        default=5000, type=int)
    parser.add_argument('--cp', dest='check_point',
                        default=5000, type=int)
    parser.add_argument('--lr', dest='base_lr',
                        default=0.0008, type=float)
    parser.add_argument('--lr_step', dest='decay_step',
                        default=20, type=int)
    parser.add_argument('--cls_lr_ratio', dest='cls_lr_ratio',
                        default=10.0, type=float)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        default=0,  type=int)
    parser.add_argument('--tnw', dest='train_num_workers',
                        default=4, type=int)
    parser.add_argument('--vnw', dest='val_num_workers',
                        default=4, type=int)
    parser.add_argument('--detail', dest='discribe',
                        default='train', type=str)
    parser.add_argument('--size', dest='resize_resolution',
                        default=512, type=int)
    parser.add_argument('--crop', dest='crop_resolution',
                        default=352, type=int)
    parser.add_argument('--cls_2', dest='cls_2',
                        action='store_true')
    parser.add_argument('--cls_mul', dest='cls_mul',
                        default=True, action='store_true')
    parser.add_argument('--swap_num', default=[5, 5],
                    nargs=2, metavar=('swap1', 'swap2'),
                    type=int, help='specify a range')
    args = parser.parse_args()
    return args

def auto_load_resume(load_dir):
    folders = os.listdir(load_dir)
    date_list = [int(x.split('_')[1].replace(' ',0)) for x in folders]
    choosed = folders[date_list.index(max(date_list))]
    weight_list = os.listdir(os.path.join(load_dir, choosed))
    acc_list = [x[:-4].split('_')[-1] if x[:7]=='weights' else 0 for x in weight_list]
    acc_list = [float(x) for x in acc_list]
    choosed_w = weight_list[acc_list.index(max(acc_list))]
    return os.path.join(load_dir, choosed, choosed_w)


args = parse_args()
mode = args.mode
client = args.client
args.dataset = "{}_{}".format(mode, client)
if mode == "Back":
    cfg_mode = smic_back_online()
elif mode == "Front":
    cfg_mode = smic_front_online()
else:
    raise "Mode error!!!"
if __name__ == '__main__':
    # ========================================================日志模块========================================================
    log_path = r"D:\Solution\log\train_model" if platform.system().lower() == 'windows' else './logs'
    os.makedirs(log_path, exist_ok=True)
    log_server = LogServer(app='adc-dcl-train', log_path=log_path)
    filename = get_str_datetime()
    log_server.re_configure_logging('adc-dcl-train' + str(filename) + "_log.txt")
    print("logfile is: ", str(filename) + "_log.txt")
    log_server.logging("logfile is %s" % (str(filename) + "_log.txt"))
    print("**********load log config file success*******************")
    # ========================================================日志模块========================================================

    log_server.logging("torch num threads:{}".format(torch.get_num_threads()))
    set_torch_threads = 2 if torch.get_num_threads() > 2 else torch.get_num_threads()
    torch.set_num_threads(set_torch_threads)
    log_server.logging("set_torch_threads: {}, now torch num threads:{}".format(set_torch_threads, torch.get_num_threads()))

    print(args, flush=True)
    _ = log_server.logging("{}".format(args)) if log_server else 1
    Config = LoadConfig(args, 'train')
    Config.save_dir = args.save_dir
    os.makedirs(Config.save_dir, exist_ok=True)
    Config.cls_2 = args.cls_2
    Config.cls_2xmul = args.cls_mul
    assert Config.cls_2 ^ Config.cls_2xmul

    transformers = load_data_transformers(args.resize_resolution, args.crop_resolution, args.swap_num)

    # inital dataloader
    train_set = dataset(Config = Config, \
                        swap_size=args.swap_num,\
                        anno = Config.train_anno,\
                        # common_aug = transformers["adc_aug"],\
                        # common_aug = transformers["adc_oi_center_aug"],\
                        common_aug = transformers["adc_oi_resize_aug"],\
                        resize_aug = transformers["adc_oi_resize_aug"],\
                        swap = transformers["swap"],\
                        totensor = transformers["adc_train_totensor"],\
                        train = True)

    trainval_set = dataset(Config = Config, \
                        swap_size=args.swap_num, \
                        anno = Config.train_anno,\
                        common_aug = transformers["None"],\
                        swap = transformers["None"],\
                        totensor = transformers["adc_val_oi_resize_totensor"],\
                        train = False,
                        train_val = True)

    val_set = dataset(Config = Config, \
                      swap_size=args.swap_num, \
                      anno = Config.val_anno,\
                      common_aug = transformers["None"],\
                      swap = transformers["None"],\
                      totensor = transformers["adc_val_oi_resize_totensor"],\
                      val_resize_totensor = transformers["adc_val_oi_resize_totensor"],\
                      test=True)

    dataloader = {}
    dataloader['train'] = torch.utils.data.DataLoader(train_set,\
                                                batch_size=args.train_batch,\
                                                shuffle=True,\
                                                num_workers=args.train_num_workers,\
                                                collate_fn=collate_fn4train if not Config.use_backbone else collate_fn4backbone,
                                                #drop_last=True if Config.use_backbone else False,
                                                drop_last=True,
                                                pin_memory=True)

    setattr(dataloader['train'], 'total_item_len', len(train_set))

    dataloader['trainval'] = torch.utils.data.DataLoader(trainval_set,\
                                                batch_size=args.val_batch,\
                                                shuffle=False,\
                                                num_workers=args.val_num_workers,\
                                                collate_fn=collate_fn4val if not Config.use_backbone else collate_fn4backbone,
                                                drop_last=True if Config.use_backbone else False,
                                                pin_memory=True)

    setattr(dataloader['trainval'], 'total_item_len', len(trainval_set))
    setattr(dataloader['trainval'], 'num_cls', Config.numcls)

    dataloader['val'] = torch.utils.data.DataLoader(val_set,\
                                                batch_size=args.val_batch,\
                                                shuffle=False,\
                                                num_workers=args.val_num_workers,\
                                                collate_fn=collate_fn4test if not Config.use_backbone else collate_fn4backbone,
                                                drop_last=True if Config.use_backbone else False,
                                                pin_memory=True)

    setattr(dataloader['val'], 'total_item_len', len(val_set))
    setattr(dataloader['val'], 'num_cls', Config.numcls)


    cudnn.benchmark = True

    print('Choose model and train set', flush=True)
    _ = log_server.logging("Choose model and train set") if log_server else 1
    model = MainModel(Config)

    # load model
    args.resume = os.path.join(cfg_mode.online_model_dir, cfg_mode.online_model_name)
    if not os.path.exists(args.resume):
        args.resume = None
    if (args.resume is None) and (not args.auto_resume):
        print('train from imagenet pretrained models ...', flush=True)
        _ = log_server.logging("train from imagenet pretrained models ...") if log_server else 1
    else:
        if not args.resume is None:
            resume = args.resume
            print('load from pretrained checkpoint %s ...'% resume, flush=True)
            _ = log_server.logging('load from pretrained checkpoint %s ...'% resume) if log_server else 1
        elif args.auto_resume:
            resume = auto_load_resume(Config.save_dir)
            print('load from %s ...'%resume, flush=True)
            _ = log_server.logging('load from %s ...'%resume) if log_server else 1
        else:
            raise Exception("no checkpoints to load")

        model_dict = model.state_dict()
        pretrained_dict = torch.load(resume)
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    print('Set cache dir', flush=True)
    _ = log_server.logging("Set cache dir") if log_server else 1
    time = datetime.datetime.now()
    filename = '%s_%d%d%d_%s'%(args.discribe, time.month, time.day, time.hour, Config.dataset)
    save_dir = os.path.join(Config.save_dir, filename)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.cuda()
    model = nn.DataParallel(model)

    # optimizer prepare
    if Config.use_backbone:
        ignored_params = list(map(id, model.module.classifier.parameters()))
    else:
        ignored_params1 = list(map(id, model.module.classifier.parameters()))
        ignored_params2 = list(map(id, model.module.classifier_swap.parameters()))
        ignored_params3 = list(map(id, model.module.Convmask.parameters()))

        ignored_params = ignored_params1 + ignored_params2 + ignored_params3
    print('the num of new layers:', len(ignored_params), flush=True)
    _ = log_server.logging("the num of new layers:{}".format(len(ignored_params))) if log_server else 1
    base_params = filter(lambda p: id(p) not in ignored_params, model.module.parameters())

    lr_ratio = args.cls_lr_ratio
    base_lr = args.base_lr
    if Config.use_backbone:
        optimizer = optim.SGD([{'params': base_params},
                               {'params': model.module.classifier.parameters(), 'lr': base_lr}], lr = base_lr, momentum=0.9)
    else:
        optimizer = optim.SGD([{'params': base_params},
                               {'params': model.module.classifier.parameters(), 'lr': lr_ratio*base_lr},
                               {'params': model.module.classifier_swap.parameters(), 'lr': lr_ratio*base_lr},
                               {'params': model.module.Convmask.parameters(), 'lr': lr_ratio*base_lr},
                              ], lr = base_lr, momentum=0.9)


    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=0.1)

    # train entry
    train(cfg_mode.online_model_name,
          Config,
          model,
          epoch_num=args.epoch,
          start_epoch=args.start_epoch,
          optimizer=optimizer,
          exp_lr_scheduler=exp_lr_scheduler,
          data_loader=dataloader,
          save_dir=save_dir,
          data_size=args.crop_resolution,
          savepoint=args.save_point,
          checkpoint=args.check_point,
          log_server=log_server)

    f = open(cfg_mode.best_model_txt, "w", encoding="utf-8")
    f.write(save_dir)
    f.close()
