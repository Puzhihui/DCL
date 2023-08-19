import os
import pandas as pd
import torch

from transforms import transforms
from utils.autoaugment import ImageNetPolicy

# pretrained model checkpoints
pretrained_model = {'resnet50' : './models/pretrained/resnet50-19c8e357.pth',
                    "se_resnext101_32x4d": "./models/pretrained/se_resnext101_32x4d-3b2fe3d8.pth"}
customize_model = ['efficientnet-b4']

# transforms dict
def load_data_transformers(resize_reso=512, crop_reso=448, swap_num=[7, 7]):
    center_resize = 600
    Normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    data_transforms = {
       	'swap': transforms.Compose([
            transforms.Randomswap((swap_num[0], swap_num[1])),
        ]),
        'adc_aug': transforms.Compose([
            transforms.CenterCrop((800, 800)),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomHorizontalFlip(0.5),
            transforms.CenterCrop((600, 600)),
            transforms.ColorJitter(0.2, 0.1, 0.1, 0.01),
            transforms.RandomRotation(degrees=15),
            transforms.CenterCrop((crop_reso, crop_reso)),
        ]),
        'adc_resize_aug': transforms.Compose([
            transforms.CenterCrop((832, 832)),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(0.2, 0.1, 0.1, 0.01),
            transforms.RandomRotation(degrees=15),
            transforms.Resize((crop_reso, crop_reso)),
        ]),
        'adc_oi_center_aug': transforms.Compose([
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(0.2, 0.1, 0.1, 0.01),
            transforms.RandomRotation(degrees=15),
            transforms.CenterCrop((crop_reso, crop_reso)),
        ]),
        'adc_oi_resize_aug': transforms.Compose([
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(0.2, 0.1, 0.1, 0.01),
            transforms.RandomRotation(degrees=15),
            transforms.Resize((crop_reso, crop_reso)),
        ]),
        'adc_train_totensor': transforms.Compose([
            transforms.CenterCrop((crop_reso, crop_reso)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'adc_val_totensor': transforms.Compose([
            transforms.CenterCrop((crop_reso, crop_reso)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'adc_val_resize_totensor': transforms.Compose([
            transforms.CenterCrop((832, 832)),
            transforms.Resize((crop_reso, crop_reso)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'adc_val_oi_resize_totensor': transforms.Compose([
            transforms.Resize((crop_reso, crop_reso)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'None': None,
    }
    return data_transforms


class LoadConfig(object):
    def __init__(self, args, version):
        if version == 'train':
            get_list = ['train', 'val']
        elif version == 'val':
            get_list = ['val']
        elif version == 'test':
            get_list = ['test']
        else:
            raise Exception("train/val/test ???\n")

        ###############################
        #### add dataset info here ####
        ###############################

        # put image data in $PATH/data
        # put annotation txt file in $PATH/anno

        if args.dataset == 'jssi_aoi':
            self.dataset = args.dataset
            self.rawdata_root = '/data3/pzh/data/jssi/aoi'
            self.anno_root = './datasets/jssi_aoi'
            self.numcls = 2
        elif args.dataset == 'jssi_photo_center':
            self.dataset = args.dataset
            self.rawdata_root = '/data3/pzh/data/jssi/photo'
            self.anno_root = './datasets/jssi_photo_center'
            self.numcls = 2
        elif args.dataset == 'jssi_photo_resize':
            self.dataset = args.dataset
            self.rawdata_root = '/data3/pzh/data/jssi/photo'
            self.anno_root = './datasets/jssi_photo_resize'
            self.numcls = 2
        elif args.dataset == 'jssi_photo_center_resize':
            self.dataset = args.dataset
            self.rawdata_root = '/data3/pzh/data/jssi/photo'
            self.anno_root = './datasets/jssi_photo_center_resize'
            self.numcls = 2
        elif args.dataset == 'ht_less500_center_resize':
            self.dataset = args.dataset
            self.rawdata_root = '/data4/exp_data/'
            self.anno_root = './datasets/ht_less500_center_resize'
            self.numcls = 2
        elif args.dataset == 'ht_more500_center_resize':
            self.dataset = args.dataset
            self.rawdata_root = '/data4/exp_data/'
            self.anno_root = './datasets/ht_more500_center_resize'
            self.numcls = 2
        elif args.dataset == 'smic_om_back':
            self.dataset = args.dataset
            self.rawdata_root = r'D:\Solution\datas\smic_om_back' # /data3/pzh/data/smic/smic_om_3
            self.anno_root = './datasets/smic_om_back'
            self.numcls = 4
        elif args.dataset == 'smic_om_front':
            self.dataset = args.dataset
            self.rawdata_root = r'D:\Solution\datas\smic_om_front'
            self.anno_root = './datasets/smic_om_front'
            self.numcls = 4
        elif args.dataset == 'smic_om_front_2':
            self.dataset = args.dataset
            self.rawdata_root = r'D:\Solution\datas\smic_om_front_2'
            self.anno_root = './datasets/smic_om_front_2'
            self.numcls = 2
        else:
            raise Exception('dataset not defined ???')

        # annotation file organized as :
        # path/image_name cls_num\n

        if 'train' in get_list:
             self.train_anno = pd.read_csv(os.path.join(self.anno_root, 'train.txt'),\
                                           sep=", ",\
                                           header=None,\
                                           names=['ImageName', 'label'],
                                           engine='python')

        if 'val' in get_list:
            self.val_anno = pd.read_csv(os.path.join(self.anno_root, 'val.txt'),\
                                           sep=", ",\
                                           header=None,\
                                           names=['ImageName', 'label'],
                                           engine='python')

        if 'test' in get_list:
            self.test_anno = pd.read_csv(os.path.join(self.anno_root, 'test.txt'),\
                                           sep=", ",\
                                           header=None,\
                                           names=['ImageName', 'label'],
                                           engine='python')

        self.swap_num = args.swap_num

        self.save_dir = './net_model'
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.backbone = args.backbone

        self.use_dcl = True
        self.use_backbone = False if self.use_dcl else True
        self.use_Asoftmax = False
        self.use_focal_loss = False
        self.use_fpn = False
        self.use_hier = False

        self.weighted_sample = False
        self.cls_2 = True
        self.cls_2xmul = False

        self.log_folder = './logs'
        if not os.path.exists(self.log_folder):
            os.mkdir(self.log_folder)


class smic_back_online():
    online_model_dir = r"D:\Solution\code\smic\automatic_defect_classification_server\service\weights\smic"
    online_model_name = "smic_back_m6.pth"
    best_model_txt = r"D:\Solution\code\smic\DCL\smic_tools\back_best_model_path.txt"

    train_data_path = r'D:\Solution\datas\smic_om_back'
    txt_root_path =   r'D:\Solution\code\smic\DCL\datasets\smic_om_back'

class smic_front_online():
    online_model_dir = r"D:\Solution\code\smic\automatic_defect_classification_server\service\weights\smic"
    online_model_name = "smic_front_m6.pth"
    best_model_txt = r"D:\Solution\code\smic\DCL\smic_tools\front_best_model_path.txt"

    train_data_path = r'D:\Solution\datas\smic_om_front'
    txt_root_path =   r'D:\Solution\code\smic\DCL\datasets\smic_om_front'
