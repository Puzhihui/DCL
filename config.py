import os
import pandas as pd
import torch

from transforms import transforms
# from utils.autoaugment import ImageNetPolicy

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
    def __init__(self, args, version, only_dataset=False):
        if version == 'train':
            self.get_list = ['train', 'val']
        elif version == 'val':
            self.get_list = ['val']
        elif version == 'test':
            self.get_list = ['test']
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

        # 中芯北方M6
        # 数据格式 D:\Solution\datas + Front_M6\Front_M6_val + recipeName + false\discolor
        elif args.dataset == 'Back_M6':
            self.dataset = args.dataset
            self.train_path_list = [r'D:\Solution\datas\Back_M6', r'D:\Solution\datas\BackDark_M6']
            self.val_path_list = [r'D:\Solution\datas\Back_M6_val', r'D:\Solution\datas\BackDark_M6_val']

            self.anno_root = './datasets/Back_M6'
            self.multi_classes = {'BSDC': "0", 'BSNS': "1", 'SCRATCH': "2", "FALSE": "3", 'BSCS': "4", "BSCSS": "5"}
            self.numcls = len(self.multi_classes)
            # online setting
            self.online_model = r'D:\Solution\code\smic\automatic_defect_classification_server\service\weights\smic\back_model.pth'
        elif args.dataset == 'Front_M6':
            self.dataset = args.dataset
            # 明场
            self.train_path_list = [r'D:\Solution\datas\Front_M6', r'D:\Solution\datas\FrontDark_M6']
            self.val_path_list = [r'D:\Solution\datas\Front_M6_val', r'D:\Solution\datas\FrontDark_M6_val']

            self.anno_root = './datasets/Front_M6'
            self.multi_classes = {'PADC': "0", 'PAOH': "1", 'PASC': "2", "FALSE": "3", 'PASD': "4", 'SINR': "5", "PASP": "6", "PANS": "7"}
            self.numcls = len(self.multi_classes)
            # online setting
            self.online_model = r'D:\Solution\code\smic\automatic_defect_classification_server\service\weights\smic\front_model.pth'

        # 中芯京城M24
        elif args.dataset == 'Back_M24':
            self.dataset = args.dataset
            # 明场
            self.train_path_list = [r'D:\Solution\datas\Back_M24', r'D:\Solution\datas\BackDark_M24']
            self.val_path_list = [r'D:\Solution\datas\Back_M24_val', r'D:\Solution\datas\BackDark_M24_val']

            self.anno_root = './datasets/Back_M24'
            self.multi_classes = {'BSDC': "0", 'BSOH': "1", 'SCRATCH': "2", "FALSE": "3"}
            self.numcls = len(self.multi_classes)
            # online setting
            self.online_model = r'D:\Solution\code\smic\automatic_defect_classification_server\service\weights\smic\back_model.pth'
        elif args.dataset == 'Front_M24':
            self.dataset = args.dataset
            # 明场
            self.train_path_list = [r'D:\Solution\datas\Front_M24', r'D:\Solution\datas\FrontDark_M24']
            self.val_path_list = [r'D:\Solution\datas\Front_M24_val', r'D:\Solution\datas\FrontDark_M24_val']

            self.anno_root = './datasets/Front_M24'
            self.multi_classes = {'PADC': "0", 'PAOH': "1", 'PASC': "2", "FALSE": "3"}
            self.numcls = len(self.multi_classes)
            # online setting
            self.online_model = r'D:\Solution\code\smic\automatic_defect_classification_server\service\weights\smic\front_model.pth'

        # 中芯天津M47
        elif args.dataset == 'Back_M47':
            self.dataset = args.dataset
            # 明场
            self.train_path_list = [r'D:\Solution\datas\Back_M47', r'D:\Solution\datas\BackDark_M47']
            self.val_path_list = [r'D:\Solution\datas\Back_M47_val', r'D:\Solution\datas\BackDark_M47_val']

            self.anno_root = './datasets/Back_M47'
            self.multi_classes = {'false': "0", 'MissingCorner': "1", 'other': "2", "particle": "3", 'scratch': "4", "discolor": "5"}
            self.numcls = len(self.multi_classes)
            # online setting
            self.online_model = r'D:\Solution\code\smic\automatic_defect_classification_server\service\weights\smic\back_model.pth'
        elif args.dataset == 'Front_M47':
            self.dataset = args.dataset
            # 明场
            self.train_path_list = [r'D:\Solution\datas\Front_M47', r'D:\Solution\datas\FrontDark_M47']
            self.val_path_list = [r'D:\Solution\datas\Front_M47_val', r'D:\Solution\datas\FrontDark_M47_val']

            self.anno_root = './datasets/Front_M47'
            self.multi_classes = {'false': "0", 'bubble': "1", 'burr': "2", "other": "3", 'particle': "4",
                                  'voiding': "5", "wrinkle": "6", "discolor": "7", "scratch": 8}
            self.numcls = len(self.multi_classes)
            # online setting
            self.online_model = r'D:\Solution\code\smic\automatic_defect_classification_server\service\weights\smic\front_model.pth'
        else:
            raise Exception('dataset {} not defined ???'.format(args.dataset))

        if only_dataset:
            return

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

    def load_txt(self):
        if 'train' in self.get_list:
             self.train_anno = pd.read_csv(os.path.join(self.anno_root, 'train.txt'),\
                                           sep=", ",\
                                           header=None,\
                                           names=['ImageName', 'label'],
                                           engine='python')

        if 'val' in self.get_list:
            self.val_anno = pd.read_csv(os.path.join(self.anno_root, 'val.txt'),\
                                           sep=", ",\
                                           header=None,\
                                           names=['ImageName', 'label'],
                                           engine='python')

        if 'test' in self.get_list:
            self.test_anno = pd.read_csv(os.path.join(self.anno_root, 'test.txt'),\
                                           sep=", ",\
                                           header=None,\
                                           names=['ImageName', 'label'],
                                           engine='python')
