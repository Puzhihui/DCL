import os
import pandas as pd
import torch
import platform

# from transforms import transforms
from cvtorchvision import cvtransforms

# pretrained model checkpoints
pretrained_model = {'resnet50' : './models/pretrained/resnet50-19c8e357.pth',
                    "se_resnext101_32x4d": "./models/pretrained/se_resnext101_32x4d-3b2fe3d8.pth"}
customize_model = ['efficientnet-b4', 'efficientnet-b1', 'efficientnet-b3']

# transforms dict
def load_data_transformers(resize_reso=512, crop_reso=448, swap_num=[7, 7]):
    center_resize = 600
    Normalize = cvtransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    data_transforms = {
       	'swap': cvtransforms.Compose([
            cvtransforms.Randomswap((swap_num[0], swap_num[1])),
        ]),
        'adc_aug': cvtransforms.Compose([             # 先center crop 800 再center crop目标尺寸
            cvtransforms.CenterCrop((800, 800)),
            cvtransforms.RandomVerticalFlip(0.5),
            cvtransforms.RandomHorizontalFlip(0.5),
            # # transforms.CenterCrop((600, 600)),
            # # transforms.ColorJitter(0.2, 0.1, 0.1, 0.01),
            # # transforms.RandomRotation(degrees=15),
            cvtransforms.CenterCrop((crop_reso, crop_reso)),
        ]),
        'adc_resize_aug': cvtransforms.Compose([        # train 先center crop 832  再resize目标尺寸
            cvtransforms.CenterCrop((832, 832)),
            cvtransforms.RandomVerticalFlip(0.5),
            cvtransforms.RandomHorizontalFlip(0.5),
            # # transforms.ColorJitter(0.2, 0.1, 0.1, 0.01),
            # # transforms.RandomRotation(degrees=15),
            cvtransforms.Resize((crop_reso, crop_reso)),
        ]),
        # 'adc_oi_center_aug': cvtransforms.Compose([        # train 原图center crop
        #     cvtransforms.RandomVerticalFlip(0.5),
        #     cvtransforms.RandomHorizontalFlip(0.5),
        #     # transforms.ColorJitter(0.2, 0.1, 0.1, 0.01),
        #     # transforms.RandomRotation(degrees=15),
        #     cvtransforms.CenterCrop((crop_reso, crop_reso)),
        # ]),
        'adc_oi_resize_aug': cvtransforms.Compose([         # train 原图resize
            cvtransforms.RandomVerticalFlip(0.5),
            cvtransforms.RandomHorizontalFlip(0.5),
            # # transforms.ColorJitter(0.2, 0.1, 0.1, 0.01),
            # # transforms.RandomRotation(degrees=15),
            cvtransforms.Resize((crop_reso, crop_reso)),
        ]),
        'adc_train_totensor': cvtransforms.Compose([
            cvtransforms.ToTensor(),
            cvtransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'adc_val_totensor': cvtransforms.Compose([             # val 原图center crop
            cvtransforms.CenterCrop((crop_reso, crop_reso)),
            cvtransforms.ToTensor(),
            cvtransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'adc_val_oi_resize_totensor': cvtransforms.Compose([    # val 原图resize
            cvtransforms.Resize((crop_reso, crop_reso)),
            cvtransforms.ToTensor(),
            cvtransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'adc_val_resize_totensor': cvtransforms.Compose([       # val 先centercrop 再resize
            cvtransforms.CenterCrop((832, 832)),
            cvtransforms.Resize((crop_reso, crop_reso)),
            cvtransforms.ToTensor(),
            cvtransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'None': None,
    }
    return data_transforms


class LoadConfig(object):
    def __init__(self, args, version):
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

        # ===================jssi aoi 终检===================
        # if args.dataset == 'jssi_aoi_center':
        #     self.dataset = args.dataset
        #     self.rawdata_root = '/data3/pzh/data/jssi/aoi'
        #     self.anno_root = './datasets/jssi_aoi/jssi_aoi_center'
        #     self.numcls = 2
        # elif args.dataset == 'jssi_aoi_resize':
        #     self.dataset = args.dataset
        #     self.rawdata_root = '/data3/pzh/data/jssi/aoi'
        #     self.anno_root = './datasets/jssi_aoi/jssi_aoi_resize'
        #     self.numcls = 2
        if args.dataset == 'jssi-Bumpping_aoi':
            self.dataset = args.dataset
            self.train_path_list = [r'D:\Solution\datas\jssi-Bumpping_aoi'] if platform.system() == 'Windows' else ["/data1/pzh/data/jssi/aoi/train/"]
            self.val_path_list = [r'D:\Solution\datas\jssi-Bumpping_aoi_val'] if platform.system() == 'Windows' else ["/data1/pzh/data/jssi/aoi/val/"]

            self.anno_root = './datasets/jssi-Bumpping_aoi'
            self.multi_classes = {'Bad': "0", 'Good': "1"}
            self.numcls = len(self.multi_classes)

            # online setting
            self.online_model = r'D:\Solution\code\automatic_defect_classification_server\service\weights\jssi-bumpping_aoi\dcl_2_aoi.pth'
        elif args.dataset == 'jssi-Bumpping_aoi_multi':
            self.dataset = args.dataset
            self.train_path_list = [r'D:\Solution\datas\jssi-Bumpping_aoi_multi']
            self.val_path_list = [r'D:\Solution\datas\jssi-Bumpping_aoi_multi_val']

            self.anno_root = './datasets/jssi-Bumpping_aoi_multi'
            self.multi_classes = {'Bad': "0", 'Good': "1"}
            self.numcls = len(self.multi_classes)
            # online setting
            self.online_model = r'D:\Solution\code\automatic_defect_classification_server\service\weights\jssi-bumpping_aoi\dcl_multi_aoi.pth'

        # ===================jssi photo 过程检===================
        # elif args.dataset == 'jssi_photo_center':
        #     self.dataset = args.dataset
        #     self.rawdata_root = '/data3/pzh/data/jssi/photo'
        #     self.anno_root = './datasets/jssi_photo/jssi_photo_center'
        #     self.numcls = 2
        # elif args.dataset == 'jssi_photo_resize':
        #     self.dataset = args.dataset
        #     self.rawdata_root = '/data3/pzh/data/jssi/photo'
        #     self.anno_root = './datasets/jssi_photo/jssi_photo_resize'
        #     self.numcls = 2
        elif args.dataset == 'jssi-Bumpping_photo':
            self.dataset = args.dataset
            # self.rawdata_root = '/data3/pzh/data/jssi/photo'
            self.train_path_list = [r'D:\Solution\datas\jssi-Bumpping_photo'] if platform.system() == 'Windows' else ["/data1/pzh/data/jssi/photo/train/"]
            self.val_path_list = [r'D:\Solution\datas\jssi-Bumpping_photo_val'] if platform.system() == 'Windows' else ["/data1/pzh/data/jssi/photo/val/"]

            self.anno_root = './datasets/jssi-Bumpping_photo'
            self.multi_classes = {'Bad': "0", 'Good': "1"}
            self.numcls = len(self.multi_classes)

            # online setting
            self.online_model = r'D:\Solution\code\automatic_defect_classification_server\service\weights\jssi-bumpping_photo\dcl_2_photo.pth'
        elif args.dataset == 'jssi-Bumpping_photo_multi':
            self.dataset = args.datasetself.train_path_list = [r'D:\Solution\datas\jssi-Bumpping_photo_multi']
            self.val_path_list = [r'D:\Solution\datas\jssi-Bumpping_photo_multi_val']

            self.anno_root = './datasets/jssi-Bumpping_photo_multi'
            self.multi_classes = {'Bad': "0", 'Good': "1"}
            self.numcls = len(self.multi_classes)
            # online setting
            self.online_model = r'D:\Solution\code\automatic_defect_classification_server\service\weights\jssi-bumpping_photo\dcl_multi_photo.pth'
        elif args.dataset == 'jssi_dirty_recipe':
            self.dataset = args.dataset
            self.rawdata_root = '/data3/pzh/data/jssi/jssi_dirty_recipe'
            self.anno_root = './datasets/jssi_dirty_recipe'
            self.numcls = 2
        # ===================ht==================================
        elif args.dataset == 'ht_less500_resize':
            self.dataset = args.dataset
            self.rawdata_root = '/data4/exp_data/'
            self.anno_root = './datasets/ht_fi/ht_less500_resize'
            self.numcls = 2
        elif args.dataset == 'ht_more500_center_resize':
            self.dataset = args.dataset
            self.rawdata_root = '/data4/exp_data/'
            self.anno_root = './datasets/ht_more500_center_resize'
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
        # ===================smic om==================================
        elif args.dataset == 'smic_om_3':
            self.dataset = args.dataset
            self.rawdata_root = r'D:\Solution\datas\smic_om_3' # /data3/pzh/data/smic/smic_om_3
            self.anno_root = './datasets/smic_om_3'
            self.numcls = 3
        elif args.dataset == "sic_eight":
            self.dataset = args.dataset
            self.rawdata_root = '/data3/pzh/data/sic_eight'
            self.anno_root = './datasets/sic_eight'
            self.numcls = 8
        else:
            raise Exception('dataset {} not defined ???'.format(args.dataset))

        # annotation file organized as :
        # path/image_name cls_num\n


        self.swap_num = args.swap_num

        self.save_dir = './net_model'
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.backbone = args.backbone

        self.use_dcl = True
        self.use_sagan = False
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
