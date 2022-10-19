#coding=utf8
from __future__ import print_function, division

import os,time,datetime
import numpy as np
from math import ceil
import datetime

import torch
from torch import nn
from torch.autograd import Variable
#from torchvision.utils import make_grid, save_image

from utils.utils import LossRecord, clip_gradient
from models.focal_loss import FocalLoss
from utils.eval_model import eval_turn
from utils.Asoftmax_loss import AngleLoss

import pdb

def dt():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")


def train(Config,
          model,
          epoch_num,
          start_epoch,
          optimizer,
          exp_lr_scheduler,
          data_loader,
          save_dir,
          data_size=448,
          savepoint=500,
          checkpoint=1000,
          log_server=None
          ):
    # savepoint: save without evalution
    # checkpoint: save with evaluation

    step = 0
    eval_train_flag = False
    rec_loss = []
    checkpoint_list = []

    train_batch_size = data_loader['train'].batch_size
    train_epoch_step = data_loader['train'].__len__()
    train_loss_recorder = LossRecord(train_batch_size)

    if savepoint > train_epoch_step:
        savepoint = 1*train_epoch_step
        checkpoint = savepoint

    date_suffix = dt()
    log_file = open(os.path.join(Config.log_folder, 'formal_log_r50_dcl_%s_%s.log'%(str(data_size), date_suffix)), 'a')

    add_loss = nn.L1Loss()
    get_ce_loss = nn.CrossEntropyLoss()
    get_focal_loss = FocalLoss()
    get_angle_loss = AngleLoss()

    val_best_acc, val_best_epoch = -1, -1
    for epoch in range(start_epoch,epoch_num-1):
        exp_lr_scheduler.step(epoch)
        model.train(True)

        save_grad = []
        for batch_cnt, data in enumerate(data_loader['train']):
            step += 1
            loss = 0
            model.train(True)
            if Config.use_backbone:
                inputs, labels, img_names = data
                inputs = Variable(inputs.cuda())
                labels = Variable(torch.from_numpy(np.array(labels)).cuda())

            if Config.use_dcl:
                inputs, labels, labels_swap, swap_law, img_names = data

                inputs = Variable(inputs.cuda())
                # labels = Variable(torch.from_numpy(np.array(labels)).cuda())
                # labels_swap = Variable(torch.from_numpy(np.array(labels_swap)).cuda())
                # 有些pytorch版本生成的labels和labels_swap是int32类型的，需要转换到int64
                labels = torch.from_numpy(np.array(labels))
                if labels.dtype != torch.int64:
                    labels = labels.type(torch.long)
                labels = Variable(labels.cuda())
                labels_swap = torch.from_numpy(np.array(labels_swap))
                if labels_swap.dtype != torch.int64:
                    labels_swap = labels_swap.type(torch.long)
                labels_swap = Variable(labels_swap.cuda())

                swap_law = Variable(torch.from_numpy(np.array(swap_law)).float().cuda())

            optimizer.zero_grad()

            if inputs.size(0) < 2*train_batch_size:
                outputs = model(inputs, inputs[0:-1:2])
            else:
                outputs = model(inputs, None)

            # labels = labels.type(torch.int64)
            if Config.use_focal_loss:
                ce_loss = get_focal_loss(outputs[0], labels)
            else:
                ce_loss = get_ce_loss(outputs[0], labels)

            if Config.use_Asoftmax:
                fetch_batch = labels.size(0)
                if batch_cnt % (train_epoch_step // 5) == 0:
                    angle_loss = get_angle_loss(outputs[3], labels[0:fetch_batch:2], decay=0.9)
                else:
                    angle_loss = get_angle_loss(outputs[3], labels[0:fetch_batch:2])
                loss += angle_loss

            loss += ce_loss

            alpha_ = 1
            beta_ = 1
            gamma_ = 0.01 if Config.dataset == 'STCAR' or Config.dataset == 'AIR' else 1
            # labels_swap = labels_swap.type(torch.int64)
            if Config.use_dcl:
                swap_loss = get_ce_loss(outputs[1], labels_swap) * beta_
                loss += swap_loss
                law_loss = add_loss(outputs[2], swap_law) * gamma_
                loss += law_loss

            loss.backward()
            torch.cuda.synchronize()

            optimizer.step()
            torch.cuda.synchronize()

            if Config.use_dcl:
                print_string_1 = 'step: {:-8d} / {:d} loss=ce_loss+swap_loss+law_loss: {:6.4f} = {:6.4f} + {:6.4f} + {:6.4f} '.format(step, train_epoch_step, loss.detach().item(), ce_loss.detach().item(), swap_loss.detach().item(), law_loss.detach().item())
                print(print_string_1, flush=True)
                _ = log_server.logging(print_string_1) if log_server else 1
            if Config.use_backbone:
                print_string_2 = 'step: {:-8d} / {:d} loss=ce_loss+swap_loss+law_loss: {:6.4f} = {:6.4f} '.format(step, train_epoch_step, loss.detach().item(), ce_loss.detach().item())
                print(print_string_2, flush=True)
                _ = log_server.logging(print_string_2) if log_server else 1
            rec_loss.append(loss.detach().item())

            train_loss_recorder.update(loss.detach().item())

            # evaluation & save
            if step % checkpoint == 0:
                rec_loss = []
                print(32*'-', flush=True)
                _ = log_server.logging(32*'-') if log_server else 1
                print_string_3 = 'step: {:d} / {:d} global_step: {:8.2f} train_epoch: {:04d} rec_train_loss: {:6.4f}'.format(step, train_epoch_step, 1.0*step/train_epoch_step, epoch, train_loss_recorder.get_val())
                print(print_string_3, flush=True)
                _ = log_server.logging(print_string_3) if log_server else 1
                print('current lr:%s' % exp_lr_scheduler.get_lr(), flush=True)
                _ = log_server.logging('current lr:%s' % exp_lr_scheduler.get_lr()) if log_server else 1
                if eval_train_flag:
                    trainval_acc1, trainval_acc2, trainval_acc3 = eval_turn(Config, model, data_loader['trainval'], 'trainval', epoch, log_file)
                    if abs(trainval_acc1 - trainval_acc3) < 0.01:
                        eval_train_flag = False

                val_acc1, val_acc2, val_acc3 = eval_turn(Config, model, data_loader['val'], 'val', epoch, log_file)

                save_path = os.path.join(save_dir, 'weights_%d_%d_%.4f.pth' % (epoch, batch_cnt, val_acc1))
                torch.cuda.synchronize()
                torch.save(model.state_dict(), save_path)
                if val_acc1 > val_best_acc:
                    val_best_acc = val_acc1
                    val_best_epoch = epoch
                    best_weight_save_path = os.path.join(save_dir, 'best_weights_%d_%d_%.4f.pth' % (val_best_epoch, batch_cnt, val_best_acc))
                    torch.save(model.state_dict(), best_weight_save_path)
                    best_model_save_path = os.path.join(save_dir, 'best_model.pth')
                    torch.save(model.state_dict(), best_model_save_path)
                    print_string_4 = "save best weight to {} and {}".format(best_weight_save_path, best_model_save_path)
                    print(print_string_4)
                    _ = log_server.logging(print_string_4) if log_server else 1
                print_string_5 = 'saved model to {}, best_epoch:{}, best_acc:{}'.format(save_path, val_best_epoch, val_best_acc)
                print(print_string_5, flush=True)
                _ = log_server.logging(print_string_5) if log_server else 1
                torch.cuda.empty_cache()

            # save only
            elif step % savepoint == 0:
                train_loss_recorder.update(rec_loss)
                rec_loss = []
                save_path = os.path.join(save_dir, 'savepoint_weights-%d-%s.pth'%(step, dt()))

                checkpoint_list.append(save_path)
                if len(checkpoint_list) == 6:
                    os.remove(checkpoint_list[0])
                    del checkpoint_list[0]
                torch.save(model.state_dict(), save_path)
                torch.cuda.empty_cache()


    log_file.close()



