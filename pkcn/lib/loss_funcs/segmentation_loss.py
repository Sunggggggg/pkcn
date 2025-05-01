from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from loss_funcs.maps_loss import focal_loss
import json
import numpy as np
from config import args
import pickle

smpl_face = pickle.load(open('model_data/parameters/SMPL_NEUTRAL.pkl','rb'), encoding='latin1')['f'].astype(np.int)


seg_name = ["hips", "background", "leftUpLeg", "rightUpLeg", "spine", "leftLeg", "rightLeg", "spine1", "leftFoot", "rightFoot", "spine2",
            "leftToeBase", "rightToeBase", "neck", "leftShoulder", "rightShoulder", "head", "leftArm", "rightArm",
            "leftForeArm", "rightForeArm", "leftHand", "rightHand", "leftHandIndex1", "rightHandIndex1"]

with open("smpl_vert_segmentation.json", "r") as f:
    part_segm = json.load(f)

non_parameter = -896.86181640625

def make_seg_gt(outputs, img_size=64):

    maps = outputs["verts_camed"].cpu() * (img_size//2) + (img_size//2)
    x = maps[:,:, 0].int()
    y = maps[:,:, 1].int()
    xy = (y * img_size + x).cpu().numpy()
    batch = xy.shape[0]
    mask = torch.ones((batch, len(seg_name), img_size * img_size))
    loss_flag = torch.ones((batch, len(seg_name))).bool()

    for idx, name in enumerate(seg_name):
        if name == 'background':
            xy_idx = xy
        else:
            seg_idx = part_segm[name]
            xy_idx = xy[:, seg_idx]

        for i, label in enumerate(xy_idx):
            label = label[label < img_size * img_size]
            label = label[label>-1]
            if np.any(label):
                if name == 'background':
                    mask[i, idx] = 1.
                    mask[i, idx, label] = 0.
                else:
                    mask[i, idx, label] = 1.
            else:
                loss_flag[i, idx] = False
    return mask.reshape(batch, len(seg_name), img_size, img_size).float(), loss_flag


def make_xy(verts, cam_trans):
    cam_trans[:,-1] += 0.3
    verts = verts + cam_trans.unsqueeze(1)
    X, Y, Z = verts[:, :, 0], verts[:, :, 1], verts[:, :, 2],
    fx, px, fy, py = 64, 32, 64, 32
    x = fx * X / Z + px
    y = fy * Y / Z + py
    return x.int(), y.int()

def _calc_segmentation_loss(outputs, bce):
    if outputs["segm_maps"].shape[0]==0:
        print("ERROR")
        return {'Segmentation': 0}
    epoch = outputs["meta_data"]["epoch"][0][0]
    if epoch>args().segloss_epoch:
        return {'Segmentation': 0}

    segmap = outputs["segm_maps"] # batch x 25 x 64 x 64
    device = segmap.device

    segm_gt, flag = make_seg_gt(outputs) # batch x 25 x 64 x 64

    loss = dice_loss(segmap[flag].float(), segm_gt[flag].float().to(device))
    loss_dict = {'Segmentation': loss}
    return loss_dict

def dice_loss(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    # binary cross entropy loss
    bce = F.binary_cross_entropy(pred, target, reduction='mean')

    # dice coefficient
    intersection = (pred * target).sum(dim=(1,2))
    union = pred.sum(dim=(1,2)) + target.sum(dim=(1,2))
    dice = 2.0 * (intersection + smooth) / (union + 2 * smooth)

    # dice loss
    dice_loss = 1.0 - dice
    # total loss
    loss = bce + dice_loss.mean()

    return loss