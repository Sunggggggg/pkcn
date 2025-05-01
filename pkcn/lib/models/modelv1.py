from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from models.base import Base
from models.CoordConv import get_coord_maps
from models.basic_modules import BasicBlock,Bottleneck
from models.smpl_wrapper import SMPLWrapper
from models.ktd import KTD

import config
from config import args
from loss_funcs import Loss
from maps_utils.result_parser import ResultParser

if args().model_precision=='fp16':
    from torch.cuda.amp import autocast

BN_MOMENTUM = 0.1

class ROMP(Base):
    def __init__(self, backbone=None,**kwargs):
        super(ROMP, self).__init__()
        print('=========== Model infomation ===========')
        print(f'Backbone : {args().backbone}')
        print(f'# of viewpoint ')
        
        self.backbone = backbone
        self._result_parser = ResultParser()
        self.params_map_parser = SMPLWrapper()
        self._build_encoder()
        for param in self.parameters():
            param.requires_grad = False 
        
        self._build_decoder()
        if args().model_return_loss:
            self._calc_loss = Loss()

        if not args().fine_tune and not args().eval:
            self.init_weights()
            self.backbone.load_pretrain_params()

    def matching_forward(self, meta_data, **cfg):
        if args().model_precision=='fp16':
            with autocast():
                outputs = self.feed_forward(meta_data)
                outputs, meta_data = self._result_parser.matching_forward(outputs, meta_data, cfg)

                outputs = self.tail_forward(outputs)
                outputs = self.params_map_parser(outputs,meta_data)

        else:
            outputs = self.feed_forward(meta_data)
            outputs, meta_data = self._result_parser.matching_forward(outputs, meta_data, cfg)

            outputs = self.tail_forward(outputs)
            outputs = self.params_map_parser(outputs,meta_data)

        outputs['meta_data'] = meta_data
        if cfg['calc_loss']:
            outputs.update(self._calc_loss(outputs))
        return outputs

    @torch.no_grad()
    def parsing_forward(self, meta_data, **cfg):
        if args().model_precision=='fp16':
            with autocast():
                outputs = self.feed_forward(meta_data)
                outputs, meta_data = self._result_parser.parsing_forward(outputs, meta_data, cfg)

                outputs = self.tail_forward(outputs)
                outputs = self.params_map_parser(outputs,meta_data)
        else:
            outputs = self.feed_forward(meta_data)
            outputs, meta_data = self._result_parser.parsing_forward(outputs, meta_data, cfg)

            outputs = self.tail_forward(outputs)
            outputs = self.params_map_parser(outputs,meta_data)

        outputs['meta_data'] = meta_data
        return outputs

    def feed_forward(self, meta_data):
        x = self.backbone(meta_data['image'].contiguous().cuda())   # [B, 32, 128, 128]
        outputs = self.head_forward(x)
        return outputs

    def head_forward(self,x):
        """
        output 
            params_maps torch.Size([1, 400, 64, 64])
            center_map torch.Size([1, 1, 64, 64])
            segmentation_maps torch.Size([1, 128, 64, 64])
            feature_maps torch.Size([1, 128, 64, 64])
        """
        x = torch.cat((x, self.coordmaps.to(x.device).repeat(x.shape[0],1,1,1)), 1)
        
        params_maps = self.final_layers[1](x)
        center_maps = self.final_layers[2](x)
        segment_maps = self.final_layers[3](x)
        feature_maps = self.final_layers[4](x)

        output = {'params_maps':params_maps.float(), 'center_map':center_maps.float(), 
                  'segmentation_maps':segment_maps.float(), 'feature_maps':feature_maps.float()} #, 'kp_ae_maps':kp_heatmap_ae.float()
        return output

    def tail_forward(self,outputs):
        """ 
        outputs 
            params_maps torch.Size([1, 400, 64, 64])
            center_map torch.Size([1, 1, 64, 64])
            segmentation_maps torch.Size([1, 128, 64, 64])
            feature_maps torch.Size([1, 128, 64, 64])
            detection_flag torch.Size([2])
            params_pred torch.Size([2, 400])
            centers_pred torch.Size([2, 2])
            centers_conf torch.Size([2, 1])
            reorganize_idx torch.Size([2])
        """
        num_channels = self.output_cfg["NUM_CHANNELS"]  # 16
        num_joints = self.head_cfg["NUM_JOINTS"]        # 24
        reorganize_idx = outputs["reorganize_idx"]

        if torch.cuda.device_count()>1:
            front_idx = reorganize_idx.min()
            reorganize_idx = outputs["reorganize_idx"] - front_idx

        smpl_segmenation = outputs["segmentation_maps"][reorganize_idx].flatten(2)  # [N, 128, 4096] 
        pose_features = outputs["feature_maps"][reorganize_idx].flatten(2)          # [N, 128, 4096] 
        cam_shape_features = self.pose_shape_layer(outputs["feature_maps"])[reorganize_idx].flatten(2) # [N, 128, 4096]  

        params_pred = outputs["params_pred"].reshape(-1,num_joints+1,num_channels)  # [N, 24+1, 16]
        params_pred = self.idx_mlp(params_pred)                                     # [N, 25, 128]

        segm_maps = torch.bmm(params_pred, smpl_segmenation)                        # [N, 25, 4096]
        attn_maps = F.softmax(segm_maps[:,1:,:], dim=-1)

        pose_maps = torch.bmm(attn_maps, pose_features.transpose(1,2))                          # [N, 24, 128]
        cam_shape_maps = torch.bmm(attn_maps, cam_shape_features.transpose(1, 2)).flatten(1)    # [N, 24*128]
        pose_params = self.pose_mlp(pose_maps).flatten(1)                                       # [N, 24, 6]

        beta = self.shape_mlp(cam_shape_maps)   # [N, 10]
        cam = self.cam_mlp(cam_shape_maps)      # [N, 3]
        cam[:,0] = torch.pow(1.1,cam[:,0])

        params_pred = torch.cat([cam, pose_params, beta], 1)
        outputs["params_pred"] = params_pred
        outputs["segm_maps"] = segm_maps.reshape(-1,num_joints+1,64,64)
        
        return outputs

    def _build_encoder(self):
        self.outmap_size = args().centermap_size
        num_channels, num_joints = 16, 24

        self.head_cfg = {'NUM_HEADS': 1, 'NUM_CHANNELS': 64, 'NUM_BASIC_BLOCKS': args().head_block_num, "NUM_JOINTS":num_joints}
        self.output_cfg = {'NUM_PARAM_MAP': (num_joints+1) * num_channels, 'NUM_CENTER_MAP': 1,
                           'NUM_CHANNELS': num_channels}

        self.final_layers = self._make_final_layers(self.backbone.backbone_channels)
        self.coordmaps = get_coord_maps(128)

    def _build_decoder(self):
        num_channels, num_joints = 16, 24
        self.idx_mlp = nn.Sequential(nn.Linear(num_channels,num_channels*4),
                                     nn.BatchNorm1d(num_joints+1),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(num_channels*4,num_channels*8),
                                     nn.BatchNorm1d(num_joints+1),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(num_channels * 8, num_channels * 8)
                                     )

        self.pose_mlp = KTD(num_channels * 8)

        self.shape_mlp = nn.Linear(num_joints * num_channels * 2, 10)
        self.cam_mlp = nn.Linear(num_joints * num_channels * 2, 3)

        self.pose_shape_layer = nn.Sequential(
            nn.Conv2d(num_channels * 8, num_channels * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels * 8, num_channels * 2, kernel_size=1, stride=1, padding=0),
        )

    def _make_final_layers(self, input_channels):
        final_layers = []
        input_channels += 2

        final_layers.append(None)
        final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_PARAM_MAP']))   # 64
        final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_CENTER_MAP']))  # 1
        final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_CHANNELS'] * 8))# 400
        final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_CHANNELS'] * 8))# 400

        return nn.ModuleList(final_layers)

    def _make_head_layers(self, input_channels, output_channels):
        head_layers = []
        num_channels = self.head_cfg['NUM_CHANNELS']

        kernel_sizes, strides, paddings = self._get_trans_cfg()
        for kernel_size, padding, stride in zip(kernel_sizes, paddings, strides):
            head_layers.append(nn.Sequential(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding),
                    nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)))
        for i in range(self.head_cfg['NUM_HEADS']):
            layers = []
            for _ in range(self.head_cfg['NUM_BASIC_BLOCKS']):
                layers.append(nn.Sequential(BasicBlock(num_channels, num_channels)))
            head_layers.append(nn.Sequential(*layers))

        head_layers.append(nn.Conv2d(in_channels=num_channels,out_channels=output_channels,\
            kernel_size=1,stride=1,padding=0))

        return nn.Sequential(*head_layers)

    def _get_trans_cfg(self):
        if self.outmap_size == 32:
            kernel_sizes = [3,3]
            paddings = [1,1]
            strides = [2,2]
        elif self.outmap_size == 64:
            kernel_sizes = [3]
            paddings = [1]
            strides = [2]
        elif self.outmap_size == 128:
            kernel_sizes = [3]
            paddings = [1]
            strides = [1]

        return kernel_sizes, strides, paddings
