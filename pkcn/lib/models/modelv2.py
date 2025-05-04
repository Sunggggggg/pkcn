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
from models.basic_modules import BasicBlock
from models.smpl_wrapper import SMPLWrapper
from models.ktd import KTD
from models.transformer import SVTransformer

import config
from config import args
from loss_funcs import Loss
from maps_utils.result_parser import ResultParser
from utils.mapping_utils import reassign_ids

if args().model_precision=='fp16':
    from torch.cuda.amp import autocast

BN_MOMENTUM = 0.1

class ROMP(Base):
    def __init__(self, backbone=None,**kwargs):
        super(ROMP, self).__init__()
        print('=========== Model infomation ===========')
        print(f'Backbone : {args().backbone}')
        print(f'# of viewpoints {args().num_views}')
        print(f'Use mask info. : {args().use_mask}')
        
        self.num_views = args().num_views
        self.backbone = backbone
        self._result_parser = ResultParser()
        self.params_map_parser = SMPLWrapper()
        self._build_encoder()
        for param in self.parameters():
            param.requires_grad = False 
        
        self._build_decoder()
        self._build_multiview_module()
        if args().model_return_loss:
            self._calc_loss = Loss()

        if not args().fine_tune and not args().eval:
            self.init_weights()
            self.backbone.load_pretrain_params()
            
        self.view_keys = ['image', 'person_centers', 'centermap', 'offsets', 
                          'all_person_detected_mask', 'full_kp2d', 'valid_masks', 
                          'kp_3d', 'params', 'global_params', 'image_org', 'subject_ids', 'rot_flip',
                          'mask']
        self.meta_keys = ['batch_ids', 'epoch', 'iter_idx']

    def matching_forward(self, meta_data, **cfg):
        ### Model forward ###
        def process_view(view):
            view_batch = {k : v[:, view] for k, v in meta_data.items() if k in self.view_keys}
            meta_batch = {k : v for k, v in meta_data.items() if k in self.meta_keys}
            mini_batch = view_batch | meta_batch
            
            if args().model_precision == 'fp16':
                with autocast():
                    outputs = self.feed_forward(mini_batch)
                    outputs, mini_batch = self._result_parser.matching_forward(outputs, mini_batch, cfg)
            
            else:
                outputs = self.feed_forward(mini_batch)
                outputs, mini_batch = self._result_parser.matching_forward(outputs, mini_batch, cfg)
            
            outputs['meta_data'] = mini_batch
            return outputs
            
        outputs_list = [process_view(view) for view in range(self.num_views)]
        outputs_list = self.multiview_forward(outputs_list)

        return outputs_list

    @torch.no_grad()
    def parsing_forward(self, meta_data, **cfg):
        ### Model forward ###
        def process_view(view):
            view_batch = {k : v[:, view] for k, v in meta_data.items() if k in self.view_keys}
            meta_batch = {k : v for k, v in meta_data.items() if k in self.meta_keys}
            mini_batch = view_batch | meta_batch
            
            if args().model_precision == 'fp16':
                with autocast():
                    outputs = self.feed_forward(mini_batch)
                    outputs, mini_batch = self._result_parser.parsing_forward(outputs, mini_batch, cfg)
            else:
                outputs = self.feed_forward(mini_batch)
                outputs, mini_batch = self._result_parser.parsing_forward(outputs, mini_batch, cfg)
            
            outputs['meta_data'] = mini_batch
            return outputs
            
        outputs_list = [process_view(view) for view in range(self.num_views)]
        outputs_list = self.multiview_matching_forward(outputs_list)

        return outputs_list

    def feed_forward(self, meta_data):
        img = meta_data['image'].contiguous().cuda()
        mask = meta_data['mask'].contiguous().cuda()

        x = self.backbone(img)   # [B, 32, 128, 128]
        outputs = self.head_forward(x, img, mask)
        return outputs

    def head_forward(self, x, img, mask):
        """
        x       : [B, 32, 128, 128]
        img     : => [B, 3, 512, 512]
        mask    : => [B, 1, 512, 512]
        """
        x = torch.cat((x, self.coordmaps.to(x.device).repeat(x.shape[0],1,1,1)), 1)
        img = img.permute(0, 3, 1, 2)   # [B, 3, H, W]
        mask = mask.permute(0, 3, 1, 2) # [B, 1, H, W]
        
        params_maps = self.final_layers[1](x)
        center_maps = self.final_layers[2](x)
        segment_maps = self.final_layers[3](x)
        feature_maps = self.final_layers[4](x)
        mask_maps = self.final_layers[5](mask)

        offset_x = torch.cat([img, mask], dim=1)                # [B, 4, 512, 512]
        offset_maps = self.final_layers[6](offset_x) + 0.5      # [B, 2, 256]

        params_maps += mask_maps
        
        output = {'params_maps':params_maps.float(),        # [B, 400, cm, cm]
                  'center_map':center_maps.float(),         # [B, 1, cm, cm]
                  'segmentation_maps':segment_maps.float(), # [B, 128, cm, cm]
                  'feature_maps':feature_maps.float(),      # [B, 128, cm, cm]
                  'mask_maps': mask_maps.float(),           # [B, 1, cm, cm]
                  'offset_maps': offset_maps.float()        # [B, 2, cm, cm]
                  }
        
        return output

    def tail_forward(self,outputs, view):
        """ Handing only single person
        outputs 
            params_maps torch.Size([B, 400, 64, 64])
            center_map torch.Size([B, 1, 64, 64])
            segmentation_maps torch.Size([B, 128, 64, 64])
            feature_maps torch.Size([B, 128, 64, 64])

            # Detection
            detection_flag torch.Size([2])
            params_pred torch.Size([2, 400])
            centers_pred torch.Size([2, 2])
            centers_conf torch.Size([2, 1])
            reorganize_idx torch.Size([2])
        """
        num_channels = self.output_cfg["NUM_CHANNELS"]  # 16
        num_joints = self.head_cfg["NUM_JOINTS"]        # 24
        reorganize_idx = outputs["reorganize_idx"]      # batch 기준으로 사람이 있는 것

        if torch.cuda.device_count()>1:
            front_idx = reorganize_idx.min()
            reorganize_idx = outputs["reorganize_idx"] - front_idx

        smpl_segmenation = outputs["segmentation_maps"][reorganize_idx].flatten(2)  # [1, 128, 4096]
        pose_features = outputs["feature_maps"][reorganize_idx].flatten(2) 
        cam_shape_features = self.pose_shape_layer(outputs["feature_maps"])[reorganize_idx].flatten(2)

        center_feature = outputs["params_pred"]
        params_pred = outputs["params_pred"].reshape(-1,num_joints+1,num_channels)  # [N, 24+1, 16] 사람 수만큼
        params_pred = self.idx_mlp(params_pred)                                     # [N, 25, 128]

        segm_maps = torch.bmm(params_pred, smpl_segmenation)                        # [N, 25, 4096]
        attn_maps = F.softmax(segm_maps[:,1:,:], dim=-1)

        pose_maps = torch.bmm(attn_maps, pose_features.transpose(1,2))                          # [N, 24, 128]
        cam_shape_maps = torch.bmm(attn_maps, cam_shape_features.transpose(1, 2)).flatten(1)    # [N, 24, 128] @ [N, 128, 24]
        pose_params = self.pose_mlp(pose_maps).flatten(1)                                       # [N, 24, 6]

        beta = self.shape_mlp(cam_shape_maps)   # [N, 10]
        cam = self.cam_mlp(cam_shape_maps)      # [N, 3]
        cam[:,0] = torch.pow(1.1,cam[:,0])

        params_pred = torch.cat([cam, pose_params, beta], 1)
        
        ### Output ###
        outputs[f"feat_{view}"] = pose_maps
        outputs["params_pred"] = params_pred
        outputs["segm_maps"] = segm_maps.reshape(-1,num_joints+1,args().centermap_size, args().centermap_size)  # [N, 25, 64, 64]
        
        outputs["center_feature"] = center_feature
        return outputs
    
    def multiview_forward(self, outputs_list, eval=False):
        if eval : 
            batch_size = args().val_batch_size
        else : 
            batch_size = args().batch_size

        outputs_per_view_list, feat_list = [], []
        for view, outputs in enumerate(outputs_list) :
            meta_data = outputs['meta_data']

            outputs_per_view = self.tail_forward(outputs, view)
            outputs_per_view = self.params_map_parser(outputs_per_view, meta_data)
            per_view_feat = outputs_per_view[f"feat_{view}"]        # [~B, 24, 128]

            if batch_size == per_view_feat.shape[0] :
                outputs_per_view_list.append(outputs_per_view)
                feat_list.append(per_view_feat)

        if len(feat_list) == 0 :
            print("None")
            return None
        
        multi_view_feat = torch.stack(feat_list, dim=1)             # [B, num_views, 24, 128]
        multi_view_feat = self.multiview_encoder(multi_view_feat)   # [B, 24, 128]
        
        pose_params = self.global_pose_mlp(multi_view_feat).flatten(1)  # [B, 24, 6]
        betas = self.global_shape_mlp(multi_view_feat.flatten(1))       # [B, 10]
        
        for idx, outputs in enumerate(outputs_per_view_list) :
            reorganize_idx = outputs["reorganize_idx"]
            if batch_size == len(reorganize_idx) :
                outputs_per_view_list[idx].update(self.params_map_parser.forward_refine(outputs['params'], pose_params, betas))

        return outputs_per_view_list
    
    @torch.no_grad()
    def multiview_matching_forward(self, outputs_list):
        max_person = args().max_person

        outputs_per_view_list = []
        for view, outputs in enumerate(outputs_list) :
            meta_data = outputs['meta_data']
            
            outputs_per_view = self.tail_forward(outputs, view)
            outputs_per_view = self.params_map_parser(outputs_per_view, meta_data)
            per_view_feat = outputs_per_view[f"feat_{view}"]        # [N, 24, 128] (N : 사람의 수)

            if 0 < per_view_feat.shape[0] <= max_person :
                outputs_per_view_list.append(outputs_per_view)
                
        ### Matching algo. ###
        appear_feat_list = []
        for outputs in outputs_per_view_list:
            appear_feat_list.append(outputs['center_feature'])   # [N, 400]

        # appear_feat = torch.stack(appear_feat_list, dim=0)
        ids = reassign_ids(appear_feat_list)
        outputs_per_view_list, multi_view_feat, ray_list = self.re_id_mapping(ids, outputs_per_view_list)
        multi_view_feat = self.multiview_encoder(multi_view_feat, ray_list)       # [N, 2, 24, 128]
        
        pose_params = self.global_pose_mlp(multi_view_feat).flatten(1)  # [B, 24, 6]
        betas = self.global_shape_mlp(multi_view_feat.flatten(1))       # [B, 10]
        
        for idx, outputs in enumerate(outputs_per_view_list) :
            reorganize_idx = outputs["reorganize_idx"]
            if 0 < reorganize_idx.shape[0] <= max_person :
                outputs_per_view_list[idx].update(self.params_map_parser.forward_refine(outputs['params'], pose_params, betas))
                
        return outputs_per_view_list, pose_params, betas

    def re_id_mapping(self, ids, outputs_per_view_list):
        mutiview_feat = []
        re_id_outputs_per_view_list = outputs_per_view_list.copy()

        for view_idx, (re_id, outputs) in enumerate(zip(ids, outputs_per_view_list)) :
            if re_id.tolist() != [-1, -1] :
                # re_id_outputs_per_view_list[view_idx]['center_feature'] = outputs['center_feature'].expand(args().max_person, 400)[re_id]
                # re_id_outputs_per_view_list[view_idx]['centers_pred'] = outputs['centers_pred'].expand(args().max_person, 2)[re_id]
                re_id_outputs_per_view_list[view_idx]['params_pred'] = outputs['params_pred'].expand(args().max_person, 3+6+138+10)[re_id]
                re_id_outputs_per_view_list[view_idx]['verts'] = outputs['verts'].expand(args().max_person, 6890, 3)[re_id]
                re_id_outputs_per_view_list[view_idx][f'feat_{view_idx}'] = outputs[f'feat_{view_idx}'].expand(args().max_person, 24, 128)[re_id]
                re_id_outputs_per_view_list[view_idx]['j3d'] = outputs['j3d'].expand(args().max_person, 54, 3)[re_id]
                re_id_outputs_per_view_list[view_idx]['joints_smpl24'] = outputs['joints_smpl24'].expand(args().max_person, 24, 3)[re_id]
                re_id_outputs_per_view_list[view_idx]['joints_h36m17'] = outputs['joints_h36m17'].expand(args().max_person, 17, 3)[re_id]
                re_id_outputs_per_view_list[view_idx]['verts_camed'] = outputs['verts_camed'].expand(args().max_person, 6890, 3)[re_id]
                re_id_outputs_per_view_list[view_idx]['pj2d'] = outputs['pj2d'].expand(args().max_person, 54, 2)[re_id]
                re_id_outputs_per_view_list[view_idx]['cam_trans'] = outputs['cam_trans'].expand(args().max_person, 3)[re_id]
                re_id_outputs_per_view_list[view_idx]['pj2d_org'] = outputs['pj2d_org'].expand(args().max_person, 54, 2)[re_id]

                re_id_outputs_per_view_list[view_idx]['params']['global_orient'] = outputs['params']['global_orient'].expand(args().max_person, 3)[re_id]
                re_id_outputs_per_view_list[view_idx]['params']['body_pose'] = outputs['params']['body_pose'].expand(args().max_person, 69)[re_id]
                re_id_outputs_per_view_list[view_idx]['params']['betas'] = outputs['params']['betas'].expand(args().max_person, 10)[re_id]
                re_id_outputs_per_view_list[view_idx]['params']['cam'] = outputs['params']['cam'].expand(args().max_person, 3)[re_id]
                
                re_id_outputs_per_view_list[view_idx]['ray_d'] = outputs['ray_d'].expand(args().max_person, 3)[re_id]
                re_id_outputs_per_view_list[view_idx]['attn_weight'] = outputs['attn_weight']   # [B, Cate, 256]
    
                mutiview_feat.append(outputs[f'feat_{view_idx}'].expand(args().max_person, 24, 128)[re_id])
                
        mutiview_feat = torch.stack(mutiview_feat, dim=1)   # [사람수, N, 24, 400]
        return re_id_outputs_per_view_list, mutiview_feat

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

        final_layers.append(None)   # 0
        final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_PARAM_MAP']))   # 1
        final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_CENTER_MAP']))  # 2
        final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_CHANNELS'] * 8))# 3
        final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_CHANNELS'] * 8))# 4
        
        final_layers.append(self._make_mask_layers(1, 1))                       # 5
        final_layers.append(self._make_mask_layers(4, 2))  # 6

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
    
    def _make_mask_layers(self, input_dim, output_dim):
        conv_block = nn.Sequential(
            # 512x512x1 → 256x256x32
            nn.Conv2d(input_dim, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # 256x256x32 → 128x128x64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 128x128x64 → 64x64x128
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # 64x64x128 → 64x64x400
            nn.Conv2d(32, output_dim, kernel_size=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
        )
        
        return conv_block

    def _build_multiview_module(self):
        num_channels = self.output_cfg['NUM_CHANNELS']  # 16
        input_dim = embed_dim = num_channels * 8        # 128
        num_joints = self.head_cfg['NUM_JOINTS']
        
        # Multiview Spatial Transformer
        self.multiview_encoder = SVTransformer(input_dim, embed_dim, num_joints)
        self.global_pose_mlp = KTD(num_channels * 8)
        self.global_shape_mlp = nn.Linear(num_joints * embed_dim, 10)

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
        elif self.outmap_size == 16:
            kernel_sizes = [3]
            paddings = [1]
            strides = [1]

        return kernel_sizes, strides, paddings
