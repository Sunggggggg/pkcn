import os,sys
import torch
import torch.nn as nn
import numpy as np 
import logging
import torch.nn.functional as F

import config
from config import args
import constants
from models.smpl_wrapper import SMPLWrapper

from maps_utils import HeatmapParser,CenterMap
from utils.center_utils import process_gt_center, expand_flat_inds
from utils.rot_6D import rot6D_to_angular

def parameter_sampling_with_offset(maps, batch_ids, flat_inds, offsets):
    B, C, H, W = maps.shape
    N = flat_inds.size(0)

    # Fix deprecated warning
    y = torch.div(flat_inds, W, rounding_mode='trunc').float()
    x = (flat_inds % W).float()

    y += offsets[:, 0]
    x += offsets[:, 1]

    norm_y = (y / (H - 1)) * 2 - 1
    norm_x = (x / (W - 1)) * 2 - 1
    grid = torch.stack([norm_x, norm_y], dim=1).view(N, 1, 1, 2)

    selected_maps = maps[batch_ids]  # [N, C, H, W]
    sampled = F.grid_sample(selected_maps, grid, mode='bilinear', align_corners=True)
    return sampled.squeeze(-1).squeeze(-1)

class ResultParser(nn.Module):
    def __init__(self, with_smpl_parser=True):
        super(ResultParser,self).__init__()
        self.map_size = args().centermap_size
        self.heatmap_parser = HeatmapParser()
        self.centermap_parser = CenterMap()
        self.match_preds_to_gts_for_supervision = args().match_preds_to_gts_for_supervision

    def matching_forward(self, outputs, meta_data, cfg):
        outputs,meta_data = self.match_params(outputs, meta_data, cfg)
        return outputs,meta_data

    @torch.no_grad()
    def parsing_forward(self, outputs, meta_data, cfg):
        outputs, meta_data = self.parse_maps(outputs, meta_data, cfg)
        # outputs, meta_data = self.parse_maps_v2(outputs, meta_data, cfg)
        return outputs, meta_data

    def match_params(self, outputs, meta_data, cfg):
        gt_keys = ['params', 'full_kp2d', 'kp_3d', 'subject_ids', 'valid_masks', 'heatmap', 'verts']
        exclude_keys = ['heatmap','centermap','AE_joints','person_centers','all_person_detected_mask']

        center_gts_info = process_gt_center(meta_data['person_centers'])
        center_preds_info = self.centermap_parser.parse_centermap(outputs['center_map'])
        mc_centers = self.match_gt_pred(center_gts_info, center_preds_info, outputs['center_map'].device, cfg['is_training'])
        batch_ids, flat_inds, person_ids = mc_centers['batch_ids'], mc_centers['flat_inds'], mc_centers['person_ids']

        if len(batch_ids)==0:
            if 'new_training' in cfg:
                if cfg['new_training']:
                    outputs['detection_flag'] = torch.Tensor([False for _ in range(len(meta_data['batch_ids']))]).cuda()
                    outputs['reorganize_idx'] = meta_data['batch_ids'].cuda()
                    return outputs, meta_data
            batch_ids, flat_inds = torch.zeros(1).long().to(outputs['center_map'].device), (torch.ones(1)*self.map_size**2/2.).to(outputs['center_map'].device).long()
            person_ids = batch_ids.clone()
        outputs['detection_flag'] = torch.Tensor([True for _ in range(len(batch_ids))]).cuda()

        if 'params_maps' in outputs and 'params_pred' not in outputs:      
            outputs['params_pred'] = self.parameter_sampling(outputs['params_maps'], batch_ids, flat_inds, use_transform=True)
        
        if 'params_pred_block' not in outputs :
            all_inds = [expand_flat_inds(flat_inds[i], args().centermap_size, args().centermap_size, False)
                        for i in range(len(flat_inds))]

            params_pred_block = self.parameter_block_sampling(outputs['params_maps'], batch_ids, all_inds)  # [N, 5, 400]
            # params_pred_max, _ = torch.max(params_pred_block, dim=1)
            kernel_size = params_pred_block.shape[1]
            params_pred_max = torch.max_pool1d(params_pred_block.permute(0, 2, 1), kernel_size=kernel_size, stride=1)
            params_pred_max = params_pred_max.squeeze(-1)
            
            outputs['params_pred_max'] = params_pred_max
            
        if 'offset_maps' in outputs:
            offset = self.parameter_sampling(outputs['offset_maps'], batch_ids, flat_inds, use_transform=True)
            params_pred = parameter_sampling_with_offset(outputs['params_maps'], batch_ids, flat_inds, offset)
            outputs['params_pred'] = params_pred
        
        ### Ray dirction sampling ###
        if 'ray_maps' in meta_data :
            """
            outputs['ray_maps'] : [B, 3, 64, 64]
            """
            outputs['ray_d'] = self.parameter_sampling(meta_data['ray_maps'], batch_ids, flat_inds, use_transform=True)
        
        ### Depth value sampling ###
        if 'depth_image' in meta_data :
            """
            meta_data['depth_image'] : [B, 64, 64]
            """
            depth_image = meta_data['depth_image'].reshape(-1, 1, args().centermap_size, args().centermap_size)
            depth_pred = self.parameter_sampling(depth_image, batch_ids, flat_inds, use_transform=True) # [B, 1]
            pred_h = flat_inds % args().centermap_size  # [B, ]
            pred_w = torch.div(flat_inds, args().centermap_size, rounding_mode='floor')

            outputs['depth_pred'] = depth_pred
            
        outputs, meta_data = self.reorganize_data(outputs, meta_data, exclude_keys, gt_keys, batch_ids, person_ids)
        outputs['centers_pred'] = torch.stack([flat_inds%args().centermap_size, flat_inds//args().centermap_size],1)
        
        return outputs, meta_data

    def match_gt_pred(self,center_gts_info, center_preds_info, device, is_training):
        """
        vpred_batch_ids : [N]
        flat_inds       : 
        cyxs            : [N, 2]
        top_score       : [N]
        """
        vgt_batch_ids, vgt_person_ids, vgt_centers = center_gts_info
        vpred_batch_ids, flat_inds, cyxs, top_score = center_preds_info
        mc = {key:[] for key in ['batch_ids', 'flat_inds', 'person_ids', 'conf']}

        if self.match_preds_to_gts_for_supervision:
            for match_ind in torch.arange(len(vgt_batch_ids)):
                batch_id, person_id, center_gt = vgt_batch_ids[match_ind], vgt_person_ids[match_ind], vgt_centers[match_ind]
                pids = torch.where(vpred_batch_ids==batch_id)[0]
                if len(pids) == 0:
                    continue

                closet_center_ind = pids[torch.argmin(torch.norm(cyxs[pids].float()-center_gt[None].float().to(device),dim=-1))]
                center_matched = cyxs[closet_center_ind].long()
                cy, cx = torch.clamp(center_matched, 0, self.map_size-1)
                flat_ind = cy * args().centermap_size + cx
                mc['batch_ids'].append(batch_id)
                mc['flat_inds'].append(flat_ind)
                mc['person_ids'].append(person_id)
                mc['conf'].append(top_score[closet_center_ind])
            
            keys_list = list(mc.keys())

            for key in keys_list:
                if key != 'conf':
                    mc[key] = torch.Tensor(mc[key]).long().to(device)
                if args().max_supervise_num!=-1 and is_training:
                    mc[key] = mc[key][:args().max_supervise_num]
        else:
            mc['batch_ids'] = vgt_batch_ids.long().to(device)
            mc['flat_inds'] = flatten_inds(vgt_centers.long()).to(device)
            mc['person_ids'] = vgt_person_ids.long().to(device)
            mc['conf'] = torch.zeros(len(vgt_person_ids)).to(device)
        return mc
        
    def parameter_sampling(self, maps, batch_ids, flat_inds, use_transform=True):
        device = maps.device
        if use_transform:
            batch, channel = maps.shape[:2]
            maps = maps.view(batch, channel, -1).permute((0, 2, 1)).contiguous()

        results = maps[batch_ids,flat_inds].contiguous()
        return results
    
    def parameter_block_sampling(self, maps, batch_ids, all_inds):
        batch, channel = maps.shape[:2]
        maps = maps.view(batch, channel, -1).permute(0, 2, 1).contiguous()

        results = [maps[b, inds] for b, inds in zip(batch_ids, all_inds)]

        return torch.stack(results)

    def reorganize_gts(self, meta_data, key_list, batch_ids):
        for key in key_list:
            if key in meta_data:
                if isinstance(meta_data[key], torch.Tensor):
                    meta_data[key] = meta_data[key][batch_ids]
                elif isinstance(meta_data[key], list):
                    meta_data[key] = np.array(meta_data[key])[batch_ids.cpu().numpy()]
        return meta_data

    def reorganize_data(self, outputs, meta_data, exclude_keys, gt_keys, batch_ids, person_ids):
        exclude_keys += gt_keys
        outputs['reorganize_idx'] = meta_data['batch_ids'][batch_ids]
        info_vis = []
        for key, item in meta_data.items():
            if key not in exclude_keys:
                info_vis.append(key)
        meta_data = self.reorganize_gts(meta_data, info_vis, batch_ids)
        for gt_key in gt_keys:
            if gt_key in meta_data:
                try:
                    meta_data[gt_key] = meta_data[gt_key][batch_ids,person_ids]
                except Exception as error:
                    print(gt_key,'meets error: ',error)
        
        return outputs,meta_data

    @torch.no_grad()
    def parse_maps(self,outputs, meta_data, cfg):
        center_preds_info = self.centermap_parser.parse_centermap_heatmap_adaptive_scale_batch(outputs['center_map'])
        batch_ids, flat_inds, cyxs, top_score = center_preds_info
        if len(batch_ids)==0:
            if 'new_training' in cfg:
                if cfg['new_training']:
                    outputs['detection_flag'] = torch.Tensor([False for _ in range(len(meta_data['batch_ids']))]).cuda()
                    outputs['reorganize_idx'] = meta_data['batch_ids'].cuda()
                    return outputs, meta_data
            
            batch_ids, flat_inds = torch.zeros(1).long().to(outputs['center_map'].device), (torch.ones(1)*self.map_size**2/2.).to(outputs['center_map'].device).long()
            outputs['detection_flag'] = torch.Tensor([False for _ in range(len(batch_ids))]).cuda()
        else:
            outputs['detection_flag'] = torch.Tensor([True for _ in range(len(batch_ids))]).cuda()

        if 'params_pred' not in outputs and 'params_maps' in outputs:
            """
            outputs['params_maps'] : [B, 400, 64, 64]
            batch_ids : [N] (0<batch_ids<B)
            flat_inds : [K] (0<flat_inds<4096) 사람이 많으면 K가 증가함
            
            outputs['params_pred'] : [N, 400] (N : 검출된 사람의 수)
            """
            outputs['params_pred'] = self.parameter_sampling(outputs['params_maps'], batch_ids, flat_inds, use_transform=True)
            
        ### Ray dirction sampling ###
        if 'ray_maps' in meta_data :
            """
            outputs['ray_maps'] : [B, 3, 64, 64]
            """
            outputs['ray_d'] = self.parameter_sampling(meta_data['ray_maps'], batch_ids, flat_inds, use_transform=True)
        
        ### Depth value sampling ###
        if 'depth_image' in meta_data :
            """
            outputs['depth_image'] : [B, 64, 64]
            """
            depth_image = meta_data['depth_image'].reshape(-1, 1, args().centermap_size, args().centermap_size)
            depth_pred = self.parameter_sampling(depth_image, batch_ids, flat_inds, use_transform=True) # [N, 1]
            pred_h = flat_inds % args().centermap_size  # [N]
            pred_w = torch.div(flat_inds, args().centermap_size, rounding_mode='floor')

            center_3d = torch.stack([pred_h, pred_w, depth_pred.squeeze(1)], dim=-1)    # [N, 3]
            outputs['depth_pred'] = depth_pred
            outputs['center_3d'] = center_3d
            
        if 'params_pred_block' not in outputs :
            all_inds = [expand_flat_inds(flat_inds[i], args().centermap_size, args().centermap_size, False)
                        for i in range(len(flat_inds))]

            params_pred_block = self.parameter_block_sampling(outputs['params_maps'], batch_ids, all_inds)  # [N, 5, 400]
            # params_pred_max, _ = torch.max(params_pred_block, dim=1)
            kernel_size = params_pred_block.shape[1]
            params_pred_max = torch.max_pool1d(params_pred_block.permute(0, 2, 1), kernel_size=kernel_size, stride=1)
            params_pred_max = params_pred_max.squeeze(-1)
            
            outputs['params_pred_max'] = params_pred_max
            
        if 'centers_pred' not in outputs:
            pred_h = flat_inds%args().centermap_size
            pred_w = torch.div(flat_inds, args().centermap_size, rounding_mode='floor')
            
            outputs['centers_pred'] = torch.stack([pred_h, pred_w], 1)  # [N, 2]
            outputs['centers_conf'] = self.parameter_sampling(outputs['center_map'], batch_ids, flat_inds, use_transform=True)
        
        outputs['reorganize_idx'] = meta_data['batch_ids'][batch_ids]
        info_vis = ['image', 'offsets','imgpath']
        meta_data = self.reorganize_gts(meta_data, info_vis, batch_ids)
        
        return outputs,meta_data

    @torch.no_grad()
    def parse_maps_v2(self, outputs, meta_data, cfg):
        """
        sampling appearance block 
        """
        center_preds_info = self.centermap_parser.parse_centermap_heatmap_adaptive_scale_batch(outputs['center_map'])
        batch_ids, flat_inds, cyxs, top_score = center_preds_info
        if len(batch_ids)==0:
            if 'new_training' in cfg:
                if cfg['new_training']:
                    outputs['detection_flag'] = torch.Tensor([False for _ in range(len(meta_data['batch_ids']))]).cuda()
                    outputs['reorganize_idx'] = meta_data['batch_ids'].cuda()
                    return outputs, meta_data
            
            batch_ids, flat_inds = torch.zeros(1).long().to(outputs['center_map'].device), (torch.ones(1)*self.map_size**2/2.).to(outputs['center_map'].device).long()
            outputs['detection_flag'] = torch.Tensor([False for _ in range(len(batch_ids))]).cuda()
        else:
            outputs['detection_flag'] = torch.Tensor([True for _ in range(len(batch_ids))]).cuda()
            
        if 'params_pred' not in outputs and 'params_maps' in outputs:
            """
            outputs['params_maps'] : [B, 400, 64, 64]
            batch_ids : [N] (0<batch_ids<B)
            flat_inds : [K] (0<flat_inds<4096) 사람이 많으면 K가 증가함
            
            outputs['params_pred'] : [N, 400] (N : 검출된 사람의 수)
            """
            outputs['params_pred'] = self.parameter_sampling(outputs['params_maps'], batch_ids, flat_inds, use_transform=True)
            
        ### Extract appearance feature block ###
        if 'params_pred_block' not in outputs :
            all_inds = [expand_flat_inds(flat_inds[i], args().centermap_size, args().centermap_size, False)
                        for i in range(len(flat_inds))]

            params_pred_block = self.parameter_block_sampling(outputs['params_maps'], batch_ids, all_inds)  # [N, 5, 400]
            params_pred_max, _ = torch.max(params_pred_block, dim=1)
            
            outputs['params_pred_max'] = params_pred_max
            
        if 'centers_pred' not in outputs:
            pred_h = flat_inds%args().centermap_size
            pred_w = torch.div(flat_inds, args().centermap_size, rounding_mode='floor')
            
            outputs['centers_pred'] = torch.stack([pred_h, pred_w], 1)  # [N, 2]
            outputs['centers_conf'] = self.parameter_sampling(outputs['center_map'], batch_ids, flat_inds, use_transform=True)
        
        outputs['reorganize_idx'] = meta_data['batch_ids'][batch_ids]
        info_vis = ['image', 'offsets','imgpath']
        meta_data = self.reorganize_gts(meta_data, info_vis, batch_ids)
        
        return outputs,meta_data
    
    @torch.no_grad()
    def match_parse_maps(self,outputs, meta_data, cfg):
        center_preds_info = self.centermap_parser.parse_centermap_heatmap_adaptive_scale_batch(outputs['center_map'])
        batch_ids, flat_inds, cyxs, top_score = center_preds_info
        if len(batch_ids)==0:
            if 'new_training' in cfg:
                if cfg['new_training']:
                    outputs['detection_flag'] = torch.Tensor([False for _ in range(len(meta_data['batch_ids']))]).cuda()
                    outputs['reorganize_idx'] = meta_data['batch_ids'].cuda()
                    return outputs, meta_data
            
            batch_ids, flat_inds = torch.zeros(1).long().to(outputs['center_map'].device), (torch.ones(1)*self.map_size**2/2.).to(outputs['center_map'].device).long()
            person_ids = batch_ids.clone()
            outputs['detection_flag'] = torch.Tensor([False for _ in range(len(batch_ids))]).cuda()
        else:
            outputs['detection_flag'] = torch.Tensor([True for _ in range(len(batch_ids))]).cuda()

        if 'params_pred' not in outputs and 'params_maps' in outputs:
            """
            outputs['params_maps'] : [B, 400, 64, 64]
            batch_ids : [N] (0<batch_ids<B)
            flat_inds : [K] (0<flat_inds<4096) 사람이 많으면 K가 증가함
            
            outputs['params_pred'] : [N, 400]
            """
            outputs['params_pred'] = self.parameter_sampling(outputs['params_maps'], batch_ids, flat_inds, use_transform=True)
            
        if 'centers_pred' not in outputs:
            pred_h = flat_inds%args().centermap_size
            pred_w = torch.div(flat_inds, args().centermap_size, rounding_mode='floor')
            
            outputs['centers_pred'] = torch.stack([pred_h, pred_w], 1)  # [N, 2]
            outputs['centers_conf'] = self.parameter_sampling(outputs['center_map'], batch_ids, flat_inds, use_transform=True)
        
        outputs['reorganize_idx'] = meta_data['batch_ids'][batch_ids]
        info_vis = ['image', 'offsets','imgpath']
        meta_data = self.reorganize_gts(meta_data, info_vis, batch_ids)
        
        return outputs,meta_data

    def parse_kps(self, heatmap_AEs, kp2d_thresh=0.1):
        kps = []
        heatmap_AE_results = self.heatmap_parser.batch_parse(heatmap_AEs.detach())
        for batch_id in range(len(heatmap_AE_results)):
            kp2d, kp2d_conf = heatmap_AE_results[batch_id]
            kps.append(kp2d[np.array(kp2d_conf)>kp2d_thresh])
        return kps


def flatten_inds(coords):
    coords = torch.clamp(coords, 0, args().centermap_size-1)
    return coords[:,0].long()*args().centermap_size+coords[:,1].long()

def _check_params_pred_(params_pred_shape, batch_length):
    assert len(params_pred_shape)==2, logging.error('outputs[params_pred] dimension less than 2, is {}'.format(len(params_pred_shape)))
    assert params_pred_shape[0]==batch_length, logging.error('sampled length not equal.')

def _check_params_sampling_(param_maps_shape, dim_start, dim_end, batch_ids, sampler_flat_inds_i):
    assert len(param_maps_shape)==3, logging.error('During parameter sampling, param_maps dimension is not equal 3, is {}'.format(len(param_maps_shape)))
    assert param_maps_shape[2]>dim_end>=dim_start, \
    logging.error('During parameter sampling, param_maps dimension -1 is not larger than dim_end and dim_start, they are {},{},{}'.format(param_maps_shape[-1],dim_end,dim_start))
    assert (batch_ids>=param_maps_shape[0]).sum()==0, \
    logging.error('During parameter sampling, batch_ids {} out of boundary, param_maps_shape[0] is {}'.format(batch_ids,param_maps_shape[0]))
    assert (sampler_flat_inds_i>=param_maps_shape[1]).sum()==0, \
    logging.error('During parameter sampling, sampler_flat_inds_i {} out of boundary, param_maps_shape[1] is {}'.format(sampler_flat_inds_i,param_maps_shape[1]))