import torch
import constants
from config import args
import numpy as np

def denormalize_center(center, size=args().centermap_size):
    center = (center+1)/2*size

    center[center<1] = 1
    center[center>size - 1] = size - 1
    if isinstance(center, np.ndarray):
        center = center.astype(np.int32)
    elif isinstance(center, torch.Tensor):
        center = center.long()
    return center

def process_gt_center(center_normed):
    """
    Args:
        center_normed : 정규화된 중심좌표 [B, max_persons, 2]
    Returns:
        valid_batch_inds : [B * max_persons] 
        valid_person_ids : [B * max_persons]
        center_gt_valid : [B, max_persons] 
    """
    valid_mask = center_normed[:,:,0]>-1
    valid_inds = torch.where(valid_mask)
    valid_batch_inds, valid_person_ids = valid_inds[0], valid_inds[1]
    center_gt = ((center_normed+1)/2*args().centermap_size).long()  
    center_gt_valid = center_gt[valid_mask] # 
    return (valid_batch_inds, valid_person_ids, center_gt_valid)

def expand_flat_inds(flat_inds: torch.Tensor,
                     H: int,
                     W: int,
                     diagonal: bool = False) -> torch.Tensor:
    """
    인접한 inds 추가
    """
    flat_inds = flat_inds.view(-1).to(torch.long)       # (N,)
    rows, cols = flat_inds // W, flat_inds % W

    # 4‑이웃 offset
    offsets = torch.tensor([-1, 1, -W, W], dtype=flat_inds.dtype, device=flat_inds.device)
    masks   = torch.stack([cols > 0,
                           cols < W - 1,
                           rows > 0,
                           rows < H - 1], dim=1)        # (N,4)

    if diagonal:                                        # ↖ ↗ ↙ ↘
        diag_off = torch.tensor([-W-1, -W+1, W-1, W+1], dtype=flat_inds.dtype, device=flat_inds.device)
        diag_msk = torch.stack([ (cols>0) & (rows>0),
                                 (cols<W-1) & (rows>0),
                                 (cols>0) & (rows<H-1),
                                 (cols<W-1) & (rows<H-1) ], dim=1)
        offsets  = torch.cat([offsets, diag_off], dim=0)    # (8,)
        masks    = torch.cat([masks,  diag_msk], dim=1)     # (N,8)

    # (N,8) = flat.unsqueeze(1) + offsets
    neigh = flat_inds.unsqueeze(1) + offsets
    neigh = neigh[masks]                                   # 유효 위치만
    all_inds = torch.unique(torch.cat([flat_inds, neigh])) # 중복 제거
    return all_inds
