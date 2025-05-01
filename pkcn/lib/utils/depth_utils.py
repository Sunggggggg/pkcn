import os
import re
import glob
import torch
import numpy as np
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt

### Function ###
def get_num(path):
    return int(re.search(r'(\d+)', os.path.basename(path)).group(1))

def load_camera(camera_file):
    camera_param = np.load(camera_file)
    
    output = {}
    ids = camera_param["ids"]                     # [8]
    intrinsics = camera_param["intrinsics"]       # [8, 3, 3]
    extrinsics = camera_param["extrinsics"]       # [8, 3, 4]
    dist_coeffs = camera_param["dist_coeffs"]     # [8, 5]

    for idx, id in enumerate(ids) :
        fx = intrinsics[idx][0, 0].tolist()
        fy = intrinsics[idx][1, 1].tolist()
        cx = intrinsics[idx][0, 2].tolist()
        cy = intrinsics[idx][1, 2].tolist()
        
        R = extrinsics[idx][:3, :3].tolist()    # [3, 3]
        t = extrinsics[idx][:3, 3].tolist()     # [3]
        
        output[str(id)] = {"R": R, "t": t, "f": [fx, fy], "c": [cx, cy]}

    return output

def transform_camera_to_world_2dgrid(points_cam, R_c2w, T_c2w):
    """
    (H, W, 3) 형태의 카메라 좌표계 포인트를 월드 좌표계로 변환
    """
    H, W, _ = points_cam.shape
    points_flat = points_cam.reshape(-1, 3)  # [H*W, 3]
    T_c2w = T_c2w.reshape(1, 3)
    points_world = (R_c2w @ points_flat.T).T + T_c2w  # [H*W, 3]
    return points_world.reshape(H, W, 3)

def load_depth_map(path):
    depth_img = Image.open(path).convert("L")
    return np.array(depth_img).astype(np.float32) / 255.0

def scale_depth_and_translation(depth, T_c2w, ref_median):
    curr_median = np.median(depth)
    if curr_median == 0:
        return depth, T_c2w
    scale = ref_median / curr_median
    return depth * scale, T_c2w * scale

def depth_to_pointcloud(depth, fx, fy, cx, cy):
    H, W = depth.shape[:2]
    i, j = np.meshgrid(np.arange(W), np.arange(H))
    Z = depth
    X = (i - cx) * Z / fx
    Y = (j - cy) * Z / fy
    points = np.stack((X, Y, Z), axis=-1)  # [H, W, 3]
    return points.reshape(-1, 3)  # [N, 3]로 반환

def create_o3d_pointcloud(points, voxel_size=0.01):
    points = np.asarray(points)
    assert points.ndim == 2 and points.shape[1] == 3, "Shape must be [N, 3]"
    assert np.isfinite(points).all(), "Contains NaN or Inf"

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd.voxel_down_sample(voxel_size)

def align_icp(source_pcd, target_pcd, threshold=0.05):
    """ICP 기반 정합"""
    result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    return result.transformation

def save_pcd_matplotlib(pcd, save_path="pcd_plot.png"):
    points = np.asarray(pcd.points)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=points[:, 2], cmap='viridis', s=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=90, azim=90)  # 원하는 시점 설정

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 시각화 저장 완료: {save_path}")

def get_id(depth_img_list, intrinsic_list):
    """
    depth_img_list : N * [512, 512, 1]
    intrinsic_list : N * [3, 3]
    """
    num_view = len(depth_img_list)

    ### 2. Load depth, intri ###
    medians = [np.median(d) for d in depth_img_list]
    pointclouds = []
    target_median_depth = 1.5
    for i in range(num_view):
        scale = target_median_depth / medians[i]
        scaled_depth = depth_img_list[i] * scale
        fx = intrinsic_list[i][0, 0]
        fy = intrinsic_list[i][1, 1]
        cx = intrinsic_list[i][2, 0]
        cy = intrinsic_list[i][2, 1]

        pc = depth_to_pointcloud(scaled_depth, fx, fy, cx, cy)
        pointclouds.append(pc)

    pcds = [create_o3d_pointcloud(pc, voxel_size=0.01) for pc in pointclouds]

    # view1을 기준으로 나머지 view2~4 ICP 정합
    base_pcd = pcds[0]
    merged_points = np.asarray(base_pcd.points)

    transformation_list = [np.eye(4)]
    for i in range(1, num_view):
        transformation = align_icp(pcds[i], base_pcd)
        aligned_pcd = pcds[i].transform(transformation)
        merged_points = np.vstack([merged_points, np.asarray(aligned_pcd.points)])

        transformation_list.append(transformation)

    transformation_list = np.stack(transformation_list, 0)
    return transformation_list

def batch_get_id(depth_img_batch, intrinsic_batch):
    """
    Args:
        depth_img_batch (_type_): [B, N, H, W, 1]
        intrinsic_batch (_type_): [B, N, 3, 3]
    """
    batch_size = depth_img_batch.shape[0]

    output_list = []
    for b in range(batch_size):
        output_list.append(get_id(depth_img_batch[b].detach().cpu().numpy(), 
                                  intrinsic_batch[b].detach().cpu().numpy()))

    output_list = np.stack(output_list) # [B, N, 4, 4]
    output_list = torch.from_numpy(output_list).to(depth_img_batch.dtype)

    return output_list

def assign_ids(x):
    """
    x: list of torch.Tensor, each of shape [<=2, 3]
    returns: list of length 4, each is [2] int ID mapping, -1 for missing
    """
    id_maps = []

    # 기준 시점 (view 0)
    ref = x[0]
    if ref.shape[0] == 0:
        return [[-1, -1] for _ in range(4)]
    elif ref.shape[0] == 1:
        ref = ref.repeat(2, 1)  # [2, 3]

    id_maps.append([0, 1])  # 기준은 그대로

    # 다른 시점
    for i in range(1, len(x)):
        cur = x[i]

        if cur.shape[0] == 0:
            id_maps.append([-1, -1])
            continue
        elif cur.shape[0] == 1:
            cur = cur.repeat(2, 1)

        # 거리 계산
        dist = torch.cdist(ref, cur)  # [2, 2]
        assign = torch.argmin(dist, dim=1).tolist()  # [2]

        id_maps.append(assign)

    return id_maps

def apply_transformation_torch(point, transformation):
    """
    point: torch.Tensor [3], 1D
    transformation: torch.Tensor [4, 4]
    return: transformed point [3]
    """
    assert point.shape == (3,)
    assert transformation.shape == (4, 4)

    point_h = torch.cat([point, torch.tensor([1.0], device=point.device, dtype=point.dtype)])  # [4]
    point_transformed = transformation @ point_h  # [4]
    return point_transformed[:3]

def center_depth_point(centers, intrinsics, transformations):
    """
    centers         : [num_person, N, 3]
    intrinsics      : [B, N, 3, 3]
    transformations : [B, N, 4, 4]
    outputs         : [B, N, num_person, 3]
    """
    transformations = transformations.to(intrinsics.device)
    num_person, num_views = centers.shape[:2]
    batch_size = intrinsics.shape[0]

    outputs = torch.empty([batch_size, num_views, num_person, 3], device=intrinsics.device)
    for b, (intrinsic, transformation) in enumerate(zip(intrinsics, transformations)) :
        for n in range(num_views):
            center = centers[:, n]  # [num_person, 3]
            fx = intrinsic[n][0, 0]
            fy = intrinsic[n][1, 1]
            cx = intrinsic[n][2, 0]
            cy = intrinsic[n][2, 1]

            RT = transformation[n]
            for person_id, pos in enumerate(center) :
                u, v, z = pos
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                point = torch.stack((x, y, z), dim=0)
                transform_point = apply_transformation_torch(point, RT)

                outputs[b, n, person_id] = transform_point

    return outputs