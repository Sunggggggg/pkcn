import sys, os

import transforms3d
from config import args
from dataset.image_base import *
from dataset.camera_parameters import h36m_cameras_intrinsic_params
from utils.loader_utils import default_collate

human36_joints_name = (
    "Pelvis",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "Torso",
    "Neck",
    "Nose",
    "Head",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
)

human36_joints_index = {
    "Pelvis": 0,
    "R_Hip": 1,
    "R_Knee": 2,
    "R_Ankle": 3,
    "L_Hip": 4,
    "L_Knee": 5,
    "L_Ankle": 6,
    "Torso": 7,
    "Neck": 8,
    "Nose": 9,
    "Head": 10,
    "L_Shoulder": 11,
    "L_Elbow": 12,
    "L_Wrist": 13,
    "R_Shoulder": 14,
    "R_Elbow": 15,
    "R_Wrist": 16,
}

class H36M(Image_base):
    def __init__(self, train_flag=True, split='train', regress_smpl=True, **kwargs):
        super(H36M, self).__init__(train_flag, regress_smpl)
        self.train_flag = train_flag
        
        ### Preprocessed PKL file ###
        annots_file = os.path.join(args().dataset_rootdir, "h36m_NeuralAnnot_{}.pkl".format(split))
        print("h36m_annot_file name : {}..... ".format(annots_file))

        with open(annots_file, "rb") as f:
            annots = pickle.load(f, encoding="latin1")
            file_paths = np.arange(0, len(annots), 1)
        
        self.file_paths = self.clip_size = file_paths
        self.camera_view = args().num_views
        self.clip_paths_counting_set()
        
        self.root_inds = [constants.SMPL_ALL_54["Pelvis"]]
        self.pelvis_inds = [constants.SMPL_ALL_54["Pelvis"]]
        self.joint_mapper = constants.joint_mapping(
            human36_joints_index, constants.SMPL_ALL_54
        )
        self.joint3d_mapper = constants.joint_mapping(
            human36_joints_index, constants.SMPL_ALL_54
        )
        self.kps_vis = (self.joint_mapper != -1).astype(np.float32)[:, None]
        
    def clip_paths_counting_set(self):
        """
        self.file_paths : [[view0_img0, view1_img0, view2_img0, view3_img0],
                            [view0_img1, view1_img1, view2_img1, view3_img1], ...]
        """
        clip_paths = [[]]
        
        for idx, _ in enumerate(self.file_paths):
            if len(clip_paths[-1]) == self.camera_view:
                clip_paths.append([])
            clip_paths[-1].append(idx)

        self.clip_paths = clip_paths
    
    def __len__(self) :
        return len(self.clip_paths)
    
    def __getitem__(self, index):
        mini_batch = []
        clip_paths = self.clip_paths[index]

        if len(clip_paths) >= self.clip_size:
            clip_paths = clip_paths[:self.clip_size]
        else:
            clip_paths = clip_paths + [clip_paths[-1]] * (
                self.clip_size - len(clip_paths)
            )
            
        for i, idx in enumerate(clip_paths):
            mini_batch.append(self.get_item_single_frame(idx))

        mini_batch = default_collate(mini_batch)
        return mini_batch
    
    def get_item_single_frame(self, index):
        img_name = self.file_paths[index]
        info = self.annots[img_name].copy()
        
        imgpath = os.path.join(self.image_folder, info["img_path"])
        if args().backbone == 'dinov2':
            image = Image.open(imgpath).convert('RGB')
            image = np.asarray(image)
        else :  # HRNet
            image = cv2.imread(imgpath)[:, :, ::-1].copy()

        ### Camera ###
        camR = np.array(info["cam_param"]["R"], dtype=np.float32).reshape(3, 3)
        camT = np.array(info["cam_param"]["t"], dtype=np.float32)
        camTT = camT.reshape(3, 1).copy()
        camExtrin = np.concatenate((camR, camTT), axis=1)
        
        cam_focalxy = np.array(info["cam_param"]["focal"], dtype=np.float32)
        cam_cxcy = np.array(info["cam_param"]["princpt"], dtype=np.float32)
        camIntrin = np.array(
            [[cam_focalxy[0], 0, cam_cxcy[0]],[0, cam_focalxy[1], cam_cxcy[1]],[0, 0, 1]]
        )
        
        ### SMPL param. ###
        world_global_orient = np.array(info['smpl_param']['pose'][:3])
        w2c = np.array(info["cam_param"]["R"], dtype=np.float32).reshape(3, 3)
        angle = np.linalg.norm(world_global_orient)
        
        world_global_orient = transforms3d.axangles.axangle2mat(world_global_orient / angle, angle)
        cam_global_orient = np.dot(w2c, world_global_orient)
        axis, angle = transforms3d.axangles.mat2axangle(cam_global_orient)
        cam_global_orient = axis * angle
        
        body_pose = np.array(info["smpl_param"]['pose'][3:])    # [69]
        betas = np.array(info['smpl_param']['shape'])
        params = np.concatenate([cam_global_orient, body_pose, betas])[None]
        global_params = np.concatenate([world_global_orient, body_pose, betas])   # [82]
        
        ### 2D keypoint, 3D keypoint (Cam) ###
        kp2d = info['joint_img'][..., :2].reshape(-1, 2).copy()
        kp3d = info['joint_cam'].reshape(-1, 3).copy() / 1000.
        
        kp2d = self.map_kps(kp2d, maps=self.joint_mapper)           # [17, 2]
        kp3ds = self.map_kps(kp3d, maps=self.joint3d_mapper)[None]  # [1, 17, 3]
        root_trans = kp3ds[:, human36_joints_name.index("Pelvis")]
        
        kp2ds = np.concatenate([kp2d, self.kps_vis], 1)[None]       # [1, 17, 3]
        kp3ds = kp3ds - root_trans
        
        # vmask_2d | 0: kp2d/bbox | 1: track ids | 2: detect all people in image
        # vmask_3d | 0: kp3d | 1: smpl global orient | 2: smpl body pose | 3: smpl body shape | 4: smpl verts
        vmask_2d = np.array([[True, True, True]])
        vmask_3d = np.array([[True, True, True, True, True, True, True]])
        img_info = {
            "imgpath": imgpath, 
            "image": image, 
            "kp2ds": kp2ds, 
            "track_ids": [self.select_id],
            "vmask_2d": vmask_2d, 
            "vmask_3d": vmask_3d, 
            "kp3ds": kp3ds, 
            "params": params, 
            "global_params": global_params, 
            "img_size": image.shape[:2], 
            "ds": "h36m",
            "camIntrin": camIntrin, 
            "camExtrin": camExtrin,
            "use_depth": False,
            'use_mask': False,
        }
        
        return img_info