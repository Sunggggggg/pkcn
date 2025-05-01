import sys, os
import os.path as osp

from config import args
from dataset.image_base import *
from utils.loader_utils import default_collate
from utils.transforms import *

axis_angle_to_rotation_6d = lambda x : matrix_to_rotation_6d(axis_angle_to_matrix(x.reshape(-1, 3))).reshape(-1)
tt = lambda x : torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x

def world_to_camera(R, phi):
    """ 
    R : [3, 3] rotation matrix
    phi : [3] orientation
    """
    if isinstance(R, np.ndarray):
        R = torch.from_numpy(R).float()
    if isinstance(phi, np.ndarray):
        phi = torch.from_numpy(phi).float()

    rot_phi = R @ axis_angle_to_matrix(phi).reshape(3, 3)
    rot_phi = matrix_to_axis_angle(rot_phi).reshape(3)
    return rot_phi.numpy()

class Hi4D(Image_base):
    def __init__(self, train_flag=True, split='train', regress_smpl=True, **kwargs):
        super(Hi4D, self).__init__(train_flag, regress_smpl)
        self.train_flag = train_flag
        self.target_id = kwargs.get('target_id', None)
        Hi4D_CAMERA_LIST = [4, 16, 28, 40, 52, 64, 76, 88]
        
        ### BUDDI SPLIT ###
        TRAIN_SPLIT = [0, 1, 2, 9, 10, 13, 14, 17, 18, 21, 23, 27, 28, 37]
        VAL_SPLIT = [16, 19, 22]
        TEST_SPLIT = [12, 15, 32]
        pair_list = eval(f"{split.upper()}_SPLIT")    
        use_depth_info = self.use_depth_info = split.upper() == 'TEST'
        
        ### ### 
        CAMERA_LIST = [4, 16, 28, 40, 52, 64, 76, 88]
        if args().num_views == 4:
            CAMERA_LIST = [4, 16, 52, 64]
        else :
            CAMERA_LIST = CAMERA_LIST[:args().num_views]
        self.camera_view = self.clip_size = len(CAMERA_LIST)

        ### Data root ###
        Hi4D_ROOT = args().dataset_rootdir

        intrinsics, extrinsics, smpl_param, file_collects, depth_collects = [], [], [], [], []
        for pair_id in pair_list :
            subj_folder = osp.join(Hi4D_ROOT, f"pair{pair_id:02d}")
            action_list = [x for x in os.listdir(subj_folder) if osp.isdir(osp.join(subj_folder, x))]

            for action in action_list :
                # Camera param. #
                camera_file = osp.join(subj_folder, action, 'cameras', 'rgb_cameras.npz')
                camera_param = np.load(camera_file)

                # SMPL #
                smpl_folder = osp.join(subj_folder, action, 'smpl')
                smpl_file_list = sorted(glob.glob(f"{smpl_folder}/*.npz"))
                smpl_param += smpl_file_list
                
                seqlen = len(smpl_file_list)
                # Image # 
                file_collect = np.empty((seqlen, len(CAMERA_LIST)), dtype=object)
                depth_collect = np.empty((seqlen, len(CAMERA_LIST)), dtype=object)
                intrin_collect = np.empty((seqlen, len(CAMERA_LIST), 3, 3))
                extrin_collect = np.empty((seqlen, len(CAMERA_LIST), 3, 4))
                for idx, cam_id in enumerate(CAMERA_LIST) :
                    ### Image file ###
                    img_folder = osp.join(subj_folder, action, 'images', str(cam_id))
                    img_file_list = sorted(glob.glob(f"{img_folder}/*.jpg"))
                    assert len(smpl_file_list) == len(img_file_list)
                    
                    file_collect[:, idx] = img_file_list    # [T]
                    
                    ### Camera ###
                    intrinsic = np.array(camera_param['intrinsics'][Hi4D_CAMERA_LIST.index(cam_id)])[None] # [1, 3, 3] 
                    extrinsic = np.array(camera_param['extrinsics'][Hi4D_CAMERA_LIST.index(cam_id)])[None] # [1, 3, 4]
                    intrinsic = np.repeat(intrinsic, len(img_file_list), axis=0)    # [T, 3, 3]
                    extrinsic = np.repeat(extrinsic, len(img_file_list), axis=0)    # [T, 3, 4]
                    
                    intrin_collect[:, idx] = intrinsic 
                    extrin_collect[:, idx] = extrinsic
                    
                    ### Depth map ###
                    if use_depth_info :
                        depth_folder = osp.join('./data/depth_map', f"pair{pair_id:02d}", action, str(cam_id))
                        depth_file_list = sorted(glob.glob(f"{depth_folder}/*.png"))
                        depth_collect[:, idx] = depth_file_list
                
                file_collect = file_collect.reshape(-1) # [T*[view0, view1, view2, view3]] Tx4
                intrin_collect = intrin_collect.reshape(-1, 3, 3)   # [Tx4, 3, 3]
                extrin_collect = extrin_collect.reshape(-1, 3, 4)   # [Tx4, 3, 4]
                
                file_collects.append(file_collect)
                intrinsics.append(intrin_collect)
                extrinsics.append(extrin_collect)
                
                if use_depth_info :
                    depth_collect = depth_collect.reshape(-1)
                    depth_collects.append(depth_collect)
                
        file_paths = np.concatenate(file_collects, axis=0)      # [Total_seq * 4]
        intrinsics = np.concatenate(intrinsics, axis=0)     # [Total_seq * 4, 3, 3]
        extrinsics = np.concatenate(extrinsics, axis=0)     # [Total_seq * 4, 3, 4]
        
        print(f">>> Total imgs : {len(file_paths)} / Total seq. : {len(file_paths)//self.camera_view} ")
        print(f">>> Total camera view : {self.camera_view} ")
        self.file_paths = file_paths
        self.clip_paths_counting_set()
        
        if use_depth_info :
            depth_paths = np.concatenate(depth_collects, axis=0)    # [Total_seq * 4]
            self.depth_paths = depth_paths
        
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.smpl_param = smpl_param
        
        SMPL_24 = {
            "Pelvis_SMPL": 0, "L_Hip_SMPL": 1, "R_Hip_SMPL": 2, "Spine_SMPL": 3,
            "L_Knee": 4, "R_Knee": 5, "Thorax_SMPL": 6, "L_Ankle": 7,
            "R_Ankle": 8, "Thorax_up_SMPL": 9, "L_Toe_SMPL": 10, "R_Toe_SMPL": 11,
            "Neck": 12, "L_Collar": 13, "R_Collar": 14, "Jaw": 15,
            "L_Shoulder": 16, "R_Shoulder": 17, "L_Elbow": 18, "R_Elbow": 19,
            "L_Wrist": 20, "R_Wrist": 21, "L_Hand": 22, "R_Hand": 23,
        }
        
        self.joint_mapper = constants.joint_mapping(
            SMPL_24, constants.SMPL_ALL_54
        )
        self.joint3d_mapper = constants.joint_mapping(
            SMPL_24, constants.SMPL_ALL_54
        )
        self.kps_vis = (self.joint_mapper != -1).astype(np.float32)[:, None]
        
        assert len(smpl_param) == len(file_paths)//self.camera_view
        # assert len(smpl_param) == len(depth_paths)//self.camera_view, f"{len(smpl_param)} != {len(depth_paths)//self.camera_view}"
        assert len(file_paths) == len(intrinsics) == len(extrinsics)
        
        if self.regress_smpl:
            from smplx import SMPLLayer
            self.smpl = SMPLLayer('model_data/parameters/SMPL_NEUTRAL.pkl', ext='pkl').eval()
            
            self.smplr = SMPLR(use_gender=False)
            self.root_inds = None
        
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

        if self.target_id is not None:
            self.select_id = self.target_id
        else :
            self.select_id = 0 if random.random() > 0.5 else 1 # select one of instances #TODO

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

    def get_image_info(self, index) :        
        imgpath = self.file_paths[index]
        image = cv2.imread(imgpath)[:, :, ::-1].copy()

        ### Camera ###
        camIntrin = np.array(self.intrinsics[index])    # [3, 3]
        camExtrin = np.array(self.extrinsics[index])    # [3, 4]
        w2c = torch.from_numpy(camExtrin[:, :3]).float()

        ### SMPL param. ###
        smpl_param_file = self.smpl_param[index // self.camera_view]
        smpl_param = dict(np.load(smpl_param_file))
        world_global_orient = np.array(smpl_param['global_orient'][self.select_id])
        body_pose = np.array(smpl_param['body_pose'][self.select_id])   # [23*3]
        betas = np.array(smpl_param['betas'][self.select_id])           # [10]
        
        ### W2C proj. ###
        cam_global_orient = world_to_camera(w2c, world_global_orient)
        params = np.concatenate([cam_global_orient, body_pose, betas], axis=-1)[None]   # [1, 82]
        global_params = np.concatenate([world_global_orient, body_pose, betas])   # [82]
        
        ### 2D keypoint, 3D keypoint (Cam) ###
        joints_3d = smpl_param['joints_3d']     # [2, 24, 3]
        joint_3d = joints_3d[self.select_id]    # [24, 3]
    
        kp2d, kp3d = [], []
        for j in range(joint_3d.shape[0]):
            padded_v = np.pad(joint_3d[j], (0,1), 'constant', constant_values=(0,1))
            img_xyz = camIntrin @ camExtrin @ padded_v.T    # world to image
            cam_xyz = camExtrin @ padded_v                  # world to camera
            pix = (img_xyz/img_xyz[2])[:2]

            kp2d.append(pix)
            kp3d.append(cam_xyz)

        kp3d, kp2d = np.array(kp3d), np.array(kp2d)
        
        kp2d = self.map_kps(kp2d.reshape(-1, 2).copy(), maps=self.joint_mapper)
        kp3ds = self.map_kps(kp3d, maps=self.joint3d_mapper)[None]
        root_trans = kp3ds[:, [0]]
        
        kp2ds = np.concatenate([kp2d, self.kps_vis], 1)[None]
        kp3ds = kp3ds - root_trans

        ### Mask ###
        

        
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
            "ds": "hi4d",
            "camIntrin": camIntrin, 
            "camExtrin": camExtrin,
            "use_depth": False,
            
            "kp3d": kp3d,
        }
        
        ### Depth map ###
        if self.use_depth_info :
            depthpath = self.depth_paths[index]
            depth_img = Image.open(depthpath).convert('L')
            depth_vis = np.array(depth_img.copy()).astype(np.uint8)
            depth_img = np.array(depth_img).astype(np.float32) / 255.0
            H, W = depth_img.shape[:2]
            depth_img = depth_img.reshape(H, W, 1) 
            
            img_info['use_depth'] = True
            img_info['depth_image'] = depth_img
            img_info['depth_vis'] = depth_vis
            
        return img_info
    
    def smpl_forward(self, smpl_params):
        return self.smpl(global_orient=axis_angle_to_matrix(tt(smpl_params[:, :3].reshape(-1, 1, 3))), 
                         body_pose=axis_angle_to_matrix(tt(smpl_params[:, 3:72].reshape(-1, 23, 3))),
                         betas=tt(smpl_params[:, 72:].reshape(-1, 10))).vertices
