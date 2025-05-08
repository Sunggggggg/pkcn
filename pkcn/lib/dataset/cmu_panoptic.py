import os.path as osp
from pycocotools.coco import COCO
import sys, os
from dataset.image_base import *
from utils.transforms import projectPoints

TRAIN_LIST = [
    '160422_ultimatum1',
    '160224_haggling1',
    '160226_haggling1',
    '161202_haggling1',
    '160906_ian1',
    '160906_ian2',
    '160906_ian3',
    '160906_band1',
    '160906_band2',
    # '160906_band3',
]
VAL_LIST = ['160906_pizza1', '160422_haggling1', '160906_ian5', '160906_band4']

class CMU_Panoptic(Image_base):
    def __init__(self,train_flag=True, split='test',joint_format='h36m', **kwargs):
        super(CMU_Panoptic, self).__init__(train_flag)
        self.dataset_root = args().dataset_rootdir
        
        ### MvP SPLIT ###
        if split == 'train':
            self.sequence_list = TRAIN_LIST
            self._interval = 3
            self.cam_list = [(0, 12), (0, 6), (0, 23), (0, 13), (0, 3)][
                            :self.num_views]
            self.num_views = len(self.cam_list)
        elif split == 'validation':
            self.sequence_list = VAL_LIST
            self._interval = 12
            self.cam_list = [(0, 12), (0, 6), (0, 23), (0, 13), (0, 3)][
                            :self.num_views]
            self.num_views = len(self.cam_list)
            
        self.db_file = 'group_{}_cam{}.pkl'.\
            format(self.image_set, self.num_views)
        self.db_file = os.path.join(self.dataset_root, self.db_file)

        if osp.exists(self.db_file):
            info = pickle.load(open(self.db_file, 'rb'))
            assert info['sequence_list'] == self.sequence_list
            assert info['interval'] == self._interval
            assert info['cam_list'] == self.cam_list
            self.db = info['db']
        else:
            self.db = self._get_db()
            info = {
                'sequence_list': self.sequence_list,
                'interval': self._interval,
                'cam_list': self.cam_list,
                'db': self.db
            }
            pickle.dump(info, open(self.db_file, 'wb'))
        self.db_size = len(self.db)
        
    def _get_db(self,):
        width, height = 1920, 1080
        db = []
        for seq in self.sequence_list :
            cameras = self._get_cam(seq)
            curr_anno = osp.join(self.dataset_root, seq, 'hdPose3d_stage1_coco19')
            anno_files = sorted(glob.iglob('{:s}/*.json'.format(curr_anno)))
            
            for i, file in enumerate(anno_files):
                if i % self._interval == 0:
                    with open(file) as dfile: bodies = json.load(dfile)['bodies']
                    if len(bodies) == 0: continue
                    
                for k, v in cameras.items():
                    postfix = osp.basename(file).replace('body3DScene', '')
                    prefix = '{:02d}_{:02d}'.format(k[0], k[1])
                    image = osp.join(seq, 'hdImgs', prefix,
                                        prefix + postfix)
                    image = image.replace('json', 'jpg')
                    
                    all_poses_3d, all_poses_vis_3d, all_poses, all_poses_vis = [], [], [], []
                    for body in bodies :
                        pose3d = np.array(body['joints19']).reshape((-1, 4))
                        pose3d = pose3d[:self.num_joints]
                        joints_vis = pose3d[:, -1] > 0.1
                        if not joints_vis[self.root_id]: continue

                        # Coordinate transformation
                        M = np.array([[1.0, 0.0, 0.0],
                                        [0.0, 0.0, -1.0],
                                        [0.0, 1.0, 0.0]])
                        pose3d[:, 0:3] = pose3d[:, 0:3].dot(M)

                        all_poses_3d.append(pose3d[:, 0:3] * 10.0)
                        all_poses_vis_3d.append(np.repeat(np.reshape(joints_vis, (-1, 1)), 3, axis=1))

                        pose2d = np.zeros((pose3d.shape[0], 2))
                        pose2d[:, :2] = projectPoints(
                            pose3d[:, 0:3].transpose(), v['K'], v['R'],
                            v['t'], v['distCoef']).transpose()[:, :2]
                        x_check = \
                            np.bitwise_and(pose2d[:, 0] >= 0,
                                            pose2d[:, 0] <= width - 1)
                        y_check = \
                            np.bitwise_and(pose2d[:, 1] >= 0,
                                            pose2d[:, 1] <= height - 1)
                        check = np.bitwise_and(x_check, y_check)
                        joints_vis[np.logical_not(check)] = 0

                        all_poses.append(pose2d)
                        all_poses_vis.append(np.repeat(np.reshape(joints_vis, (-1, 1)), 2, axis=1))

                    if len(all_poses_3d) > 0:
                        our_cam = {}
                        our_cam['R'] = v['R']
                        our_cam['T'] = -np.dot(v['R'].T, v['t']) * 10.0 / 1000.  # cm -> mm -> m
                        our_cam['standard_T'] = v['t'] * 10.0 / 1000.
                        our_cam['fx'] = np.array(v['K'][0, 0])
                        our_cam['fy'] = np.array(v['K'][1, 1])
                        our_cam['cx'] = np.array(v['K'][0, 2])
                        our_cam['cy'] = np.array(v['K'][1, 2])
                        our_cam['k'] = v['distCoef'][[0, 1, 4]].reshape(3, 1)
                        our_cam['p'] = v['distCoef'][[2, 3]].reshape(2, 1)

                        db.append({
                            'key': "{}_{}{}".format(
                                seq, prefix, postfix.split('.')[0]),
                            'image': osp.join(self.dataset_root, image),
                            'joints_3d': all_poses_3d,
                            'joints_3d_vis': all_poses_vis_3d,
                            'joints_2d': all_poses,
                            'joints_2d_vis': all_poses_vis,
                            'camera': our_cam
                        })
        
    def _get_cam(self, seq):
        cam_file = osp.join(self.dataset_root, seq, 'calibration_{:s}.json'.format(seq))
        with open(cam_file) as cfile:
            calib = json.load(cfile)

        M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        cameras = {}
        for cam in calib['cameras']:
            if (cam['panel'], cam['node']) in self.cam_list:
                sel_cam = {}
                sel_cam['K'] = np.array(cam['K'])
                sel_cam['distCoef'] = np.array(cam['distCoef'])
                sel_cam['R'] = np.array(cam['R']).dot(M)
                sel_cam['t'] = np.array(cam['t']).reshape((3, 1))
                cameras[(cam['panel'], cam['node'])] = sel_cam
        return cameras
