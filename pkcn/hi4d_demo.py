import os
from tqdm import tqdm
from .base import *
from .eval import val_result, multiview_val_result
from loss_funcs import Loss, Learnable_Loss, batch_l2_loss, batch_l2_loss_param
np.set_printoptions(precision=2, suppress=True)

def save_obj(verts, faces, obj_mesh_name="mesh.obj"):
    with open(obj_mesh_name, "w") as fp:
        for v in verts:
            fp.write("v %f %f %f\n" % (v[0], v[1], v[2]))

        for f in faces:  # Faces are 1-based, not 0-based in obj files
            fp.write("f %d %d %d\n" % (f[0] + 1, f[1] + 1, f[2] + 1))

class Tester(Base):
    def __init__(self):
        super(Tester, self).__init__()
        self._build_model_()

        self.eval_datasets = 'hi4d_test'
        self.set_up_val_loader()
        self.faces = self.model.module.params_map_parser.smpl_model.faces
        self.dataset_rootdir = args().dataset_rootdir
        self.output_dir = args().output_dir
   
        logging.info('Following buddi(CVPR24) proto. test split!')
        logging.info(f'# of views : {args().num_views}')

    def test(self):
        eval_model = self.model
        logging.info('Hi4D Test split / Eval mode')
        self.eval_cfg['mode'] = 'parsing'

        mesh_folder = os.path.join(self.output_dir)
        inf_time_list = []
        
        for ds_name, test_loader in self.dataset_test_list.items():
            logging.info('Evaluation on {}'.format(ds_name))
            pbar = tqdm(enumerate(test_loader), desc='obj file')
            for iter_num, meta_data in pbar:
                imgpath_list = meta_data['imgpath']
                
                meta_data_org = meta_data.copy()
                outputs_list, inf_time = self.multiview_network_forward(eval_model, meta_data, self.eval_cfg)

                inf_time_list.append(inf_time)
                for idx, (outputs, imgpath) in enumerate(zip(outputs_list, imgpath_list)) :
                    if not outputs['detection_flag']: print('Detection failure!!! {}'.format(outputs['meta_data']['imgpath'])); continue
                    imgpath = imgpath[0]
                    if '/4/' in imgpath:
                    # if True :
                        imgpath = imgpath.replace(self.dataset_rootdir, mesh_folder)
                        obj_file = imgpath.replace('images', 'meshes').replace('.jpg', '.obj')

                        root = os.path.dirname(obj_file)
                        name = os.path.basename(obj_file)
                        os.makedirs(root, exist_ok=True)
                        os.makedirs(root.replace('meshes', 'params'), exist_ok=True)
                        
                        refine_verts = outputs['refine_verts'].contiguous()
                        cam_trans = outputs['cam_trans'].contiguous()

                        pred_pose_params, pred_betas = outputs['body_pose'], outputs['betas']

                        final_verts = (refine_verts + cam_trans[:, None]).detach().cpu().numpy()
                        detected_person = len(final_verts)
                        for person_id in range(detected_person):
                            obj_file = os.path.join(root, f"{person_id}_{name}")
                            save_obj(final_verts[person_id], self.faces, obj_file)
                            
                            param_file = obj_file.replace('meshes', 'params').replace('.obj', '.pkl')
                            
                            param_dict = {
                                'body_pose': pred_pose_params[person_id],
                                'betas': pred_betas[person_id]
                            }
                            with open(param_file, 'wb') as f :
                                pickle.dump(param_dict, f)

                            pbar.set_postfix_str(f'{detected_person}, {obj_file}, {param_file}, {pred_pose_params.shape}, {pred_betas.shape}')

        avg = lambda x : sum(x) / len(x)
        print(f">>> Inference time : {avg(inf_time_list)}")  


def main():
    with ConfigContext(parse_args(sys.argv[1:])):
        trainer = Tester()
        trainer.test()

if __name__ == '__main__':
    main()