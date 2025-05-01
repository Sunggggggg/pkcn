import sys 
whether_set_yml = ['configs_yml' in input_arg for input_arg in sys.argv]
if sum(whether_set_yml)==0:
    default_webcam_configs_yml = "--configs_yml=configs/image.yml"
    print('No configs_yml is set, set it to the default {}'.format(default_webcam_configs_yml))
    sys.argv.append(default_webcam_configs_yml)
from .base_predictor import *
import constants
import glob
from utils.util import collect_image_list
import time
import numpy as np

def reorganize_results(self, outputs, img_paths, reorganize_idx):
    prefix = 'refine_'
    results = {}
    cam_results = outputs['params']['cam'].detach().cpu().numpy().astype(np.float16)
    trans_results = outputs['cam_trans'].detach().cpu().numpy().astype(np.float16)
    smpl_pose_results = outputs[prefix+'params']['poses'].detach().cpu().numpy().astype(np.float16)
    smpl_shape_results = outputs[prefix+'params']['betas'].detach().cpu().numpy().astype(np.float16)
    joints_54 = outputs[prefix+'j3d'].detach().cpu().numpy().astype(np.float16)
    verts_results = outputs[prefix+'verts'].detach().cpu().numpy().astype(np.float16)
    
    pj2d_results = outputs['pj2d'].detach().cpu().numpy().astype(np.float16)
    pj2d_org_results = outputs['pj2d_org'].detach().cpu().numpy().astype(np.float16)
    center_confs = outputs['centers_conf'].detach().cpu().numpy().astype(np.float16)
    
    cmap = outputs["center_map"]
    vids_org = np.unique(reorganize_idx)
    for idx, vid in enumerate(vids_org):
        verts_vids = np.where(reorganize_idx==vid)[0]
        img_path = img_paths[verts_vids[0]]                
        results[img_path] = [{} for idx in range(len(verts_vids))]
        for subject_idx, batch_idx in enumerate(verts_vids):
            results[img_path][subject_idx]['cam'] = cam_results[batch_idx]
            results[img_path][subject_idx]['cam_trans'] = trans_results[batch_idx]
            results[img_path][subject_idx]['poses'] = smpl_pose_results[batch_idx]
            results[img_path][subject_idx]['betas'] = smpl_shape_results[batch_idx]
            results[img_path][subject_idx]['j3d_all54'] = joints_54[batch_idx]
            results[img_path][subject_idx]['verts'] = verts_results[batch_idx]
            results[img_path][subject_idx]['pj2d'] = pj2d_results[batch_idx]
            results[img_path][subject_idx]['pj2d_org'] = pj2d_org_results[batch_idx]
            results[img_path][subject_idx]['center_map'] = cmap[reorganize_idx][batch_idx]
            results[img_path][subject_idx]['center_conf'] = center_confs[batch_idx]
    return results

class Image_processor(Predictor):
    def __init__(self, **kwargs):
        super(Image_processor, self).__init__(**kwargs)
        self.__initialize__()

    @torch.no_grad()
    def run(self, image_folder, output_file):
        print('Processing {}, saving to {}'.format(image_folder, self.output_dir))
        os.makedirs(self.output_dir, exist_ok=True)
        counter = Time_counter(thresh=1)
        total = 0

        file_list = collect_image_list(image_folder=image_folder, collect_subdirs=self.collect_subdirs, img_exts=constants.img_exts)
        internet_loader = self._create_single_data_loader(dataset='internet', train_flag=False, file_list=file_list, shuffle=False)
        counter.start()
        results_all = []
        for test_iter,meta_data in enumerate(internet_loader):
            # start = time.time()
            outputs_list, pred_pose_params, pred_betas = self.net_forward(meta_data, cfg=self.demo_cfg)
            for view_id, outputs in enumerate(outputs_list):
                reorganize_idx = outputs['reorganize_idx'].cpu().numpy()
                counter.count(self.val_batch_size)
                results = self.reorganize_results(outputs, outputs['meta_data']['imgpath'], reorganize_idx)

                results_all.append(results)
        np.savez(output_file, results=results_all)

def main():
    with ConfigContext(parse_args(sys.argv[1:])) as args_set:
        print('Loading the configurations from {}'.format(args_set.configs_yml))
        processor = Image_processor(args_set=args_set)
        inputs = args_set.inputs
        outputs = args_set.output_dir
        processor.run(inputs, outputs)

if __name__ == '__main__':
    main()