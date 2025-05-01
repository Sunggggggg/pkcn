
from .base import *
from loss_funcs import calc_mpjpe, calc_pampjpe, align_by_parts
from evaluation import h36m_evaluation_act_wise, cmup_evaluation_act_wise
import time

def save_obj(verts, faces, obj_mesh_name="mesh.obj"):
    with open(obj_mesh_name, "w") as fp:
        for v in verts:
            fp.write("v %f %f %f\n" % (v[0], v[1], v[2]))

        for f in faces:  # Faces are 1-based, not 0-based in obj files
            fp.write("f %d %d %d\n" % (f[0] + 1, f[1] + 1, f[2] + 1))

@torch.no_grad()
def val_result(self, loader_val, evaluation = False):
    if self.distributed_training:
        eval_model = self.model.module
    elif self.master_batch_size!=-1:
        print("master")
        eval_model = nn.DataParallel(self.model.module)
    else:
        eval_model = self.model
    if self.backbone == 'resnet':
        eval_model.train()
    else:
        print("eval")
        eval_model.eval()
    ED = _init_error_dict()

    for iter_num, meta_data in enumerate(loader_val):
        if meta_data is None:
            print("NONE!!+++++++++++")
            continue

        meta_data_org = meta_data.copy()
        outputs_list, pred_pose_params, pred_betas = self.multiview_network_forward(eval_model, meta_data, self.eval_cfg)
        outputs = outputs_list[0]

        if not outputs['detection_flag']:
            print('Detection failure!!! {}'.format(outputs['meta_data']['imgpath']))
            continue

        error_dict= {'3d':{'error':[], 'idx':[]},'2d':{'error':[], 'idx':[]}}
        data_set = outputs['meta_data']['data_set']
        flat_data_set = [item for sublist in data_set for item in sublist]
        
        for ds in set(flat_data_set):
            val_idx = np.where(np.array(outputs['meta_data']['data_set'])==ds)[0]
            real_3d = outputs['meta_data']['kp_3d'][val_idx].contiguous().cuda()
            if ds in constants.dataset_smpl2lsp:
                real_3d = real_3d[:,self.All54_to_LSP14_mapper].contiguous()
                if (self.All54_to_LSP14_mapper==-1).sum()>0:
                    real_3d[:,self.All54_to_LSP14_mapper==-1] = -2.
                predicts = outputs['joints_h36m17'][:, :14].contiguous()
                align_inds = [constants.LSP_14['R_Hip'], constants.LSP_14['L_Hip']]
                
            else:
                predicts = outputs['j3d'][val_idx,:24].contiguous()
                real_3d = real_3d[:,:24].contiguous()
                align_inds = [constants.SMPL_24['Pelvis_SMPL']]

            if args().calc_PVE_error and ds in constants.PVE_ds:
               target_theta = torch.cat([outputs['meta_data']['params'][val_idx,:-10].cpu(), outputs['meta_data']['params'][val_idx,-10:].cpu()],1)
               pred_theta = torch.cat([outputs['params']['global_orient'], outputs['params']['body_pose'], outputs['params']['betas']],1)
               ED['PVE'][ds]['target_theta'].append(target_theta)
               ED['PVE'][ds]['pred_theta'].append(pred_theta[val_idx].float().detach().cpu())

            abs_error, aligned_poses = calc_mpjpe(real_3d, predicts, align_inds=align_inds, return_org=True)
            abs_error = abs_error.float().cpu().numpy()*1000
            rt_error = calc_pampjpe(real_3d, predicts).float().cpu().numpy()*1000

            ED['MPJPE'][ds].append(abs_error.astype(np.float32))
            ED['PA_MPJPE'][ds].append(rt_error.astype(np.float32))
            ED['imgpaths'][ds].append(np.array(outputs['meta_data']['imgpath'])[val_idx])
            error_dict['3d']['error'].append(abs_error); error_dict['3d']['idx'].append(val_idx)

        if iter_num % self.val_batch_size == 0:
            print('{}/{}'.format(iter_num, len(loader_val)))
            MPJPE_result, PA_MPJPE_result, eval_matrix = print_results(ED.copy())
            if not evaluation:
                outputs_list, _, _ = self.multiview_network_forward(eval_model, meta_data_org, self.val_cfg)
                outputs = outputs_list[0]
                
            vis_ids = np.arange(max(min(self.val_batch_size, len(outputs['reorganize_idx'])), 8)//8), None
            save_name = '{}_{}'.format(self.global_count,iter_num)
            data_set = outputs['meta_data']['data_set']
            flat_data_set = [item for sublist in data_set for item in sublist]
            
            for ds in set(flat_data_set):
                save_name += '_{}'.format(ds)
                
            # self.visualizer.visulize_result(outputs, outputs['meta_data'], show_items=['mesh', 'joint_sampler', 'j3d', 'pj2d', 'classify'],\
            #     vis_cfg={'settings': ['save_img'], 'vids': vis_ids, 'save_dir':self.result_img_dir, 'save_name':save_name}, kp3ds=(*aligned_poses, bones)) #'org_img', 
            
    print('{} on local_rank {}'.format(['Evaluation' if evaluation else 'Validation'], self.local_rank))
    MPJPE_result, PA_MPJPE_result, eval_matrix = print_results(ED)

    return MPJPE_result, PA_MPJPE_result, eval_matrix

@torch.no_grad()
def multiview_val_result(self, loader_val, evaluation = False):
    if self.distributed_training:
        eval_model = self.model.module
    elif self.master_batch_size!=-1:
        print("master")
        eval_model = nn.DataParallel(self.model.module)
    else:
        eval_model = self.model
    if self.backbone == 'resnet':
        eval_model.train()
    else:
        print("eval")
        eval_model.eval()
    ED = _init_error_dict()

    self.eval_cfg['mode'] = 'parsing'

    for iter_num, meta_data in enumerate(loader_val):
        if meta_data is None:
            print("NONE!!+++++++++++")
            continue
        
        meta_data_org = meta_data.copy()
        outputs_list, pred_pose_params, pred_betas, _ = self.multiview_network_forward(eval_model, meta_data, self.eval_cfg)
        for outputs in outputs_list :
            if not outputs['detection_flag']:
                print('Detection failure!!! {}'.format(outputs['meta_data']['imgpath']))
                continue

            error_dict= {'3d':{'error':[], 'idx':[]},'2d':{'error':[], 'idx':[]}}
            data_set = outputs['meta_data']['data_set']
    
            for ds in set(data_set):
                val_idx = np.where(np.array(outputs['meta_data']['data_set']) == ds)[0]

                real_3d = outputs['meta_data']['kp_3d'][val_idx, :24].contiguous().cuda()    # [8, 54, 3]
                predicts = outputs['refine_j3d'][val_idx, :24].contiguous()
                align_inds = [constants.SMPL_24['Pelvis_SMPL']]

                if args().calc_PVE_error and ds in constants.PVE_ds:
                    target_theta = torch.cat([outputs['meta_data']['params'][val_idx,:-10].cpu(), 
                                              outputs['meta_data']['params'][val_idx,-10:].cpu()], 1)
                    pred_theta = torch.cat([outputs['params']['global_orient'],
                                             outputs['refine_params']['body_pose'], 
                                             outputs['refine_params']['betas']], 1)
                    ED['PVE'][ds]['target_theta'].append(target_theta)
                    ED['PVE'][ds]['pred_theta'].append(pred_theta[val_idx].float().detach().cpu())

                abs_error, aligned_poses = calc_mpjpe(real_3d, predicts, align_inds=align_inds, return_org=True)
                abs_error = abs_error.float().cpu().numpy()*1000
                rt_error = calc_pampjpe(real_3d, predicts).float().cpu().numpy()*1000

                ED['MPJPE'][ds].append(abs_error.astype(np.float32))
                ED['PA_MPJPE'][ds].append(rt_error.astype(np.float32))
                ED['imgpaths'][ds].append(np.array(outputs['meta_data']['imgpath'])[val_idx])
                error_dict['3d']['error'].append(abs_error); error_dict['3d']['idx'].append(val_idx)
                
                # refine_verts = outputs['refine_verts'].contiguous()
                # cam_trans = outputs['cam_trans'].contiguous()   
                # final_verts = (refine_verts + cam_trans[:, None]).detach().cpu().numpy()    
                # detected_person = len(final_verts)
                # for person_id in range(detected_person):
                #     save_obj(final_verts[person_id], self.faces, 'test.obj')
                #     exit()
                

        if iter_num % self.val_batch_size == 0:
            print('{}/{}'.format(iter_num, len(loader_val)))
            MPJPE_result, PA_MPJPE_result, eval_matrix = print_results(ED.copy())
            if not evaluation:
                outputs_list, _, _, _ = self.multiview_network_forward(eval_model, meta_data_org, self.val_cfg)
                outputs = outputs_list[0]
                
            vis_ids = np.arange(max(min(self.val_batch_size, len(outputs['reorganize_idx'])), 8)//8), None
            save_name = '{}_{}'.format(self.global_count,iter_num)
            data_set = outputs['meta_data']['data_set']
            flat_data_set = [item for sublist in data_set for item in sublist]
            
            for ds in set(flat_data_set):
                save_name += '_{}'.format(ds)

        if iter_num == 2 :
            break
                
    print('{} on local_rank {}'.format(['Evaluation' if evaluation else 'Validation'], self.local_rank))
    MPJPE_result, PA_MPJPE_result, eval_matrix = print_results(ED)

    return MPJPE_result, PA_MPJPE_result, eval_matrix


def print_results(ED):
    MPJPE_list, PA_MPJPE_list = [], []
    for key, results in ED['MPJPE'].items():
        if len(results)>0:
            MPJPE_list += results
            PA_MPJPE_list += ED['PA_MPJPE'][key]
    MPJPE_result = np.concatenate(MPJPE_list,axis=0).mean()
    PA_MPJPE_result = np.concatenate(PA_MPJPE_list,axis=0).mean()

    eval_matrix = {}
    eval_matrix.update(process_matrix(ED['MPJPE'],'MPJPE'))
    eval_matrix.update(process_matrix(ED['PA_MPJPE'],'PA_MPJPE'))

    if args().calc_PVE_error:
        for ds_name in constants.PVE_ds:
            if len(ED['MPJPE'][ds_name])>0:
                eval_matrix['{}-PVE'.format(ds_name)] = np.mean(compute_error_verts(target_theta=torch.cat(ED['PVE'][ds_name]['target_theta'],0), \
                    pred_theta=torch.cat(ED['PVE'][ds_name]['pred_theta'],0), smpl_path=os.path.join(args().smpl_model_path, ''))) * 1000

    print_table(eval_matrix)
    
    if len(ED['MPJPE']['h36m'])>0:
        print('Detail results on Human3.6M dataset:')
        PA_MPJPE_acts = h36m_evaluation_act_wise(np.concatenate(ED['PA_MPJPE']['h36m'],axis=0),np.concatenate(np.array(ED['imgpaths']['h36m']),axis=0),constants.h36m_action_names)
        MPJPE_acts = h36m_evaluation_act_wise(np.concatenate(ED['MPJPE']['h36m'],axis=0),np.concatenate(np.array(ED['imgpaths']['h36m']),axis=0),constants.h36m_action_names)
        table = PrettyTable(['Protocol']+constants.h36m_action_names)
        table.add_row(['1']+MPJPE_acts)
        table.add_row(['2']+PA_MPJPE_acts)
        print(table)


    return MPJPE_result, PA_MPJPE_result, eval_matrix

def process_matrix(matrix, name, times=1.):
    eval_matrix = {}
    for ds, error_list in matrix.items():
        if len(error_list)>0:
            result = np.concatenate(error_list,axis=0)
            result = result[~np.isnan(result)].mean()
            eval_matrix['{}-{}'.format(ds,name)] = result*times
    return eval_matrix

def _init_error_dict():
    ED = {}
    ED['MPJPE'], ED['PA_MPJPE'], ED['PCK3D'], ED['imgpaths'] = [{ds:[] for ds in constants.dataset_involved} for _ in range(4)]
    ED['PVE'] = {ds:{'target_theta':[], 'pred_theta':[]} for ds in constants.PVE_ds}
    return ED

def print_table(eval_matrix):
    matrix_dict = {}
    em_col_id = 0
    matrix_list = []
    for name in eval_matrix:
        ds,em = name.split('-')
        if em not in matrix_dict:
            matrix_dict[em] = em_col_id
            matrix_list.append(em)
            em_col_id += 1
    
    raw_dict = {}
    for name, result in eval_matrix.items():
        ds,em = name.split('-')
        if ds not in raw_dict:
            raw_dict[ds] = np.zeros(em_col_id).tolist()
        raw_dict[ds][matrix_dict[em]] = '{:.2f}'.format(result)

    table = PrettyTable(['DS/EM']+matrix_list)
    for idx, (ds, mat_list) in enumerate(raw_dict.items()):
        table.add_row([ds]+mat_list)
    print(table)
    print('-'*20)

