
from .base import *
from loss_funcs import _calc_MPJAE, calc_mpjpe, calc_pampjpe, align_by_parts
from loss_funcs import Loss, Learnable_Loss
from .eval import val_result, print_results
from visualization.visualization import draw_skeleton_multiperson
import pandas
import pickle

class Demo(Base):
    def __init__(self):
        super(Demo, self).__init__()
        self._build_model_()
        self.test_cfg = {'mode':'parsing', 'calc_loss': False,'with_nms':True,'new_training': args().new_training}
        self.eval_dataset = args().eval_dataset
        self.save_mesh = False
        print('Initialization finished!')

    def test_eval(self):
        if self.eval_dataset == 'pw3d_test':
            data_loader = self._create_single_data_loader(dataset='pw3d', train_flag = False, mode='vibe', split='test')
        elif self.eval_dataset == 'pw3d_val':
            data_loader = self._create_single_data_loader(dataset='pw3d', train_flag = False, mode='vibe', split='val')
        elif self.eval_dataset == 'h36m_val':
            data_loader = self._create_single_data_loader(dataset='h36m', train_flag = False, split='val')
        elif self.eval_dataset == 'hi4d_test':
            data_loader = self._create_single_data_loader(dataset='hi4d', train_flag = False, split='test')
        MPJPE, PA_MPJPE, eval_results = val_result(self,loader_val=data_loader, evaluation=True)

    def net_forward(self, meta_data, mode='val'):
        if mode=='val':
            cfg_dict = self.test_cfg
        elif mode=='eval':
            cfg_dict = self.eval_cfg
        ds_org, imgpath_org = get_remove_keys(meta_data,keys=['data_set','imgpath'])
        meta_data['batch_ids'] = torch.arange(len(meta_data['image']))
        if self.model_precision=='fp16':
            with autocast():
                outputs = self.model(meta_data, **cfg_dict)
        else:
            outputs = self.model(meta_data, **cfg_dict)

        outputs['meta_data']['data_set'], outputs['meta_data']['imgpath'] = reorganize_items([ds_org, imgpath_org], outputs['reorganize_idx'].cpu().numpy())
        return outputs

    

def main():
    with ConfigContext(parse_args(sys.argv[1:])):
        demo = Demo()
        demo.test_eval()

if __name__ == '__main__':
    main()


