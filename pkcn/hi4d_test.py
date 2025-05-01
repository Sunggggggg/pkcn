from .base import *
from .eval import val_result, multiview_val_result
np.set_printoptions(precision=2, suppress=True)

class Tester(Base):
    def __init__(self):
        super(Tester, self).__init__()
        self._build_model_()
        self.set_up_val_loader()
   
        logging.info('Following buddi(CVPR24) proto. test split!')
        logging.info(f'# of views : {args().num_views}')

    def test(self):
        logging.info('Hi4D Test split')
        for ds_name, val_loader in self.dataset_test_list.items():
            logging.info('Evaluation on {}'.format(ds_name))
            MPJPE, PA_MPJPE, eval_results = multiview_val_result(self,loader_val=val_loader, evaluation=False)

            exit()
            # self.evaluation_results_dict[ds_name]['MPJPE'].append(MPJPE)
            # self.evaluation_results_dict[ds_name]['PAMPJPE'].append(PA_MPJPE)

            # logging.info('Running evaluation results:')
            # ds_running_results = self.get_running_results(ds_name)
            # print('Running MPJPE:{}|{}; Running PAMPJPE:{}|{}'.format(*ds_running_results))

    def get_running_results(self, ds):
        mpjpe = np.array(self.evaluation_results_dict[ds]['MPJPE'])
        pampjpe = np.array(self.evaluation_results_dict[ds]['PAMPJPE'])
        mpjpe_mean, mpjpe_var, pampjpe_mean, pampjpe_var = np.mean(mpjpe), np.var(mpjpe), np.mean(pampjpe), np.var(pampjpe)
        return mpjpe_mean, mpjpe_var, pampjpe_mean, pampjpe_var

def main():
    with ConfigContext(parse_args(sys.argv[1:])):
        trainer = Tester()
        trainer.test()

if __name__ == '__main__':
    main()