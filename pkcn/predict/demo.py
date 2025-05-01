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
        results_all = {}
        for test_iter,meta_data in enumerate(internet_loader):
            # start = time.time()
            outputs = self.net_forward(meta_data, cfg=self.demo_cfg)
            # end = time.time()
            # total += end-start
            reorganize_idx = outputs['reorganize_idx'].cpu().numpy()
            counter.count(self.val_batch_size)
            results = self.reorganize_results(outputs, outputs['meta_data']['imgpath'], reorganize_idx)
            
            np.savez(output_file, results=results)

def main():
    with ConfigContext(parse_args(sys.argv[1:])) as args_set:
        print('Loading the configurations from {}'.format(args_set.configs_yml))
        processor = Image_processor(args_set=args_set)
        inputs = args_set.inputs
        outputs = args_set.output_dir
        processor.run(inputs, outputs)

if __name__ == '__main__':
    main()