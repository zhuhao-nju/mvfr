import argparse
import os
import time
import json


class BaseOptions():
    def __init__(self, useJupyterNotebookArgs=None):
        self.useJupyterNotebookArgs = useJupyterNotebookArgs

    def initialize(self, parser):
        """ 
        Changed name
        general         =>  +if_
        uv_size         ->  (half size)
        movement_num    ->  d_size
        num_sample_inout->  points_num
        """



        # global defination
        g_global = parser.add_argument_group('Global')
        g_global.add_argument('--seed', type=int, default=1, help='random seed')
        g_global.add_argument('--cuda_id', type=int, default=0, help='cuda id')

        # UVD params
        g_uvd = parser.add_argument_group('UVD')
        g_uvd.add_argument('--uv_size', type=int, default=256, help='uv space core size')
        g_uvd.add_argument('--d_size', type=int, default=201, help='displacement number')
        g_uvd.add_argument('--interval_scale', type=float, default=0.2, help='interval degree in mm')

        # Base Mesh Fitting related
        g_fit = parser.add_argument_group('Base Mesh Fitting')
        #g_fit.add_argument('--fit_dataroot', type=str, default=r'H:\mvfr_released\data', help='path to fit (data folder)')
        g_fit.add_argument('--fit_dataroot', type=str, default=r'/media/xyz/RED31/mvfr_released/demo', help='path to fit (data folder)')
        
        # Implicit function Sampling related
        g_sample = parser.add_argument_group('Implicit function Sampling')
        g_sample.add_argument('--points_num', type=int, default=10000, help='# of sampling points in training')
        g_sample.add_argument('--sigma', type=float, default=10.0, help='perturbation standard deviation for positions')

        # Implicit function Dataset related
        g_if_data = parser.add_argument_group('Implicit function Dataset')
        g_if_data.add_argument('--if_dataroot', type=str, default=r'/media/xyz/RED31/mvfr_released/demo', help='path to images (data folder)')
        g_if_data.add_argument('--if_view_num', type=int, default=10, help='view number')
        g_if_data.add_argument('--if_image_downsample_scale', type=float, default=0.5, help='input images downsample scale')
        g_if_data.add_argument('--if_feature_downsample_scale', type=float, default=0.5, help='feature downsample scale in feature extractor')
        
        # Implicit function Training related
        g_if_train = parser.add_argument_group('Implicit function Training')
        g_if_train.add_argument('--if_training_batch_size', type=int, default=1, help='batch size')
        g_if_train.add_argument('--if_learning_rate', type=float, default=1e-3, help='learning rate')
        g_if_train.add_argument('--if_training_epoch', type=int, default=200, help='end epoch to train')
        g_if_train.add_argument('--if_model_path', type=str, default="../model", help='path to save model')
        
        # Implicit function Evaling related
        g_if_eval = parser.add_argument_group('Implicit  function Evaling')
        g_if_eval.add_argument('--if_evaling_batch_size', type=int, default=1, help='input batch size')
        g_if_eval.add_argument('--if_load_model_path', type=str, default="../model", help='path to load model')
        g_if_eval.add_argument('--if_save_vt', type=bool, default=True, help='whether to save vertex texture during meshing') # TODO: use it in coding

        # Implicit function Model related
        g_if_model = parser.add_argument_group('Implicit function Model')
        g_if_model.add_argument('--norm', type=str, default='group', help='batch normalization or group normalization') # TODO: support in network
        g_if_model.add_argument('--mlp_dim', nargs='+', default=[128+1, 1024, 512, 256, 128, 1], type=int, help='# of dimensions of mlp')
        # loss func
        g_if_model.add_argument('--if_loss_func', type=str, default="MSE", help='L1 | MSE') # TODO: support in network and separate it from model.forward()

        # Regularization Dataset related
        g_reg_data = parser.add_argument_group('Regularization Dataset')
        g_reg_data.add_argument('--reg_dataroot', type=str, default=r'/media/xyz/RED31/mvfr_released/demo', help='path to data folder')
        # Regularization Training related
        g_reg_train = parser.add_argument_group('Regularization Training')
        g_reg_train.add_argument('--reg_training_batch_size', type=int, default=1, help='batch size')
        g_reg_train.add_argument('--reg_learning_rate', type=float, default=1e-3, help='learning rate')
        g_reg_train.add_argument('--reg_training_epoch', type=int, default=25, help='end epoch to train')
        g_reg_train.add_argument('--reg_model_path', type=str, default="../model", help='path to save model')

        # Regularization Evaling related
        g_reg_eval = parser.add_argument_group('Regularization Evaling')
        g_reg_eval.add_argument('--reg_evaling_batch_size', type=int, default=1, help='input batch size')
        g_reg_eval.add_argument('--reg_load_model_path', type=str, default="../model", help='path to load model')

        # Texture related
        g_tex = parser.add_argument_group('Texture Generating')
        g_tex.add_argument('--tex_uv_size', type=int, default=1024, help='texture uv size')
        
        # Pipeline evaling related
        g_pipe = parser.add_argument_group('Pipeline')
        g_pipe.add_argument('--pipe_batch_size', type=int, default=1, help='batch size for pipeline evaling. Only support 1 now.')



        return parser

    def gather_options(self):
        parser = argparse.ArgumentParser()
        parser = self.initialize(parser)
        self.parser = parser
        if self.useJupyterNotebookArgs is not None:
            # jupyter notebook sys got more params which can not be analysed when args = None
            return parser.parse_args(args=self.useJupyterNotebookArgs)
        else:
            return parser.parse_args()

    def parse(self):
        opt = self.gather_options()
        self.print_options(opt)
        # TODO: use save_options
        #self.save_options(opt)
        return opt
     
    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def save_options(self, opt):
        filepath = "./log/options-" + time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()) + ".json"
        with open(filepath, 'w') as f:
            json.dump(vars(opt), f)
        return
        
if __name__ == "__main__":
    print(__name__)
    # get options
    base_option = BaseOptions()
    opt = base_option.parse()
    #print(base_option.parser.print_help())
    