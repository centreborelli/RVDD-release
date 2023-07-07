import argparse
import os
from util import util
import torch
import models
import data


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""

        # basic parameters
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')

        # model parameters
        parser.add_argument('--model', type=str, default='recurrent', help='chooses which model to use. [recurrent]')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 4 for raw')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 4 for raw')
        parser.add_argument('--netDenoiser', type=str, default='convunet-mode=fixedfeatures', help='specify network architecture')
        parser.add_argument('--init_type', type=str, default='kaiming', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        
        # dataset parameters
        parser.add_argument('--dataroot', type=str, default='./datasets/train_dataset', required=False, help='Root folder in which to look for noisy and groundtruth.')
        parser.add_argument('--nFolder', type=str, default='noisy', required=False, help='Subfolder of "dataroot" in which to find the noisy data.')
        parser.add_argument('--gtFolder', type=str, default='gt', required=False, help='Subfolder of "dataroot" in which to find the groundtruth data.')
        parser.add_argument('--gt_linear_RGB_Folder', type=str, default='gt_linear_RGB', required=False, help='Subfolder of "dataroot" in which to find the linear_RGB groundtruth data.')
        parser.add_argument('--wFolder', type=str, default='warped', required=False, help='Subfolder of "dataroot" in which to find the warped data.')
        parser.add_argument('--flowFolder', type=str, default='flow', required=False, help='Subfolder of "dataroot" in which to find the flow data.')
        parser.add_argument('--raw_linear_RGB_Folder', type=str, default='raw_linear_RGB', required=False, help='Subfolder of "dataroot" in which to find the linear RGB raw data. Can be None.')
        parser.add_argument('--bit_depth', type=int, default=12, help='Bit depth of gt and noisy input images')
        parser.add_argument('--check_data', default=True, action='store_true', help='Checking if data has been properly generated.')
        parser.add_argument('--no_warp', action='store_true', default=False, help='Do not align frames by warping them according to the optical flow.')
        parser.add_argument('--warp_method', type=str, default='tvl1', required=False, help='Warping method [tvl1]')
        parser.add_argument('--videos',  type=str, default=None, help='which videos to use.') # works for inferaxel and axel4rec

        parser.add_argument('--dataset_mode', type=str, default='axel4rec', help='chooses how datasets are loaded.')
        parser.add_argument('--serial_batches', default=False, action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--non_blocking', default=True, action='store_true', help='if true, it non-blocks memory copying whenever it could be.')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
        parser.add_argument('--max_dataset_size', type=int, default=90000, help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--patch_width',  type=int, default=136, help='training patches of this size WxW')
        parser.add_argument('--patch_stride',  type=int, default=3, help='temporal and spatial stride (or step) to form the 3D patch')
        parser.add_argument('--patch_depth',  type=int, default=2, help='number of past frames loaded in training (including current frame)')
        parser.add_argument('--future_patch_depth', type=int, default=0, help='number of future frames loaded in training')

        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest_val', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix')

        parser.add_argument('--no_predemosaic', action='store_true', default=False, help='Do not demosaic with Hamilton-Adams before the network.')
        parser.add_argument('--raw_gt', action='store_true', default=False, help='Use raw images as ground truth in the loss (for raw denoiser) instead of RGB ones (for joint denoising and demosaicing).')

        parser.add_argument('--val_flow_from_denoised', action='store_true', default=False, help='In validation/testing compute flow using previous denoised frame (disabled for in-training validation).')


        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
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

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        if self.isTrain:
            file_name = os.path.join(expr_dir, 'opt_train.txt')
        else:
            file_name = os.path.join(expr_dir, 'opt_test.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
    
    @staticmethod
    def update_name(opt):
        warpstr = '-warp' if not opt.no_warp else ''
        suffixstr = "-" + opt.suffix if not opt.suffix == '' else ""
        opt.name = "%s-%s%s-i%do%d%s" % (opt.model, opt.netDenoiser, warpstr,
                                         opt.input_nc, opt.output_nc, suffixstr)
        return opt

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        opt = self.update_name(opt)

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
