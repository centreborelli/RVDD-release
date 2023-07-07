import os
import os.path
import numpy as np
import numpy.random
import torch
import torch.utils.data as udata
import random
import fnmatch
import random
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image

from library import *

class infer4recDataset(BaseDataset):
    """A dataset class for testing! It will provide (in a serialize way) entire
       images to denoise.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for
           existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can
                               use this flag to add training-specific or
                               test-specific options.

        Returns:
            the modified parser.
        """
        BaseDataset.modify_commandline_options(parser, is_train)
        parser.add_argument('--crop_data', type=str, default=None,
                            help='Crops all output data from this dataset. --crop_data x,y does img[:x,:y].')
        parser.add_argument('--warpeddata', action='store_true', default=False,
                            help='Force this dataloader to give warped data too.')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be
                                  a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.T, self.iT = define_transforms()
        self.rootdir = opt.dataroot

        self.israw = True
        if opt.no_predemosaic:
            assert opt.input_nc == 4, "The the input should be 4 channels!!!"
        else:
            assert opt.input_nc == 3, "The the input should be 3 channels!!!"
        self.ftype = opt.bit_depth

        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc

        # path to folder with gt sequences
        if opt.raw_gt:
            self.gt_paths = os.path.join(self.rootdir, opt.gtFolder)
        else:
            self.gt_paths = os.path.join(self.rootdir, opt.gt_linear_RGB_Folder)

        # path to folder with noisy sequences
        self.n_paths = os.path.join(self.rootdir, opt.nFolder)

        # generate gt and noisy frames
        if not opt.raw_gt:
            self.raw_linear_RGB_paths = os.path.join(self.rootdir, opt.raw_linear_RGB_Folder)

        # path to folder with flows, warped frames
        if not opt.no_warp:
            self.lbridge = CPPbridge('./build/libBridge.so')
            self.w_paths    = os.path.join(self.rootdir, opt.wFolder   , opt.nFolder, opt.warp_method, 'noisyinputs')
            self.flow_paths = os.path.join(self.rootdir, opt.flowFolder, opt.nFolder, opt.warp_method, 'noisyinputs')

        # paths to all gt sequences
        gt_paths_list = os.scandir(self.gt_paths)
        if opt.videos is None:
            gt_paths_list = [c.path for c in gt_paths_list \
                    if not c.name.startswith('.') and c.is_dir()]
        else:
            opt.videos = opt.videos.split(',')
            gt_paths_list = [c.path for c in gt_paths_list \
                    if not c.name.startswith('.') and c.is_dir() and c.name in opt.videos]
        gt_paths_list = sorted(gt_paths_list)

        # paths to all noisy sequences
        noise_paths_list = os.scandir(self.n_paths)
        if opt.videos is None:
            noise_paths_list = [c.path for c in noise_paths_list \
                    if not c.name.startswith('.') and c.is_dir()]
        else:
            noise_paths_list = [c.path for c in noise_paths_list \
                    if not c.name.startswith('.') and c.is_dir() and c.name in opt.videos]
        noise_paths_list = sorted(noise_paths_list)

        assert (len(gt_paths_list) == len(noise_paths_list))

        # paths to all warped sequences
        if not opt.no_warp:
            warped_paths_list = [os.path.join(self.w_paths, os.path.basename(d)) \
                                 for d in gt_paths_list ]


        print ('%d videos' % len(gt_paths_list))
        self.gt_paths_list = gt_paths_list
        self.noise_paths_list = noise_paths_list

        self.patch_depth = self.opt.patch_depth
        self.future_patch_depth = self.opt.future_patch_depth

        self.length = 0
        self.where = []
        self.videos_noisy_path = []
        self.videos_gt_path = []
        self.videos_w_path = []
        self.videos_flow_path = []

        # generate flows and warped frames
        if not opt.no_warp:
            self.createWarpedInputData      (gen_warp=opt.warpeddata)
            self.createFutureWarpedInputData(gen_warp=opt.warpeddata)

        # get paths to frames
        PD = self.patch_depth
        FD = self.future_patch_depth
        for v, (gt_video_path, n_video_path) in enumerate(zip(self.gt_paths_list,
                                                              self.noise_paths_list)):

            # list gt and noisy frames in the sequence
            gt_img_paths  = list_video_files_at_dir(gt_video_path)
            n_img_paths = list_video_files_at_dir(n_video_path)
            assert (len(gt_img_paths) == len(n_img_paths))

            self.where = np.concatenate( (self.where,
                                          np.array(range(len(gt_img_paths) - PD - FD + 1), dtype=np.int) \
                                          + len(self.videos_gt_path) ) )
            self.videos_noisy_path = self.videos_noisy_path + n_img_paths
            self.videos_gt_path    = self.videos_gt_path    + gt_img_paths

            if not self.opt.no_warp:
                for p, n_img_path in enumerate(n_img_paths):
                    w_path = []
                    f_path = []

                    toCode = os.path.splitext(os.path.basename(n_img_path))[0]

                    wfolder = os.path.join(self.w_paths   , pathdiff(n_img_paths[p], self.n_paths))
                    ffolder = os.path.join(self.flow_paths, pathdiff(n_img_paths[p], self.n_paths))

                    # generate filenames of past and future frames warped to frame p
                    for z in range( max(p - PD + 1,0), min(p + FD + 1, len(n_img_paths)) ):

                        # z is frame p: in this case we use the noisy frame p
                        if p == z:
                            w_path.append(n_img_path)
                            continue

                        # paths to warp/flow between frame z and frame p
                        fromCode = os.path.splitext(os.path.basename(n_img_paths[z]))[0]
                        w_path.append(warpedimagefile(wfolder, fromCode, toCode))
                        f_path.append(warpedimagefile(ffolder, fromCode, toCode))

                    self.videos_w_path.append( w_path )
                    self.videos_flow_path.append( f_path )

        self.where = self.where.astype(int)

    def __len__(self):
        return len(self.where)

    def prepare_epoch(self):
        """Load and randomize data"""
        print("nothing to do in prepare_epoch")

    def data_num_channels(self):
        return 3

    def __getitem__(self, index):
        """
        Returns a dictionary that contains gt, n, gt_path, n_path
            gt (tensor) - - Groundtruth images
            n (tensor) - - Noisy images
            gt_path (str) - - Path to the last image in gt
            n_path (str) - - Path to the last image in n
        """

        key = self.where[index]
        gt =  np.asarray([load_image(self.videos_gt_path[key+k],self.ftype) \
                          for k in range(self.patch_depth)], dtype=np.float32)
        if not self.opt.no_warp:
            flows = np.asarray([iio_read(path).astype(np.float32) if os.path.isfile(path) else \
                                np.zeros(list(gt.shape[1:3]) + list([2]), dtype=np.float32)
                                for path in self.videos_flow_path[key + self.patch_depth - 1]], dtype=np.float32)
            flows = flows.transpose(0, 3, 1, 2)
            flows = torch.from_numpy(flows)
        else:
            flows = []

        noise = np.asarray([load_image(self.videos_noisy_path[key+k],self.ftype)
                            for k in range(self.patch_depth+self.future_patch_depth)], dtype=np.float32)

        gt = gt.transpose(0, 3, 1, 2)
        gt = gt.reshape([gt.shape[0] * gt.shape[1], gt.shape[2], gt.shape[3]])
        gt = gt.transpose(1, 2, 0)

        noise = noise.transpose(0, 3, 1, 2)
        noise = noise.reshape([noise.shape[0] * noise.shape[1],
                               noise.shape[2], noise.shape[3]])
        noise = noise.transpose(1, 2, 0)

        noise = self.T(noise)

        # crop
        if hasattr(self.opt, "crop_data") and not self.opt.crop_data is None:
            x, y = [int(s) for s in self.opt.crop_data.split(',')]
            noise = noise[:,:x,:y]
            gt = gt[:x,:y,:] if self.opt.raw_gt else gt[:2*x,:2*y,:]

        output_data = {'gt': self.T(gt), 'n': noise, 'flow': flows,
                       'gt_path':self.videos_gt_path[key+self.patch_depth-1],
                       'n_path':self.videos_noisy_path[key+self.patch_depth-1]}

        return output_data
