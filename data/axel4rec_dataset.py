import os
import os.path
import numpy as np
import numpy.random
import torch
import random
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image

from library import *

class axel4recDataset(BaseDataset):
    """A dataset class for training! It will provide random 3d patches based on
       axel's way of storing data.
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
        self.opt = opt
        self.rootdir = opt.dataroot

        if opt.raw_gt:
            self.gt_paths = os.path.join(self.rootdir, opt.gtFolder)
        else:
            self.gt_paths = os.path.join(self.rootdir, opt.gt_linear_RGB_Folder)
        self.n_paths = os.path.join(self.rootdir, opt.nFolder)

        self.israw = True
        if not opt.no_predemosaic:
            assert opt.input_nc == 3, "The the input should be 3 channels!!!"
        else:
            assert opt.input_nc == 4, "The the input should be 4 channels!!!"
        self.ftype = opt.bit_depth
        if not opt.raw_gt:
            self.raw_linear_RGB_paths = os.path.join(self.rootdir, opt.raw_linear_RGB_Folder)


        self.lbridge = CPPbridge('./build/libBridge.so')
        self.w_paths    = os.path.join(self.rootdir, opt.wFolder   , opt.nFolder, opt.warp_method, 'noisyinputs')
        self.flow_paths = os.path.join(self.rootdir, opt.flowFolder, opt.nFolder, opt.warp_method, 'noisyinputs')


        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc
        
        gt_paths_list = os.scandir(self.gt_paths)
        if opt.videos is None:
            gt_paths_list = [c.path for c in gt_paths_list \
                    if not c.name.startswith('.') and c.is_dir()]
        else:
            opt.videos = opt.videos.split(',')
            gt_paths_list = [c.path for c in gt_paths_list \
                    if not c.name.startswith('.') and c.is_dir() and c.name in opt.videos]
        gt_paths_list = sorted(gt_paths_list)
        noise_paths_list = os.scandir(self.n_paths)
        if opt.videos is None:
            noise_paths_list = [c.path for c in noise_paths_list \
                    if not c.name.startswith('.') and c.is_dir()]
        else:
            noise_paths_list = [c.path for c in noise_paths_list \
                    if not c.name.startswith('.') and c.is_dir() and c.name in opt.videos]
        noise_paths_list = sorted(noise_paths_list)

        assert (len(gt_paths_list) == len(noise_paths_list))
        
        warped_paths_list = [os.path.join(self.w_paths, os.path.basename(d)) \
                             for d in gt_paths_list ]
        

        print ('%d videos' % len(gt_paths_list))
        self.gt_paths_list = gt_paths_list
        self.noise_paths_list = noise_paths_list

        self.total_depth = opt.patch_depth
        # This only works for axel4rec dataset, that creates opt.model_patch_depth
        self.patch_depth = opt.model_patch_depth
        self.future_patch_depth = opt.future_patch_depth

        self.patch_width = opt.patch_width
        self.patch_stride = opt.patch_stride

        self.createWarpedInputData      (gen_warp = self.opt.warpeddata)
        self.createFutureWarpedInputData(gen_warp = self.opt.warpeddata)
        self.prepare_epoch()

    def prepare_epoch(self):
        """
        Load and randomize data
        """
        self.videos_noisy = []
        self.videos_gt = []
        self.videos_warped = []
        self.videos_flow = []
        self.videos_noisy_path = []
        self.videos_gt_path = []

        num_frames_per_video = self.opt.frames2load

        for (video_path, video2_path) in zip(self.gt_paths_list, self.noise_paths_list):
            img_paths = list_video_files_at_dir(video_path)
            img2_paths = list_video_files_at_dir(video2_path)
            assert (len(img_paths) == len(img2_paths))

            num_frames = len(img_paths)
            start_frame = numpy.random.randint(num_frames-num_frames_per_video+1)
            img_paths = img_paths[start_frame:(start_frame+num_frames_per_video)]
            img2_paths = img2_paths[start_frame:(start_frame+num_frames_per_video)]

            video_gt = np.asarray([load_image(path,self.ftype) \
                                   for path in img_paths], dtype=np.float32)
            video_noisy = np.asarray([load_image(path,self.ftype) \
                                   for path in img2_paths], dtype=np.float32)
            self.videos_noisy_path.append(img2_paths)
            self.videos_gt_path.append(img_paths)
            self.videos_noisy.append(video_noisy)
            self.videos_gt.append(video_gt)

        PD = self.patch_depth
        FD = self.future_patch_depth

        for i, (video_path, video2_path) in enumerate( zip(self.gt_paths_list,
                                                           self.noise_paths_list) ):
            self.videos_flow.append(np.array(
                [self.getFlowInputData(i,z) \
                 for z in range(num_frames_per_video - PD - FD + 1)]).astype(np.float32) )

        if self.opt.warpeddata:
            for i, (video_path, video2_path) in enumerate( zip(self.gt_paths_list,
                                                               self.noise_paths_list) ):
                self.videos_warped.append(np.array(
                    [self.getWarpInputData(i,z) \
                     for z in range(num_frames_per_video - PD - FD + 1)]).astype(np.float32) )

        i = 0
        self.keys = []
        for v in self.videos_noisy:
            zs = range(0, v.shape[0]-self.total_depth - FD + 1, self.patch_stride)
            ys = range(self.patch_width+1, v.shape[1]+1, self.patch_stride)
            xs = range(self.patch_width+1, v.shape[2]+1, self.patch_stride)
            xx, yy, zz = np.meshgrid(xs, ys, zs)
            xx = np.asarray(xx.flatten(), dtype=np.uint32)
            yy = np.asarray(yy.flatten(), dtype=np.uint32)
            zz = np.asarray(zz.flatten(), dtype=np.uint32)
            self.keys.append(np.stack([i*np.ones([len(xx)], dtype=np.uint32), xx, yy, zz]).T)
            i = i + 1

        self.keys = np.concatenate(self.keys, axis=0)
        self.num_keys = self.keys.shape[0]
        self.indices = [i for i in range(self.num_keys)]

        random.shuffle(self.indices)


    def __len__(self):
        return self.num_keys

    def data_num_channels(self):
        return 3

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains gt, n, gt_path, n_path
            gt (tensor) - - Groundtruth 3d patch
            n (tensor) - - Noisy 3d patch
            gt_path (str) - - Image path associated to the last 2d patch in gt
            n_path (str) - - Image path associated to the last 2d patch in n
        """

        key = self.keys[self.indices[index],:]
        patch_width = self.patch_width
        i = key[0]
        x = key[1]
        y = key[2]
        z = key[3]

        # TODO is this necessary?
        if not self.opt.no_predemosaic:
            if (x-patch_width)%2 == 1: x = x - 1
            if (y-patch_width)%2 == 1: y = y - 1

        PD = self.patch_depth
        FD = self.future_patch_depth

        upfactor = 1 if self.opt.raw_gt else 2
        gt = self.videos_gt[i][z:z+self.total_depth,
                               upfactor*(y-patch_width):upfactor*y,
                               upfactor*(x-patch_width):upfactor*x,:]

        noise = self.videos_noisy[i][z:z+self.total_depth + FD,
                                     y-patch_width:y,
                                     x-patch_width:x,:]

        gt = gt.transpose(0, 3, 1, 2)
        gt = gt.reshape([gt.shape[0] * gt.shape[1], gt.shape[2], gt.shape[3]])
        gt = gt.transpose(1, 2, 0)

        noise = noise.transpose(0, 3, 1, 2)
        noise = noise.reshape([noise.shape[0] * noise.shape[1],
                               noise.shape[2], noise.shape[3]])
        noise = noise.transpose(1, 2, 0)

        flows = self.videos_flow[i][z:z+self.total_depth - PD + 1,:,
                                    y-patch_width:y,
                                    x-patch_width:x, :]
        flows = flows.transpose(0, 1, 4, 2, 3)
        flows = torch.from_numpy(flows)

        output_data = {'gt': self.T(gt),
                       'n': self.T(noise),
                       'flow': flows,
                       'gt_path': self.videos_gt_path[i][z+self.total_depth-1],
                       'n_path': self.videos_noisy_path[i][z+self.total_depth-1]}

        # add warped frame
        if self.opt.warpeddata:
            warps = self.videos_warped[i][z:z+self.total_depth - PD + 1,:,
                                          y-patch_width:y,
                                          x-patch_width:x, :]
            warps = warps.transpose(0,1,4,2,3)
            warps = warps.reshape([warps.shape[0] * warps.shape[1] * warps.shape[2],
                                   warps.shape[3], warps.shape[4]])
            warps = warps.transpose(1, 2, 0)
            output_data['warped'] = self.T(warps)

        return output_data

