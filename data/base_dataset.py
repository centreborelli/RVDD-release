"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.
"""
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod

from os import scandir
from os.path import join, isfile, basename, splitext
from library import iio_write, list_video_files_at_dir, iio_read, warpedimagefile, pathdiff, load_image
from util.util import mkdir, mkdirs
from util.flow_utils import single_warp, compute_flow_and_warp
import time

class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can
                               use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--frames2load', type=int, default=10, help='Number of frames per video to load to RAM.')
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def prepare_epoch(self):
        """Load and randomize data"""
        print("nothing to do in prepare_epoch")
        pass

    def getWarpInputData(self, idx_video, start_frame):
        """ This function returns the warped data needed
        to match the reference one (the last frame). 
        """
        i = idx_video
        z = start_frame

        videos_w = []
        toCode = splitext( basename(self.videos_noisy_path[i][z+self.patch_depth-1]) )[0]
        wfolder = join(self.w_paths, pathdiff(self.videos_noisy_path[i][z+self.patch_depth-1], self.n_paths))
        for n in range(self.patch_depth-1):
            fromCode = splitext( basename(self.videos_noisy_path[i][z+n]) )[0]
            wimagefile = warpedimagefile(wfolder, fromCode, toCode)
            if isfile(wimagefile):
                warped = load_image(wimagefile,self.ftype)
            else:
                print("warped image %s not found in function getWarpInputData" % wimagefile)
                exit()
            videos_w.append( warped )
        for n in range(self.future_patch_depth):
            fromCode = splitext( basename(self.videos_noisy_path[i][z+self.patch_depth-1+n+1]) )[0]
            wimagefile = warpedimagefile(wfolder, fromCode, toCode)
            if isfile(wimagefile):
                warped = iio_read(wimagefile).astype(np.float32)
            else:
                print("warped image %s not found in function getWarpInputData" % wimagefile)
                exit()
            videos_w.append(warped)
        return np.array(videos_w).astype(np.float32)

    def getFlowInputData(self, idx_video, start_frame):
        """ This function return the flow data needed
        to match the reference frame (the last frame).
        """
        i = idx_video
        z = start_frame

        videos_flow = []
        toCode = splitext( basename(self.videos_noisy_path[i][z+self.patch_depth-1]) )[0]
        flowfolder = join(self.flow_paths, pathdiff(self.videos_noisy_path[i][z+self.patch_depth-1], self.n_paths))
        for n in range(self.patch_depth-1):
            fromCode = splitext( basename(self.videos_noisy_path[i][z+n]) )[0]
            flowimagefile = warpedimagefile(flowfolder, fromCode, toCode)
            if isfile(flowimagefile):
                flow = iio_read(flowimagefile).astype(np.float32)
            else:
                print("flow %s not found in function getFlowInputData" % flowimagefile)
                exit()
            videos_flow.append( flow )
        for n in range(self.future_patch_depth):
            fromCode = splitext( basename(self.videos_noisy_path[i][z+self.patch_depth-1+n+1]) )[0]
            flowimagefile = warpedimagefile(flowfolder, fromCode, toCode)
            if isfile(flowimagefile):
                flow = iio_read(flowimagefile).astype(np.float32)
            else:
                print("flow %s not found in function getFlowInputData" % flowimagefile)
                exit()
            videos_flow.append( flow )
        return np.array(videos_flow).astype(np.float32)

    def createWarpedInputData(self, gen_warp=False):
        """ This function creates (if not existing) all warped data needed
        to match the reference frame (the last frame).
        """
        if not self.opt.check_data:
            return

        start_time = time.time()
        # loop on the videos in the dataset
        for video_path, video2_path in zip(self.gt_paths_list, self.noise_paths_list):
            img2_paths = list_video_files_at_dir(video2_path)

            # loop on the video frames
            for z in range(len(img2_paths)-self.patch_depth+1):

                toCode = splitext( basename(img2_paths[z+self.patch_depth-1]) )[0]

                wfolder = join(self.w_paths   , pathdiff(img2_paths[z+self.patch_depth-1], self.n_paths))
                ffolder = join(self.flow_paths, pathdiff(img2_paths[z+self.patch_depth-1], self.n_paths))
                # w stands for warped image
                # f stands for flow

                mkdir(ffolder)
                if gen_warp: mkdir(wfolder)

                img2 = iio_read(img2_paths[z+self.patch_depth-1]).astype(np.float32)

                # loop on the frames that need to be warped to img2
                for n in range(self.patch_depth-1):

                    # create filename tocode_fromcode
                    fromCode = splitext( basename(img2_paths[z+n]) )[0]

                    wimagefile = warpedimagefile(wfolder, fromCode, toCode)
                    fimagefile = warpedimagefile(ffolder, fromCode, toCode)

                    if not isfile(fimagefile) or \
                            (gen_warp and not isfile(wimagefile)):

                        # load previous noisy frame
                        img1 = iio_read(img2_paths[z+n]).astype(np.float32)

                        # flow doesn't exist, compute flow and warp
                        if not isfile(fimagefile):
                            warped, _, flow = compute_flow_and_warp(img1, img2,
                                                    flow_type=self.opt.warp_method)
                            iio_write(flow.astype(np.float32), fimagefile)

                        # flow exists, but we need warped image
                        elif (gen_warp and not isfile(wimagefile)):
                            flow = iio_read(fimagefile).astype(np.float32)
                            warped = single_warp(img1, flow)

                        # save warped frame
                        if gen_warp and not isfile(wimagefile):
                            iio_write(warped.astype(np.float32), wimagefile)

        print('Warp-Flow-Mask creation/checking: %d sec' % (time.time() - start_time))

    def createFutureWarpedInputData(self, gen_warp=False):
        """ This function creates (if not existing) all warped data needed
        to match the reference frame (the last frame). 
        """
        if (not self.opt.check_data) or self.future_patch_depth==0:
            return

        start_time = time.time()
        for video_path, video2_path in zip(self.gt_paths_list, self.noise_paths_list):
            img2_paths = list_video_files_at_dir(video2_path)

            # loop on the video frames
            for z in range(len(img2_paths)-self.future_patch_depth):

                toCode = splitext( basename(img2_paths[z]) )[0]

                wfolder = join(self.w_paths   , pathdiff(img2_paths[z], self.n_paths))
                ffolder = join(self.flow_paths, pathdiff(img2_paths[z], self.n_paths))
                # w stands for warped image
                # f stands for flow

                mkdir(ffolder)
                if gen_warp: mkdir(wfolder)

                img2 = iio_read(img2_paths[z]).astype(np.float32)

                # loop on the frames that need to be warped to img2
                for n in range(self.future_patch_depth):

                    # create filename tocode_fromcode
                    fromCode = splitext( basename(img2_paths[z+n+1]) )[0]

                    wimagefile = warpedimagefile(wfolder, fromCode, toCode)
                    fimagefile = warpedimagefile(ffolder, fromCode, toCode)

                    if not isfile(fimagefile) or \
                            (gen_warp and not isfile(fimagefile)):

                        # load previous next frame
                        img1 = iio_read(img2_paths[z+n+1]).astype(np.float32)

                        # flow doesn't exist, compute flow and warp
                        if not isfile(fimagefile):
                            warped, _, flow = compute_flow_and_warp(img1, img2,
                                                    flow_type=self.opt.warp_method)
                            iio_write(flow.astype(np.float32), fimagefile)

                        # flow exists, but we need warped image
                        elif (gen_warp and not isfile(wimagefile)):
                            flow = iio_read(fimagefile).astype(np.float32)
                            warped = single_warp(img1, flow)

                        # save warped frame
                        if gen_warp and not isfile(wimagefile):
                            iio_write(warped.astype(np.float32), wimagefile)

        print('Future Warp-Flow-Mask creation/checking: %d sec' % (time.time() - start_time))

