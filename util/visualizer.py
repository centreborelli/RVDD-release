import numpy as np
import os
import sys
import ntpath
import time
from . import util
from util.util import mkdir
from library import define_transforms
import copy

def save_images(output_images_dir, visuals, image_path, subfolder='',
                iT=None, other_iT=None):
    """Save images to the disk.

    Parameters:
        output_images_dir        -- folder to stores output images
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        width (int)              -- the images will be resized to width x width
    """
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    for label, im_data in visuals.items():
        try:
            im = util.tensor2im(im_data, iT=iT)
        except:
            im = util.tensor2im(im_data, iT=other_iT)

        image_name = '%s_%s.tif' % (name, label)
        save_path = os.path.join(output_images_dir, subfolder, image_name)
        mkdir(os.path.join(output_images_dir, subfolder))
        util.save_image(im, save_path)


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        opt_tmp = copy.deepcopy(opt)
        opt_tmp.patch_depth = 1
        self.T, self.iT = define_transforms(opt_tmp)

        self.name = opt.name
        self.saved = False

        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

    def print_current_val_losses(self, epoch, val_losses, t_comp, len_data, label = ""):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
            label  (string) -- add custom label
        """
        message = '---> validation%s: (epoch: %d, time: %.3f, #data: %d)   [' \
                % (label, epoch,  t_comp, len_data)

        for k, v in val_losses.items():
            message += '%s: %.3f, ' % (k, v)
        message = message[:-2] + "]"

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

