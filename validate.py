"""General-purpose training script for video denoising
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.flow_utils import compute_flow_and_warp
from util.Hamilton_Adam_demo import HamiltonAdam
from util.visualizer import Visualizer, save_images
import torch
import numpy as np
import copy
from os.path import isfile, dirname, basename, join
from library import load_ordereddict, save_ordereddict, pathdiff, print_dict

def compute_flows_from_denoised(data, model, opt, singleiT):

    # FIXME can't we avoid this? I just want to remosaick
    ha = HamiltonAdam('gbrg')

    # last noisy frame
    img2 = data['n'][0, -4:, :, :] # FIXME: can't we avoid the 4 hardcoded?

    # warp previous frames to align them with img2
    flowinput = []
    for n in range(opt.patch_depth-1):

        # compute optical flow and warp
        img1_flow = model.denoised.squeeze().to('cpu')
        if not opt.no_predemosaic:
            img1_flow = ha.remosaick(img1_flow)

        _, _, flow =  compute_flow_and_warp(singleiT(img1_flow), singleiT(img2),
                                                  opt.warp_method, 'bicubic')
        # add flow to inputs
        flowinput.append(flow)

    data['flow'] = torch.from_numpy( np.array(flowinput).transpose(0, 3, 1, 2) ).unsqueeze(0)

def init_validation_dataloader(opt):
    opt_val = copy.deepcopy(opt)
    opt_val.dataroot         = opt.val_dataroot
    opt_val.dataset_mode     = opt.val_dataset_mode
    opt_val.max_dataset_size = float("inf")
    opt_val.videos           = opt.val_videos
    opt_val.num_threads = 0   # test code only supports num_threads = 1
    opt_val.batch_size = 1    # test code only supports batch_size = 1
    opt_val.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    if hasattr(opt, 'model_patch_depth'):
        opt_val.patch_depth = opt.model_patch_depth

    return create_dataset(opt_val)

def compute_validation(model, val_dataset, val_image_dir, opt):
    # we need transform for a single image
    from library import define_transforms
    _, singleiT = define_transforms(opt)

    # compute optical flow from previously denoised frame
    # else use flows pre-computed from the noisy frames
    # disabled for the in-training validation, since it is slower
    val_flow_from_denoised = False if model.isTrain else opt.val_flow_from_denoised

    bak_isTrain = model.isTrain
    model.isTrain = False
    model.eval() # Correctly set the BN layers before inference

    # init accumulator to compute average loss accross isos
    val_losses = model.get_current_losses().copy()
    for k, _ in val_losses.items(): val_losses[k] = 0.0

    # run validation
    with torch.no_grad():
        lastvideopath = ''
        for i, data in enumerate(val_dataset):
            thisvideopath = dirname(data['gt_path'][0])
            data['FirstOfVideo'] = not thisvideopath == lastvideopath

            # compute optical flows required for warping
            # else, it uses the flows loaded by the dataloader
            if (not opt.no_warp) and val_flow_from_denoised and not data['FirstOfVideo']:
                compute_flows_from_denoised(data, model, opt, singleiT)

            model.set_input(data)
            model.test()
            model.compute_losses()

            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()     # get image paths
            if i % 40 == 0:  # save images to an HTML file
                print('processing (%04d)-th image... %s' % (i, img_path))

            sfolder = pathdiff(img_path[0], val_dataset.dataset.n_paths)
            save_images(val_image_dir, visuals, [basename(img_path[0])],
                        subfolder=sfolder, iT=singleiT)

            # update lastvideopath for next iteration
            lastvideopath = thisvideopath

            # accumulate into total loss (average over val dataset)
            losses = model.get_current_losses()
            print_dict(losses, suffix="", savefile=join(val_image_dir, "output.log"))
            for k, v in losses.items(): val_losses[k] += v

    # average loss
    for k in val_losses.keys(): val_losses[k] /= len(val_dataset)

    val_losses = dict([(k+"_valLoss",v) for k,v in val_losses.items()])
    val_losses['lr'] = model.optimizers[0].param_groups[0]['lr']

    # end of validation, restore training mode
    model.isTrain = bak_isTrain

    return val_losses


if __name__ == '__main__':

    # get options
    opt = TrainOptions().parse()

    # create validation dataset
    val_dataset = init_validation_dataloader(opt)
    print('Number of validation images = %d' % len(val_dataset))

    # define the folder for output images
    val_image_dir = join(opt.checkpoints_dir, opt.name, "val_visuals")

    # CUDNN optimization
    torch.backends.cudnn.benchmark = True

    # create and setup model given opt.model and other options
    model = create_model(opt)
    model.setup(opt)          # regular setup: load and print networks; create schedulers

    # set model in eval mode
    opt.isTrain = False
    model.isTrain = False

    # create a visualizer that display/save images and plots
    visualizer = Visualizer(opt)

    # compute validation
    val_start_time = time.time()
    val_losses = compute_validation(model, val_dataset, val_image_dir, opt)
    print('c --------------------------------------- ')

    # log validation loss
    visualizer.print_current_val_losses(0, val_losses,
            time.time()-val_start_time, len(val_dataset))

