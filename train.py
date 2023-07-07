"""General-purpose training script for video denoising
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer, save_images
from validate import init_validation_dataloader, compute_validation
import torch
import copy
from os.path import isfile, dirname, basename, join
from library import load_ordereddict, save_ordereddict, pathdiff, print_dict


def resume_training(model, opt, statusfile):
    if opt.autoresume and isfile(statusfile):
        status = load_ordereddict( statusfile )
        epoch = status['epoch']
        epoch_to_load = epoch if True else 'latest'
        model.load_networks(epoch_to_load)
        model.update_learning_rate(epoch)
        opt.epoch_count = epoch + 1
    else:
        if opt.autoresume:
            print("---> No status file for resuming!!!")
        opt.epoch_count = 1
        status = {'epoch': opt.epoch_count}
    return status        

if __name__ == '__main__':

    # get training options
    opt = TrainOptions().parse()

    # create training dataset
    train_dataset = create_dataset(opt)
    train_dataset_size = len(train_dataset)
    print('The number of training images = %d' % train_dataset_size)
    
    # create validation dataset
    if not opt.no_val:
        val_dataset = init_validation_dataloader(opt)
        print('Number of validation images = %d' % len(val_dataset))

        # define the folder for output images
        val_image_dir = join(opt.checkpoints_dir, opt.name, "val_visuals")

    # CUDNN optimization
    torch.backends.cudnn.benchmark = True

    # create and setup model given opt.model and other options
    model = create_model(opt)
    model.setup(opt)          # regular setup: load and print networks; create schedulers

    # create a visualizer that display/save images and plots
    visualizer = Visualizer(opt)

    statusfile = model.save_dir + "/status.pkl"
    status = resume_training(model, opt, statusfile)

    # save initialization
    if opt.epoch_count == 1:
        model.save_networks('0')

    # training epoch loop
    total_iters = 0 # init the total number of training iterations
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        model.train()

        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        # inner training loop within one epoch
        for i, data in enumerate(train_dataset):
            iter_start_time = time.time()  # timer for computation per iteration

            # information about training status
            data['epoch'] = epoch
            data['epoch_length'] = len(train_dataset)/opt.batch_size
            data['epoch_iter'] = i

            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            # print training losses and save logging information to the disk
            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)

            iter_data_time = time.time()

        # save model every <save_epoch_freq> epochs
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
            status['epoch'] = epoch
            save_ordereddict(status, statusfile)

        # compute validation
        if not opt.no_val and epoch % opt.val_epoch_freq == 0:

            val_start_time = time.time()
            val_losses = compute_validation(model, val_dataset, val_image_dir, opt)

            # log validation loss
            visualizer.print_current_val_losses(epoch, val_losses,
                    time.time()-val_start_time, len(val_dataset))

            # store checkpoint with current best validation
            if val_losses['Denoiser_valLoss'] < model.best_val_score:
                model.save_networks('latest_val')
                model.best_val_score = val_losses['Denoiser_valLoss']

        # Re-randomize training set
        print('Preparing next epoch')
        train_dataset.prepare_epoch()

        print('End of epoch %d / %d \t Time Taken: %d sec' % \
                (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        # update learning rates at the end of every epoch.
        model.update_learning_rate(epoch)
