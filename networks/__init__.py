import torch 
from torch import nn 
from torch.nn import init
from torch.optim import lr_scheduler 

from functools import partial 
from distutils.util import strtobool

from networks.unet import get_UNet_cls
from networks.new_unet import NewUNet, NewUNet_feat


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """

    if opt.lr_policy == 'linear':
        lambda_rule = lambda epoch: 1.0 - max(0, epoch + 1 - opt.niter) / float(opt.niter_decay + 1)
        
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, 
            step_size=opt.lr_decay_iters, 
            gamma=0.1
        )

    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.2, 
            threshold=0.01, 
            patience=5
        )

    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=opt.niter, 
            eta_min=0
        )

    else:
        return NotImplementedError(f"Learning rate policy {opt.lr_policy} not found")

    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__

        if hasattr(m, 'weight') and ("Conv" in classname or "Linear" in classname):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)

            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)

            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)

            else:
                raise NotImplementedError(f"Initialization method {init_type} is not implemented")

            if hasattr(m, 'bias') and m.bias is not None:
                init.zeros_(m.bias.data)

    print(f"Network initialized with {init_type} and gain={init_gain}")
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs

    if init_gain is not None:
        init_weights(net, init_type, init_gain=init_gain)

    return net


def define_net_arch(input_nc, output_nc, netG, init_type='normal', init_gain=0.02, gpu_ids=[], NoPF=-1):
    """Creates and sets up the denoiser network

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a ready-to-use denoiser network which has been initialized by <init_net>.
    """

    net = None

    if "newunet" in netG:
        # Usage: newunet-arg1=val1-arg2=val2...

        kwargs = parse_kwargs(netG)
        model_cls = NewUNet

        if "mode" in kwargs:
            if kwargs['mode'] == 'feat':
                model_cls = NewUNet_feat
            del kwargs['mode']
                
        net = model_cls(input_nc, output_nc, **kwargs)

    elif "convunet" in netG:
        # Usage: convunet-arg1=val1-arg2=val2...

        # load params
        kwargs = parse_kwargs(netG)

        # UNet mode
        mode = "default"

        if "mode" in kwargs:
            mode = kwargs['mode']
            del kwargs['mode']

        unet = get_UNet_cls(mode)

        # Other params
        kwargs['in_channels'] = input_nc
        kwargs['out_channels'] = output_nc
        kwargs['depth'] = 4 # number of scales

        net = unet(**kwargs)

    else:
        raise NotImplementedError(f"Generator model name {netG} is not recognized")

    return init_net(net, init_type, init_gain, gpu_ids)


def parse_kwargs(netG):
    keys = netG.split('-')[1:]
    keys = map(lambda x: x.split('='), keys)
    kwargs = { x: y for (x,y) in keys }

    # type conversion
    for k in kwargs.keys():
        if kwargs[k].isnumeric(): # int conversion
            kwargs[k] = int(kwargs[k])

        elif kwargs[k].lower() == "none": # None conversion
            kwargs[k] = None

        else: # bool conversion
            try:
                kwargs[k] = bool(strtobool(kwargs[k]))
            except:
                pass
    
    return kwargs
