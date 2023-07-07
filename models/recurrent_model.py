import torch
from .base_model import BaseModel
import networks
from util.flow_utils import warp, upsample_factor_2
from util.util import psnr
import random
from util.Hamilton_Adam_demo import HamiltonAdam

class recurrentModel(BaseModel):
    """ This class implements a "recurrent" model for denoising a sequence 
        of n-1 previously denoised frames plus the current noisy frame 
        from a video.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can
                               use this flag to add training-specific or
                               test-specific options.

        Returns:
            the modified parser.
        """
        if is_train:
            parser.set_defaults(patch_depth=5, no_val=False, patch_width=136, val_dataset_mode='infer4rec')

        parser.add_argument('--model_patch_depth', type=int, default=2, help='True model patch_depth (should be <= patch_depth).')
        parser.add_argument('--unroll_focus', type=str, default="gradual04_from20", help='Weights of unrolling outputs in loss: [all, ge_j, last, incr_j, gradualj, gradualjj_fromjj, gradunijj, gradunijj_fromjj], where j>0 and jj must have two digits')
        parser.add_argument('--feature_rec', action='store_true', default=False, help='Use features from last layer of previous frame (and previous denoised frame).')
        parser.add_argument('--prev_noisy_frame', action='store_true', default=False, help='Use previous noisy frame instead of the denoised one (non-frame recurrent).')
        parser.add_argument('--warp_raw', action='store_true', default=False, help='Apply warping on the 4 channel raw, even with JDD.')

        return parser

    def __init__(self, opt):
        """Initialize the recurrent class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a
                                 subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        # if only 1 unrolling is done during training, testing will be as a
        # non-recurrent network
        self.training_unrollings = opt.patch_depth - opt.model_patch_depth + 1

        # define loss function (this is the one that will be used during training)
        self.criterionL1 = torch.nn.L1Loss()

        # specify other losses to print during training/validation/testing. The training/test
        # scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['L1', 'PSNR', 'Denoiser']

        # specify the images you want to save/display. The training/test
        # scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['denoised']

        # specify the models you want to save to the disk. The training/test
        # scripts will call <BaseModel.save_networks> and
        # <BaseModel.load_networks>
        self.model_names = ['Denoise']

        # define networks
        network_input_nc = (opt.model_patch_depth + opt.future_patch_depth)*opt.input_nc

        self.netDenoise = networks.define_net_arch(
            network_input_nc, opt.output_nc, opt.netDenoiser, 
            opt.init_type, opt.init_gain, self.gpu_ids,
            NoPF=self.opt.model_patch_depth-1 # for feature recurrence
        )

        # we need this for feature recurrence
        if isinstance(self.netDenoise, torch.nn.DataParallel):
            self._netDenoise = self.netDenoise.module
        else:
            self._netDenoise = self.netDenoise

        if self.isTrain:
            # initialize optimizers; schedulers will be automatically created
            # by function <BaseModel.setup>.
            self.optimizer_Denoise = self.optimizer_fn(
                self.netDenoise.parameters(),
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
                weight_decay=opt.weight_decay
            )
            self.optimizers.append(self.optimizer_Denoise)

        if self.opt.raw_gt and (not self.opt.no_predemosaic):
            self.gt_nc = 4
        else:
            self.gt_nc = self.opt.input_nc
        self.data_nc = 4

        self.hamilton_adam = self.to_device(HamiltonAdam('gbrg'))


    def to_device(self, x):
        return x.to(self.device, non_blocking=self.opt.non_blocking)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.
        """

        self.n  = self.to_device(input['n'])
        self.gt = self.to_device(input['gt'])
        self.image_paths = input['n_path']
        self.first_frame = False if self.isTrain else input['FirstOfVideo']

        if not self.opt.no_warp:
            self.flow = self.to_device(input['flow'])

            # add unrollings dimension in testing for compatibility with training
            if not self.isTrain:
                self.flow = self.flow[:,None]

        # options that require demosaicing
        if not self.opt.no_predemosaic:
            self.n = self.hamilton_adam(self.n)

            if (not self.opt.no_warp) and (not self.opt.warp_raw):
                self.flow = upsample_factor_2(self.flow, multiply_by=2)

        if self.isTrain:
            # information about the training time
            self.epoch = input['epoch']
            self.epoch_iter = input['epoch_iter']
            self.epoch_length = input['epoch_length']

    def warp_frame(self, img1, flow):
        """Warp frame.

        Parameters:
            img1 (tensor): image to be warped
            flow (tensor): optical flow from img1 to img0

        Returns:
            warped (tensor): warped image. If no warping, warped = img1.
        """

        if not self.opt.no_warp:
            # warp frame using flow
            if (not self.opt.no_predemosaic) and self.opt.warp_raw:
                warped, _ = warp(self.hamilton_adam.remosaick(img1), flow, interp="bicubic")
                warped = self.hamilton_adam(warped)
            else:
                warped, _ = warp(img1, flow, interp="bicubic")
        else:
            # case of no warping: nothing to do here
            warped = img1

        return warped

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        B, _, H, W = self.n.size()
        D  = self.opt.model_patch_depth - 1
        TD = self.opt.patch_depth - D if self.isTrain else 1
        fD  = self.opt.future_patch_depth
        C  = self.opt.input_nc
        # B   : batch size
        # TD  : maximum number of unrollings
        # D   : number of previous frames given to the network
        # C   : number of channels of the input images
        # H,W : height and width of the frame

        ###########################################################################
        # NOTE
        # This code considers that the network can take more than one previous frame.
        # D is the number of previous frames given to the network as input.
        # That is, for frame t the denoised frame is computed as follows:
        #
        # denoised[t] = net(denoised[t-D] warped to t,
        #                   ...
        #                   denoised[t-2] warped to t,
        #                   denoised[t-1] warped to t,
        #                   noisy[t])
        #
        # This can be controlled with the command line argument --model_patch_depth
        # (model_patch_depth is the total number of input frames including
        # the current noisy frame, i.e. D+1).
        #
        # Until now we have only used '--model_patch_depth 2' (the current
        # noisy frame and only the previous denoised frame) and therefore D=1.
        #
        #
        # During training the dataloader extract from the videos 3D crops of size
        #
        # patch_width x patch_width x patch_depth
        #
        # patch_depth is the number of frames (or cropped frames) we have
        # available, and it needs to be larger or equal than model_patch_depth.
        #
        # Suppose for example than model_patch_depth = 5, and patch_depth = 10.
        # In that case we can do 6 unrollings (patch_depth - model_patch_depth + 1):
        #
        # Notation: n0 n1 n2 n3 n4 n5 n6 n7 n8 n9 are the noisy frames
        #
        # d4 = net(n0-warped-to-4, n1-warped-to-4, n2-warped-to-4, n3-warped-to-4, n4)
        # d5 = net(n1-warped-to-5, n2-warped-to-5, n3-warped-to-5, d4-warped-to-5, n5)
        # d6 = net(n2-warped-to-6, n3-warped-to-6, d4-warped-to-6, d5-warped-to-6, n6)
        # d7 = net(n3-warped-to-7, d4-warped-to-7, d5-warped-to-7, d6-warped-to-7, n7)
        # d8 = net(d4-warped-to-8, d5-warped-to-8, d6-warped-to-8, d7-warped-to-8, n8)
        # d9 = net(d5-warped-to-9, d6-warped-to-9, d7-warped-to-9, d8-warped-to-9, n9)
        #
        # Once a frame is denoised, it is used in the next unrollings. Else, the noisy
        # frames are used. In order to compute the necessary warpings, we need these
        # optical flows:
        #
        # flow 4 -> 0, flow 4 -> 1, flow 4 -> 2, flow 4 -> 3
        # flow 5 -> 1, flow 5 -> 2, flow 5 -> 3, flow 5 -> 4
        # flow 6 -> 2, flow 6 -> 3, flow 6 -> 4, flow 6 -> 5
        # flow 7 -> 3, flow 7 -> 4, flow 7 -> 5, flow 7 -> 6
        # flow 8 -> 4, flow 8 -> 5, flow 8 -> 6, flow 8 -> 7
        # flow 9 -> 5, flow 9 -> 6, flow 9 -> 7, flow 9 -> 8
        #
        # Thus we need D flows for each unrolling. Let's denote the numer
        # of unrollings by TD = patch_depth - model_patch_depth + 1.
        #
        # The dataloader stores all these flows for a batch of size B in a
        # tensor of dimensions  B x TD x D x 2 x W x H.
        #
        ###########################################################################

        # initialize recurrence
        if self.isTrain \
           or (not self.isTrain and self.training_unrollings == 1) \
           or self.first_frame:

            # initialize recurrence with previous noisy frame
            self.lastden = self.n[:,:D*C]

            if self.opt.feature_rec:
                # initialize feature recurrence with 0
                B, _, H, W = self.n.size()
                self.lastfeat = self._netDenoise.get_rec_nil_features(
                                             B,H,W, device=self.device,
                                             non_blocking=self.opt.non_blocking)

        # determine the number of unrollings
        unrollings = TD
        if self.isTrain:

            # the number of unrollings can change during training; e.g. the
            # gradualii_fromjj trains with one unrolling until epoch jj is
            # reached. This is mainly to speed up the training of recurrent
            # networks.
            if self.opt.unroll_focus[:5] == "gradu" and self.opt.unroll_focus[-7:-2] == "_from":

                # epoch where the recurrent training starts
                epoch_start_rec = float(self.opt.unroll_focus[-2:])

                # number of unrollings
                unrollings = 1 if self.epoch < epoch_start_rec else TD

            # store so that it is available for testing
            self.training_unrollings = unrollings

            # during training, we store the output of the unrollings here
            self.denoised_list = []


        # unrollings
        for a in range(unrollings):

            # collect the previous frames for the network input
            netinput = None
            if self.opt.feature_rec:
                featinput = []
                for f in self.lastfeat:
                    featinput.append( f )

            # warp bth previous frame in buffer
            for b in range(D):
                frame0 = self.n[:,(a+D)*C:(a+D+1)*C,:,:]
                frame1 = self.lastden[:,b*C:(b+1)*C,:,:]
                flow01 = self.flow[:,a,b,:,:,:] if not self.opt.no_warp else None

                # warp frame1
                warped = self.warp_frame(frame1, flow01)

                # warp features 
                if self.opt.feature_rec and (not self.opt.no_warp):
                    for i,f in enumerate(featinput):
                        if not f is None:
                            Bf,Cf,Hf,Wf = f.size()
                            onefC = int(Cf / (self._netDenoise.NoPF))
                            featinput[i][:,b*onefC:(b+1)*onefC,:,:] =  \
                                warp(featinput[i][:,b*onefC:(b+1)*onefC,:,:].clone(),
                                     self.flow[:, a, b, :, :, :], interp="bicubic")[0]

                if netinput is None:
                    # first iteration over the patch depth
                    netinput = warped
                else:
                    # following iterations over the patch depth
                    netinput = torch.cat((netinput, warped), 1)

            # set warped features
            if self.opt.feature_rec:
                self._netDenoise.set_rec_features(featinput)

            # add current noisy patch to the input
            netinput = torch.cat((netinput, self.n[:,(a+D)*C:(a+D+1)*C,:,:]), 1)

            # warp bth future frame in buffer
            for b in range(fD):
                # bth future noise frame
                frame0 = self.n[:,(a+D    )*C:(a+D+1    )*C,:,:]
                frame1 = self.n[:,(a+D+1+b)*C:(a+D+1+b+1)*C,:,:]
                flow01 = self.flow[:,a,D+b,:,:,:] if (not self.opt.no_warp) else None

                # warp frame
                warped = self.warp_frame(frame1, flow01)

                # add to input
                netinput = torch.cat((netinput, warped), 1)

            # denoise frame
            self.denoised = self.netDenoise(netinput)

            if self.isTrain:
                # add denoised frame to list
                self.denoised_list.append( self.denoised )

            # set current frame (denoised or noisy) to be used as previous
            # frame in the next unrolling
            store_frame = self.denoised if not self.opt.prev_noisy_frame else \
                          self.n[:,(a+D)*C:(a+D+1)*C,:,:]
            self.lastden = torch.cat((self.lastden[:,C:,:,:], store_frame.clone()), 1)

            if self.opt.feature_rec:
                # get last features to use them in the next unrolling
                tfeat = self._netDenoise.get_current_features()
                for i,f in enumerate(self.lastfeat):
                    if not f is None:
                        Bf,Cf,Hf,Wf = tfeat[i].size()
                        self.lastfeat[i] = torch.cat( (f[:,Cf:,:,:],  tfeat[i]), 1)

        if self.isTrain:
            # store input of last unrolling (used only for visualization)
            self.netinput = netinput


    def compute_unrolling_weights(self):

        """Compute losses; called in every training and validation iteration"""
        B, _, H, W = self.n.size()
        TD = self.opt.patch_depth - 1
        D  = self.opt.model_patch_depth - 1
        # B   : batch size
        # TD  : maximum number of unrollings
        # D   : number of frames given to the network (we never used more than 2)
        # H,W : height and width of the frame

        # During training, the loss is a weighted sum of the individual losses
        # of the denoised frames for each unrolling. The weights are controlled
        # via the command line flag --unroll_focus.
        #
        # all:     uniform weights                                [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
        # ge[ii]:  only unrolling greater or equal than ii. ge3:  [ 0 ,  0 ,  0 , 1/3, 1/3, 1/3]
        # graduni[ii]_from[jj]: gradual transition from non-recurrent to uniform,
        #                       between epochs jj and ii+jj
        # gradual[ii]_from[jj]: gradual transition from non-recurrent to 90% of weight
        #                       in the last unrolling, between epochs jj and ii+jj
        #
        # See below for more details about the gradual transitions.

        # determine the number of unrollings (read comments in the forward() function)
        unrollings = TD
        if self.opt.unroll_focus[:5] == "gradu" and self.opt.unroll_focus[-7:-2] == "_from":
            epoch_start_rec = int(self.opt.unroll_focus[-2:])
            unrollings = 1 if self.epoch < epoch_start_rec else TD

        if unrollings == 1:
            return torch.ones(1, device=self.device)

        if self.opt.unroll_focus[:2] == "ge":
            a = int(self.opt.unroll_focus[3:])
            weights = torch.zeros(TD, device=self.device)
            weights[a:] = 1
            weights = weights/torch.sum(weights)

        elif self.opt.unroll_focus[:5] == "gradu":
            # Options 'gradual[ii]_from[jj]' and 'graduni[ii]_from[jj]':
            # dynamic weights that change gradually during the ii+1 epochs,
            # between the beginning of epoch jj, and the end epoch jj+ii.
            #
            # Before epoch jj: non-recurrent training weights
            # weights0 = [1, 0, 0, ...., 0] (all weight in the first unrolling)
            #
            # After epoch jj+ii:
            # - graduni: uniform weights 1/TD*[1, 1, ..., 1]
            # - gradual: 90% of the final weights on the last unrolling
            #            the remaining 10% is split uniformly on the
            #            rest of the unrollings
            #
            #            [1/10/(TD-1), 1/10/(TD-1), ..., 9/10]

            # To make the transition very smooth, the weights are updated
            # on every **training iteration**.

            # epoch1 is where the gradual change in the weights starts (jj)
            if self.opt.unroll_focus[-7:-2] == "_from":
                epoch1 = int(self.opt.unroll_focus[-2:])
            else:
                epoch1 = 1

            # epoch2 is where the gradual change in the weights stops (jj+ii)
            epoch2 = float(self.opt.unroll_focus[7:9]) + epoch1

            if self.epoch < epoch1:
                # in this case, a single unrolling is computed. weights is a number
                weights = 1.

            else:
                # initial weights in the transition
                weights0 = torch.zeros(TD, device=self.device)
                weights0[0] = 1.

                # weights from the epoch2 until end of training
                if self.opt.unroll_focus[4:7] == 'uni':
                    # 'graduni': final weights are uniform
                    weights2 = 1./float(TD) * torch.ones(TD, device=self.device)

                    # intermediante weights (at (epoch2+epoch1)/2
                    weights1 = .5*(weights0 + weights2)

                else:
                    # 'gradual': final weights put 90% of weights on last
                    weights2 = .1/float(TD-1)*torch.ones(TD, device=self.device)
                    weights2[TD-1] = .9

                    # intermediante weights at (epoch2+epoch1)/2 are uniform
                    weights1 = 1./float(TD) * torch.ones(TD, device=self.device)

                if self.epoch >= epoch2:
                    # use the final weights
                    weights = weights2

                else:
                    # progress (but between 0 and 2 instead of 0 and 1)
                    tr_progress = 2. * min(1., 1./(epoch2 - epoch1) * (self.epoch - epoch1 + \
                                           float(self.epoch_iter)/float(self.epoch_length)))

                    if tr_progress < 1.:
                        # between 0 and epoch2/2: transition between weights0 and weights1
                        weights = (1. - tr_progress) * weights0 + tr_progress * weights1

                    else:
                        # between epoch2/2 and epoch2: transition between weights1 and weights2
                        tr_progress = tr_progress - 1.
                        weights = (1. - tr_progress) * weights1 + tr_progress * weights2

        else: # 'all' option, corresponding to uniform weights
            weights = 1./float(TD) * torch.ones(TD, device=self.device)


        return weights

    def single_frame_loss(self, den, gt_2):
        # linear loss
        return self.criterionL1(den, gt_2) * self.opt.lambda_L1


    def compute_losses(self):
        """Compute losses; called in every training and validation iteration"""
        if self.isTrain:
            B, _, H, W = self.n.size()
            TD = self.opt.patch_depth - 1
            D  = self.opt.model_patch_depth - 1
            # B   : batch size
            # TD  : maximum number of unrollings
            # D   : number of frames given to the network (we never used more than 2)
            # H,W : height and width of the frame

            # compute weights and determine number of unrollings
            weights = self.compute_unrolling_weights()
            unrollings = len(weights)

            # compute individual losses for the denoised frames produced by the unrollings
            loss_L1_vec, loss_PSNR_vec = [], []
            for a in range(unrollings):

                # skip if unrolling weight is 0
                if weights[a] == 0: continue

                gt_2 = self.gt[:,(a+D)*self.gt_nc:(a+1+D)*self.gt_nc,:,:]
                den  = self.denoised_list[a]

                # compute the loss on the remosaicked images
                if self.opt.raw_gt and (not self.opt.no_predemosaic):
                    den = self.hamilton_adam.remosaick(den)

                loss_L1_vec  .append( self.single_frame_loss(den, gt_2) )
                loss_PSNR_vec.append( psnr(den, gt_2, 2.0) )


            self.loss_L1   = torch.sum( weights * torch.stack(loss_L1_vec  , 0) )
            self.loss_PSNR = torch.sum( weights * torch.stack(loss_PSNR_vec, 0) )

            # store loss to be optimized
            self.loss_Denoiser = self.loss_L1

        else:
            # testing: only compute loss for current frame

            den = self.denoised
            gt_2 = self.gt[:,-self.gt_nc:,:,:] # last groundtruth patch
            
            
            # compute the loss on the remosaicked images
            if self.opt.raw_gt and (not self.opt.no_predemosaic):
                den = self.hamilton_adam.remosaick(den)

            self.loss_L1 = self.single_frame_loss(den,gt_2)
            self.loss_PSNR = psnr(den, gt_2, 2.0)
            self.loss_Denoiser = self.loss_L1

    def backward_Denoiser(self):
        """Compute loss and calculate gradients for netDenoise"""
        self.compute_losses()
        self.loss_Denoiser.backward()

    def optimize_parameters(self):
        """All steps needed to optimize netDenoise in each iteration"""
        self.forward()                      # apply denoiser
        self.optimizer_Denoise.zero_grad()  # set gradients to zero
        self.backward_Denoiser()            # calculate gradients

        self.optimizer_Denoise.step()       # update weights
