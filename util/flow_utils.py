import argparse
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import os
from torch.autograd import Variable
from scipy.interpolate import griddata, interpn

def torch_flow(flow):
    flow = np.expand_dims(flow, 0)
    flow = Variable(torch.Tensor(flow))
    flow = flow.permute(0, 3, 1, 2)
    return flow

def torch_image(iio_img_like):
    return torch.unsqueeze(torch.Tensor(iio_img_like.transpose(2,0,1)), 0)

### FUNCTIONS FOR INTERPOLATION

def cubic_interpolation(A, B, C, D, x):
    a,b,c,d = A.size()
    x = x.view(a,1,c,d).repeat(1,b,1,1)
    return B + 0.5*x*(C - A + x*(2.*A - 5.*B + 4.*C - D + x*(3.*(B - C) + D - A)))

def bicubic_interpolation_slow(x, vgrid):
    B, C, H, W = x.size()
    if B>0:
        output = torch.cat( [bicubic_interpolation(x[i:(i+1),:,:,:], vgrid[i:(i+1),:,:,:]) for i in range(B)], 0)
    else:
        output = bicubic_interpolation(x, vgrid)
    return output

def bicubic_interpolation(im, grid):
    B, C, H, W = im.size()
    assert B == 1, "For the moment, this interpolation only works for B=1."

    x0 = torch.floor(grid[0, 0, :, :] - 1).long()
    y0 = torch.floor(grid[0, 1, :, :] - 1).long()
    x1 = x0 + 1
    y1 = y0 + 1
    x2 = x0 + 2
    y2 = y0 + 2
    x3 = x0 + 3
    y3 = y0 + 3

    x0 = x0.clamp(0, W-1)
    y0 = y0.clamp(0, H-1)
    x1 = x1.clamp(0, W-1)
    y1 = y1.clamp(0, H-1)
    x2 = x2.clamp(0, W-1)
    y2 = y2.clamp(0, H-1)
    x3 = x3.clamp(0, W-1)
    y3 = y3.clamp(0, H-1)

    A = cubic_interpolation(im[:, :, y0, x0], im[:, :, y1, x0], im[:, :, y2, x0],
                                 im[:, :, y3, x0], grid[:, 1, :, :] - torch.floor(grid[:, 1, :, :]))
    B = cubic_interpolation(im[:, :, y0, x1], im[:, :, y1, x1], im[:, :, y2, x1],
                                 im[:, :, y3, x1], grid[:, 1, :, :] - torch.floor(grid[:, 1, :, :]))
    C = cubic_interpolation(im[:, :, y0, x2], im[:, :, y1, x2], im[:, :, y2, x2],
                                 im[:, :, y3, x2], grid[:, 1, :, :] - torch.floor(grid[:, 1, :, :]))
    D = cubic_interpolation(im[:, :, y0, x3], im[:, :, y1, x3], im[:, :, y2, x3],
                                 im[:, :, y3, x3], grid[:, 1, :, :] - torch.floor(grid[:, 1, :, :]))
    return cubic_interpolation(A, B, C, D, grid[:, 0, :, :] - torch.floor(grid[:, 0, :, :]))


### WARPING FUNCTION


def warp(x, flow, interp):
    """
    Differentiably warp a tensor according to the given optical flow.

    Args:
        x    : torch.Tensor of dimension [B, C, H, W], image to be warped.
        flow : torch.Tensor of dimension [B, 2, H, W], optical flow
        inter: str, can be 'nearest', 'bilinear' or 'bicubic'
    
    Returns:
        y   : torch.Tensor of dimension [B, C, H, W], image warped according to flow
        mask: torch.Tensor of dimension [B, 1, H, W], mask of undefined pixels in y
    """
    B, C, H, W = x.size()
    yy, xx = torch.meshgrid(torch.arange(H, device=x.device),
                            torch.arange(W, device=x.device))

    xx, yy = map(lambda x: x.view(1,1,H,W), [xx,yy])

    grid = torch.cat((xx, yy), 1).float()
    vgrid = Variable(grid) + flow.to(x.device)

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0*vgrid[:, 0, :, :]/(W-1) - 1.0
    vgrid[:, 1, :, :] = 2.0*vgrid[:, 1, :, :]/(H-1) - 1.0
    mask = (vgrid[:, 0, :, :] >= -1) * (vgrid[:, 0, :, :] <= 1) *\
           (vgrid[:, 1, :, :] >= -1) * (vgrid[:, 1, :, :] <= 1)
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid, padding_mode="border",
                                       mode=interp, align_corners=True)

    mask = mask.unsqueeze(1)
    return output, mask.type(torch.FloatTensor)

### THE WRAPPER OF THE WARPING
def single_warp(iio_img_like, np_flow, interpolation="bicubic", givemask=False):
    """
    Apply the warp function to a single image in typical shape [H, W, C] 
    and return in that same format the warped image.  
    """
    img  = torch_image(iio_img_like)
    flow = torch_flow(np_flow)

    #compute warping
    warped, mask = warp(img, flow, interpolation)

    #convert to numpy array and return output
    iio_output_img_like = warped.cpu().numpy().squeeze().transpose(1,2,0)
    # FIXME shouldn't we do the same for mask?
    if givemask:
        return iio_output_img_like, mask
    else:
        return iio_output_img_like


### THE WRAPPER OF THE FLOW
def compute_flow(iio_img1, iio_img2, flow_type = 'tvl1'):

    from library import CPPbridge
    cpplib = CPPbridge('./build/libBridge.so')

    if flow_type == 'tvl1':
        return cpplib.TVL1_flow(iio_img2, iio_img1)
    else:
        raise TypeError(f"Unknown flow type {flow_type}")


### COMBOS: COMPUTE FLOW, MASK AND WARP
def compute_flow_and_warp(iio_img1, iio_img2, flow_type = 'tvl1', 
                          interpolation = 'bicubic', iio_flow_img1 = None):

    if iio_flow_img1 is None:
        iio_flow_img1 = iio_img1

    from library import CPPbridge
    cpplib = CPPbridge('./build/libBridge.so')

    # compute flow from img2 to flow_img1
    if flow_type == 'tvl1':
        flow = cpplib.TVL1_flow(iio_img2, iio_flow_img1)
    else:
        raise TypeError(f"Unknown flow type {flow_type}")

    # warp img1
    warped, undef_mask = single_warp(iio_img1, flow, interpolation, givemask=True)

    return warped, undef_mask, flow


def upsample_factor_2(downsampled_batch, multiply_by=1.):
    """
    This function upsamples a tensor of shape B,TD,D,2, H, W en B,TD,D,C,2, 2*H, 2*W.
                            a tensor of shape B1,B2,C, H, W en B1,B2,C, 2*H, 2*W
                            a tensor of shape B,C, H, W en B,C, 2*H, 2*W
    It can be used when working with the pre-demosaicking to
    upsample the batch of flows (must be use with multiply_by_2=True)
    """

    *rem_size, C, H, W = downsampled_batch.size()

    upsampled_batch = F.interpolate(downsampled_batch.view(-1,C,H,W),
                                    scale_factor=2, mode="bilinear", align_corners=True
                                   ).view(*rem_size, C, 2*H, 2*W)

    return upsampled_batch * multiply_by


