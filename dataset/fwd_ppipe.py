## PYTHON MODULES
import argparse
import iio
import numpy as np
import os
import torch
from skimage.metrics import structural_similarity


## DATA
#white balance[seq] = [[rgb_gain12800, red_gain12800, blue_gain12800], [rgb_gain3200, red_gain3200, blue_gain3200]]
white_balance = [ [[0.7092, 1.9675, 3.6828],[0.8236, 2.2221, 3.3301]],[[0.7401, 2.1030, 3.6304],[0.7644, 1.9503, 3.5006]],[[0.9161, 2.2784, 3.6755],[0.7601, 2.0998, 3.3854]],[[0.9525, 2.3370, 3.6792],[0.7360, 2.3697, 3.4119]],[[0.9605, 2.3947, 3.4367],[0.7404, 2.3139, 3.5170]],[[0.7609, 2.2813, 3.4110],[0.8902, 2.3187, 3.4019]],[[0.7750, 2.2654, 3.5609],[0.7345, 2.0626, 3.5793]],[[0.8840, 1.9958, 3.6247],[0.8833, 2.0230, 3.3515]],[[0.6943, 2.2786, 3.3560],[0.9260, 2.3253, 3.5130]],[[0.9010, 2.2090, 3.6154],[0.6274, 1.9355, 3.3173]],[[0.7958, 1.9339, 3.4810],[0.9572, 2.2042, 3.6575]],[[0.8399, 2.0257, 3.6682],[0.9359, 2.2613, 3.6852]],[[0.7440, 2.1734, 3.4105],[0.7275, 2.3677, 3.6735]],[[0.5619, 1.9805, 3.4812],[0.8135, 1.9108, 3.6120]],[[0.8667, 2.0147, 3.6875],[0.8300, 1.9923, 3.6988]],[[0.7737, 2.2526, 3.5053],[0.9132, 2.3117, 3.4007]],[[0.7509, 2.0487, 3.3553],[0.6704, 1.9102, 3.6929]],[[0.7212, 2.0658, 3.5201],[0.6869, 2.1378, 3.5632]],[[0.7151, 2.0195, 3.5290],[0.6519, 2.1796, 3.4783]],[[0.8090, 2.3589, 3.5027],[0.6393, 1.9052, 3.6153]],[[0.7448, 1.9092, 3.4494],[0.5803, 2.3618, 3.5934]],[[0.7697, 1.9471, 3.6772],[0.7726, 2.1623, 3.6192]],[[0.6977, 2.1741, 3.3000],[0.8566, 2.0728, 3.6538]],[[0.7005, 2.2215, 3.3929],[0.7252, 2.3532, 3.6297]],[[0.8323, 1.9109, 3.6082],[0.9037, 2.3036, 3.6862]],[[0.9798, 2.2035, 3.4980],[0.8641, 1.9713, 3.4595]],[[0.7984, 2.3540, 3.3481],[0.7381, 2.0972, 3.6256]],[[0.8305, 2.0535, 3.3063],[0.8017, 2.0211, 3.5449]],[[0.7706, 2.3751, 3.5043],[0.6495, 2.1595, 3.5811]],[[0.7892, 1.9688, 3.3180],[0.8423, 2.0606, 3.5152]] ]

inv_ccm = torch.tensor([[1.07955733, -0.40125771, 0.32170038], [-0.15390743, 1.35677921, -0.20287178], [-0.00235972, -0.55155296, 1.55391268]], dtype=torch.float32).cuda()


## FUNCTIONS

def apply_mat_inv_ccm(image, inv_ccm):
    """Applies a color correction matrix."""
    shape = image.size()
    image = torch.reshape(image, [-1, 3])
    image = torch.tensordot(image, inv_ccm, dims=[[-1], [-1]])
    out   = torch.reshape(image, shape)
    return out


def apply_gains(image, rgb_gain, red_gain, blue_gain):
    gains = torch.tensor([1.0 / (red_gain*rgb_gain), 1.0/rgb_gain, 1.0 / (blue_gain*rgb_gain)])
    gains = gains[None, None, :]
    gains = gains.cuda()

    #H, W, _ = image.shape
    #mask = torch.ones(H, W, 1)
    #mask = mask.cuda()

    #safe_gains = torch.max(mask + (1.0 - mask) * gains, gains)
    #out   = image / safe_gains
    out = image / gains
    return out


def find_gains(seq, iso):
    if iso == 3200:
        return white_balance[seq][1]
    return white_balance[seq][0]

def ppipe(im, rgb_gain, red_gain, blue_gain, iso):
    
    #invert matching of 1-99 percentile between REDS and CRVD
    if iso==3200:
        im = (im-266) * (2305-245) / (3610-266) + 245
    if iso==12800:
        im = (im-268) * (2305-245) / (4075-268) + 245
    
    #linearization and BL substraction
    im = (im-240) / (4095-240)

    #CUDA-ization
    im = torch.tensor(im).cuda()

    #apply WB
    im = apply_gains(im, rgb_gain, red_gain, blue_gain)

    #apply CCM
    im = apply_mat_inv_ccm(im, inv_ccm)
    
    #gamma correction
    im[im>10**-8] = im[im>10**-8] ** (1/2.2)

    #tone mapping
    im = 3*im**2 - 2*im**3

    #multiplication by 255
    im = im.cpu().numpy() * 255

    return im

def psnr(img1, img2):
    '''
    Computes the PSNR 'metric' between two images assumed to be in the range [0,1]
    '''
    x = ((np.array(img1/255).squeeze() - np.array(img2/255).squeeze()).flatten())
    return (10*np.log10(1 / np.mean(x**2)))

ssim = lambda x, y : structural_similarity(x.astype(float), y.astype(float), multichannel=True, data_range=255)



## SCRIPT
if __name__ == "__main__":
    ## PARSE ARGUMENTS
    parser_fwd = argparse.ArgumentParser(description="Compute the forward pipeline")
    parser_fwd.add_argument("--validation_path", type=str, default="Path_to_validation_dataset", help='path to the validation dataset')
    parser_fwd.add_argument("--result_folder"  , type=str, default="%03d"                      , help="Path to the 'val_visual' dataroot")
    parser_fwd.add_argument("--videos"         , type=str, default=''                          , help='Which video sequences to use')
    parser_fwd.add_argument("--first"          , type=int, default=3                           , help='Index of the first frame')
    parser_fwd.add_argument("--last"           , type=int, default=264                         , help='Index of the last frame')
    parser_fwd.add_argument("--step"           , type=int, default=3                           , help='Step in the index. The frames will be first, first+step, first+2*step, ...')
    parser_fwd.add_argument("--bit_depth"      , type=int, default=8                           , help='Bit depth of images. Use 8 for images in range [0,255], 12 for images in range [0,4095], etc. If image in range [0,1], use 0')
    parser_fwd.add_argument("--ISO"            , type=int, default=3200                        , help='ISO level')
    
    opt = parser_fwd.parse_args()
    
    if opt.videos=='':
        list_of_sequences = ['%03d'%(i) for i in range(30)]
    else:
        #used for working only with some specific sequences
        list_of_sequences = [int(seq_name) for seq_name in opt.videos.split(',')]
    
    #file where the PSNR will be written
    plot_psnr = open(os.path.join(opt.result_folder, "PSNR.txt"), 'w')
    plot_ssim = open(os.path.join(opt.result_folder, "SSIM.txt"), 'w')
    
    list_psnr, list_ssim = [], []
    
    for seq in list_of_sequences:
        n, red_gain, blue_gain = find_gains(seq, opt.ISO)
        rgb_gain  = 1 /n
        
        for i in range(opt.first, opt.last + opt.step, opt.step):
            print("Processing sequence {:03d} - frame {:03d} / {:03d}".format(seq, i, opt.last))
            img = iio.read(os.path.join(opt.result_folder, "{:03d}/{:08d}_denoised.tif".format(seq, i)))
            #check if data are RGB
            assert (img.shape[-1]==3), "The data should have 3 channels."
        
            #normalize to the range [0,4095]
            if opt.bit_depth == 0:
                img = img * 4095
            elif opt.bit_depth == 8:
                img = img / 255 * 4095
            elif opt.bit_depth == 10:
                img = img / 1024 * 4095
        
            #compute the sRGB frame and put it in the uint8 format
            sRGB = ppipe(img, rgb_gain, red_gain, blue_gain, opt.ISO)
            sRGB = sRGB.round().clip(0,255).astype(np.uint8)
            #save the sRGB frame
            iio.write(os.path.join(opt.result_folder, "{:03d}/{:08d}_processed_pipeline.png".format(seq, i)), sRGB)
    
            #load GT RGB
            gt = iio.read(os.path.join(opt.validation_path, "gt_RGB_iso{:1d}/{:03d}/{:08d}.png".format(opt.ISO, seq, i)))
        
            #compute PSNR and SSIM
            PSNR = psnr(sRGB, gt)
            SSIM = ssim(sRGB, gt)
            list_psnr.append(PSNR)
            list_ssim.append(SSIM)
            plot_psnr.write(str(PSNR) + '\n')
            plot_ssim.write(str(SSIM) + '\n')
    
    #compute average metrics
    average_psnr = np.mean(np.array(list_psnr))
    average_ssim = np.mean(np.array(list_ssim))
    #print and save the averages
    plot_psnr.write("\n")
    plot_psnr.write("\n")
    plot_psnr.write("###  Average: {:4.2f} dB  ###".format(average_psnr))
    plot_ssim.write("\n")
    plot_ssim.write("\n")
    plot_ssim.write("###  Average: {:4.3f}  ###".format(average_ssim))
    print("Average PSNR: {:4.2f}".format(average_psnr))
    print("Average SSIM: {:4.3f}".format(average_ssim))
