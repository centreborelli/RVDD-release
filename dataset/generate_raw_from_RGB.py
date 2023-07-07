import argparse
import iio
import numpy as np
import os
from os.path import dirname, join
import torch
import skimage
from skimage.io import imsave, imread



## Parse arguments
parser = argparse.ArgumentParser(description="Generate realistic raw data from sRGB ones")
parser.add_argument("--input_val_dataset"          ,type=str, default=""  ,help='path to input (sRGB) sequences and frames for the validation set')
parser.add_argument("--input_train_dataset"        ,type=str, default=""  ,help='path to input (sRGB) sequences and frames for the train set')
parser.add_argument("--output_val_dataset"         ,type=str, default=""  ,help='path to output (raw) sequences and frames for the validation set')
parser.add_argument("--output_train_dataset"       ,type=str, default=""  ,help='path to output (raw) sequences and frames for the train set')
parser.add_argument("--nb_seq_val"                 ,type=int, default=5   ,help='number of sequences in the validation set')
parser.add_argument("--nb_seq_train"               ,type=int, default=240 ,help='number of sequences in the train set')
parser.add_argument("--ISO"                        ,type=int, default=3200,help='ISO level, either 3200 or 12800')
parser.add_argument("--first"                      ,type=int, default=0   ,help='first index')
parser.add_argument("--last"                       ,type=int, default=498 ,help='last index')
parser.add_argument("--step"                       ,type=int, default=1   ,help='step of index: frames are first, first+step, first+2*step, ...')


args = parser.parse_args()


white_balance = [ [[0.7092, 1.9675, 3.6828],[0.8236, 2.2221, 3.3301]],[[0.7401, 2.1030, 3.6304],[0.7644, 1.9503, 3.5006]],[[0.9161, 2.2784, 3.6755],[0.7601, 2.0998, 3.3854]],[[0.9525, 2.3370, 3.6792],[0.7360, 2.3697, 3.4119]],[[0.9605, 2.3947, 3.4367],[0.7404, 2.3139, 3.5170]],[[0.7609, 2.2813, 3.4110],[0.8902, 2.3187, 3.4019]],[[0.7750, 2.2654, 3.5609],[0.7345, 2.0626, 3.5793]],[[0.8840, 1.9958, 3.6247],[0.8833, 2.0230, 3.3515]],[[0.6943, 2.2786, 3.3560],[0.9260, 2.3253, 3.5130]],[[0.9010, 2.2090, 3.6154],[0.6274, 1.9355, 3.3173]],[[0.7958, 1.9339, 3.4810],[0.9572, 2.2042, 3.6575]],[[0.8399, 2.0257, 3.6682],[0.9359, 2.2613, 3.6852]],[[0.7440, 2.1734, 3.4105],[0.7275, 2.3677, 3.6735]],[[0.5619, 1.9805, 3.4812],[0.8135, 1.9108, 3.6120]],[[0.8667, 2.0147, 3.6875],[0.8300, 1.9923, 3.6988]],[[0.7737, 2.2526, 3.5053],[0.9132, 2.3117, 3.4007]],[[0.7509, 2.0487, 3.3553],[0.6704, 1.9102, 3.6929]],[[0.7212, 2.0658, 3.5201],[0.6869, 2.1378, 3.5632]],[[0.7151, 2.0195, 3.5290],[0.6519, 2.1796, 3.4783]],[[0.8090, 2.3589, 3.5027],[0.6393, 1.9052, 3.6153]],[[0.7448, 1.9092, 3.4494],[0.5803, 2.3618, 3.5934]],[[0.7697, 1.9471, 3.6772],[0.7726, 2.1623, 3.6192]],[[0.6977, 2.1741, 3.3000],[0.8566, 2.0728, 3.6538]],[[0.7005, 2.2215, 3.3929],[0.7252, 2.3532, 3.6297]],[[0.8323, 1.9109, 3.6082],[0.9037, 2.3036, 3.6862]],[[0.9798, 2.2035, 3.4980],[0.8641, 1.9713, 3.4595]],[[0.7984, 2.3540, 3.3481],[0.7381, 2.0972, 3.6256]],[[0.8305, 2.0535, 3.3063],[0.8017, 2.0211, 3.5449]],[[0.7706, 2.3751, 3.5043],[0.6495, 2.1595, 3.5811]],[[0.7892, 1.9688, 3.3180],[0.8423, 2.0606, 3.5152]] ]


## Functions

def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def find_gains(seq, iso):
    if iso == 3200:
        return white_balance[seq][1]
    return white_balance[seq][0]


def inverse_smoothstep(image):
  """Approximately inverts a global tone mapping curve."""
  image = image.permute(1, 2, 0) # Permute the image tensor to HxWxC format from CxHxW format
  image = torch.clamp(image, min=0.0, max=1.0)
  out   = 0.5 - torch.sin(torch.asin(1.0 - 2.0 * image) / 3.0) 
  out   = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
  return out


def gamma_expansion(image):
  """Converts from gamma to linear space."""
  # Clamps to prevent numerical instability of gradients near zero.
  image = image.permute(1, 2, 0) # Permute the image tensor to HxWxC format from CxHxW format
  out   = torch.clamp(image, min=1e-8) ** 2.2
  out   = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
  return out


def apply_ccm(image, ccm):
  """Applies a color correction matrix."""
  image = image.permute(1, 2, 0) # Permute the image tensor to HxWxC format from CxHxW format
  shape = image.size()
  image = torch.reshape(image, [-1, 3])
  image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
  out   = torch.reshape(image, shape)
  out   = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
  return out


def safe_invert_gains(image, rgb_gain, red_gain, blue_gain):
  """Inverts gains while safely handling saturated pixels."""
  image = image.permute(1, 2, 0) # Permute the image tensor to HxWxC format from CxHxW format
  gains = torch.stack((1.0 / red_gain, torch.tensor([1.0]), 1.0 / blue_gain)) / rgb_gain
  gains = gains.squeeze()
  gains = gains[None, None, :]
  gains = gains.cuda()
  out   = image * gains
  out   = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
  return out


def mosaic(image):
  image = image.permute(1, 2, 0) # Permute the image tensor to HxWxC format from CxHxW format
  shape = image.size()
  chan0 = image[0::2, 0::2, 1] #g
  chan1 = image[0::2, 1::2, 2] #b
  chan2 = image[1::2, 0::2, 0] #r
  chan3 = image[1::2, 1::2, 1] #g
  out  = torch.stack((chan0, chan1, chan2, chan3), dim=-1).cuda()
  out  = torch.reshape(out, (shape[0] // 2, shape[1] // 2, 4))
  out  = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
  return out


def single_image_rgb2raw(img, rgb_gain, red_gain, blue_gain):
    #rgb2cam = torch.tensor([[0.6218, 0.2241, 0.1541], [0.1064, 0.6482, 0.2454], [0.084, 0.2383, 0.6777]]).cuda()
    rgb2cam = torch.tensor([[0.95640505, 0.17353177, -0.13219438], [0.14135948, 0.80402001, 0.07771696], [0.05432832, 0.29852577, 0.67210576]]).cuda() #come from the authors of CRVD.


    ## Apply quantization noise ##
    bruit = (np.random.rand(*img.shape)-0.5)
    bruit = bruit.astype(img.dtype)
    img = img + bruit
    #img = np.clip(img,0,255)

    #img = img.transpose(2,0,1) / 255 
    img = img.transpose(2,0,1) / 266 #this option to darken more the raw image 
    img = torch.tensor(img).cuda()
    
    # Approximately inverts global tone mapping.
    image  = inverse_smoothstep(img)
    # Inverts gamma compression.
    image = gamma_expansion(image)
    # Inverts color correction.
    image = apply_ccm(image, rgb2cam)
    ## Approximately inverts white balance and brightening.
    image = safe_invert_gains(image, rgb_gain, red_gain, blue_gain)
    # Clips saturated pixels.
    image = torch.clamp(image, min=0.0, max=1.0)
    # Applies a Bayer mosaic.
    image_mosaic = mosaic(image)

    return image.permute(1,2,0).cpu().numpy(), image_mosaic.permute(1,2,0).cpu().numpy()




## Script
    
make_dirs(dirname(args.output_train_dataset))
make_dirs(dirname(args.output_val_dataset  ))
#create gt and noisy folders
make_dirs(join(args.output_train_dataset, "gt_iso%4d"               %(args.ISO)))
make_dirs(join(args.output_train_dataset, "gt_raw_linear_RGB_iso%4d"%(args.ISO)))
make_dirs(join(args.output_train_dataset, "noisy_iso%4d"            %(args.ISO)))
make_dirs(join(args.output_val_dataset  , "gt_iso%4d"               %(args.ISO)))
make_dirs(join(args.output_val_dataset  , "gt_raw_linear_RGB_iso%4d"%(args.ISO)))
make_dirs(join(args.output_val_dataset  , "gt_RGB_iso%4d"           %(args.ISO)))
make_dirs(join(args.output_val_dataset  , "noisy_iso%4d"            %(args.ISO)))

for seq in range(args.nb_seq_train):

    n, red_gain, blue_gain = find_gains(seq, args.ISO)
    n         = torch.FloatTensor([n])
    red_gain  = torch.FloatTensor([red_gain])
    blue_gain = torch.FloatTensor([blue_gain])
    rgb_gain  = 1.0 / n

    print("Train dataset, sequence %03d"%seq)

    #create gt and noisy folders
    gt_raw_path        = join(args.output_train_dataset, "gt_iso%4d/%03d"               %(args.ISO, seq)) #ground-truth raw domain
    gt_linear_RGB_path = join(args.output_train_dataset, "gt_raw_linear_RGB_iso%4d/%03d"%(args.ISO, seq)) #ground-truth linear RGB domain
    noisy_path         = join(args.output_train_dataset, "noisy_iso%4d/%03d"            %(args.ISO, seq)) #noisy
    make_dirs(gt_raw_path       )
    make_dirs(gt_linear_RGB_path)
    make_dirs(noisy_path        )

    for i in range(args.first, args.last+args.step, args.step):

        img = iio.read(args.input_train_dataset%(seq,i))
        H,W,_ = img.shape
        img = img[:2*(H//2), :2*(W//2),:]
        linear_raw_RGB, unprocess_img = single_image_rgb2raw(img, rgb_gain, red_gain, blue_gain)
        #put in 12bits range with black level 240
        linear_raw_RGB = linear_raw_RGB * (4095-240) + 240
        unprocess_img  = unprocess_img  * (4095-240) + 240 

        #apply an affine transformation that maps the 1st and 99th percentile to those of CRVD. After a first generation, the 1st and 99th percentile were respectively 245 and 2305. 
        if args.ISO==3200:
            linear_raw_RGB = (3610-266)*(linear_raw_RGB-245) / (2305-245) + 266
            unprocess_img  = (3610-266)*(unprocess_img -245) / (2305-245) + 266 
        else:
            linear_raw_RGB = (4075-268)*(linear_raw_RGB-245) / (2305-245) + 268
            unprocess_img  = (4075-268)*(unprocess_img -245) / (2305-245) + 268

        #save gt in the linear RGB and the raw domain
        imsave(join(gt_linear_RGB_path, "%08d.tiff"%i), np.round(linear_raw_RGB).clip(0, 4095).astype(np.uint16), check_contrast=False)
        iio.write(join(gt_raw_path    , "%08d.tiff"%i), unprocess_img)

        #add noise to the GT
        if args.ISO==3200:
            noisy = unprocess_img + np.sqrt( (8.0034 *unprocess_img-2043.51144).clip(0, np.inf) )*np.random.randn(*unprocess_img.shape).astype(np.float32)
        else:
            noisy = unprocess_img + np.sqrt( (28.3015*unprocess_img-6307.62081).clip(0, np.inf) )*np.random.randn(*unprocess_img.shape).astype(np.float32)

        #save noisy data
        iio.write(join(noisy_path, "%08d.tiff"%i), noisy)
        





for seq in range(args.nb_seq_val):
    
    n, red_gain, blue_gain = find_gains(seq, args.ISO)
    n         = torch.FloatTensor([n])
    red_gain  = torch.FloatTensor([red_gain])
    blue_gain = torch.FloatTensor([blue_gain])
    rgb_gain  = 1.0 / n
    
    
    print("Validation dataset, sequence %03d"%seq)
    
    #create gt and noisy folders
    gt_raw_path        = join(args.output_val_dataset, "gt_iso%4d/%03d"               %(args.ISO, seq)) #ground-truth raw domain
    gt_linear_RGB_path = join(args.output_val_dataset, "gt_raw_linear_RGB_iso%4d/%03d"%(args.ISO, seq)) #ground-truth linear RGB domain
    gt_RGB_path        = join(args.output_val_dataset, "gt_RGB_iso%4d/%03d"           %(args.ISO, seq)) #ground-truth sRGB domain
    noisy_path         = join(args.output_val_dataset, "noisy_iso%4d/%03d"            %(args.ISO, seq)) #noisy
    make_dirs(gt_raw_path       )
    make_dirs(gt_linear_RGB_path)
    make_dirs(gt_RGB_path       )
    make_dirs(noisy_path        )

    for i in range(args.first, args.last+args.step, args.step):

        img = iio.read(args.input_val_dataset%(seq, i))
        H,W,_ = img.shape
        img = img[:2*(H//2), :2*(W//2),:]
        linear_raw_RGB, unprocess_img = single_image_rgb2raw(img, rgb_gain, red_gain, blue_gain)
        linear_raw_RGB = linear_raw_RGB * (4095-240) + 240
        unprocess_img  = unprocess_img  * (4095-240) + 240
        
        #apply an affine transformation that maps the 1st and 99th percentile to those of CRVD. After a first generation, the 1st and 99th percentile were respectively 245 and 2305. 
        if args.ISO==3200:
            linear_raw_RGB = (3610-266)*(linear_raw_RGB-245) / (2305-245) + 266
            unprocess_img  = (3610-266)*(unprocess_img -245) / (2305-245) + 266 
        else:
            linear_raw_RGB = (4075-268)*(linear_raw_RGB-245) / (2305-245) + 268
            unprocess_img  = (4075-268)*(unprocess_img -245) / (2305-245) + 268 

        #save gt in the linear RGB and the raw domain
        imsave(join(gt_linear_RGB_path, "%08d.tiff"%i), np.round(linear_raw_RGB).clip(0, 4095).astype(np.uint16), check_contrast=False)
        iio.write(join(gt_raw_path    , "%08d.tiff"%i), unprocess_img)

        #post-process the gt in the linear RGB domain to get the gt in the sRGB domain
        from fwd_ppipe import ppipe
        sRGB = ppipe(linear_raw_RGB, rgb_gain, red_gain, blue_gain, args.ISO)
        #save the gt in the sRGB domain, after conversion to uint8 format
        iio.write(join(gt_RGB_path, "%08d.png"%i), sRGB.round().clip(0,255).astype(np.uint8))

        #add noise to the GT
        if args.ISO==3200:
            noisy = unprocess_img + np.sqrt( (8.0034 *unprocess_img-2043.51144).clip(0, np.inf) )*np.random.randn(*unprocess_img.shape).astype(np.float32)
        else:
            noisy = unprocess_img + np.sqrt( (28.3015*unprocess_img-6307.62081).clip(0, np.inf) )*np.random.randn(*unprocess_img.shape).astype(np.float32)

        #save noisy data
        iio.write(join(noisy_path, "%08d.tiff"%i), noisy)
