# RVDD: Recurrent Video Denoising and Demosaicing

Code for training and testing the method RVDD (Recurrent Video Denoising and Demosaicing) described [in the paper](https://openaccess.thecvf.com/content/WACV2023/html/Dewil_Video_Joint_Denoising_and_Demosaicing_With_Recurrent_CNNs_WACV_2023_paper.html):
```
@InProceedings{Dewil_2023_WACV,
    author    = {Dewil, Val\'ery and Courtois, Adrien and Rodr{\'\i}guez, Mariano and Ehret, Thibaud and Brandonisio, Nicola and Bujoreanu, Denis and Facciolo, Gabriele and Arias, Pablo},
    title     = {Video Joint Denoising and Demosaicing With Recurrent CNNs},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    pages     = {5108-5119}
}
```




## Compilation and Requirements

The code is based on PyTorch and also uses C implementation of the TV-L1 optical flow for frame alignment.
It requires the python packages listed in `requirements.txt`

For compiling the C code use:
```bash
mkdir -p build && cd build && cmake .. && make
```

## Usage: dataset generation

We used the [REDS 120fps dataset](https://seungjunnah.github.io/Datasets/reds.html) and apply a temporal downsampling by using one frame out of every three. We generated a synthetic realistic RAW dataset from sRGB one by using an inverse camera pipeline based on [this paper](https://github.com/timothybrooks/unprocessing). The parameters of the inverse camera pipeline where tuned to simulate the raw videos of the [CRVD dataset](https://github.com/cao-cong/RViDeNet), for two ISO levels 3200 and 12800. We used the heteroscedastic Gaussian noise model. 

The [REDS 120fps dataset](https://seungjunnah.github.io/Datasets/reds.html) dataset is divided in 15 parts for the training and 2 parts for the validation. Unzip all the training and the validation parts on a same `train` and `validation` folder. The script `generate_raw_from_RGB.py` which generates the synthetic raw dataset is in the folder `dataset`. The following example shows how to use it: 

**Example:**
```bash
python3 generate_raw_from_RGB.py \
  --input_val_dataset path/to/RGB/validation/data/%03d/%08d.png \
  --input_train_dataset /path/to/RGB/train/data/%03d/%08d.png \
  --output_train_dataset /path/to/train/dataroot/ \
  --output_val_dataset /path/to/validation/dataroot/ \
  --ISO 3200 --first 0 --last 270 --step 3 --nb_seq_train 240 --nb_seq_val 30
```

For each ISO level, the previous script generates clean ground-truth data (in the linear RGB and raw domain) and noisy data, stored with the following folder structure:
```
dataroot/
 ├── gt_iso3200/                     # clean .raw as 4 channel images (W/2 x H/2 x 4)
 │   ├── seq0/
 │   ├── seq1/
 │   ...
 ├── gt_linear_RGB_iso3200           # clean linear RGB (W x H x 3)
 │   ├── seq0/
 │   ├── seq1/
 │   ...
 ├── gt_RGB_iso3200                  # clean RGB (in the non linear domain) (W x H x 3)
 │   ├── seq0/
 │   ├── seq1/
 │   ...
 └── noisy_iso3200                   # noisy raw with ISO 3200 (W/2 x H/2 x 4)
     ├── seq0/
     ├── seq1/
     ...
```
**Remark:** Our synthetic raw dataset is tailored to model the CRVD dataset. After the inverse camera pipeline, a simple affine transformation is applied to the synthetic raw dataset so that the histogram of this dataset matches the histogram of the CRVD dataset (we match the 1st and 99th percentile of both dataset). For greater accuracy, the histogram of the CRVD dataset was estimated separately for the two ISO. Therefore, there is a ground-truth folder for both ISO: `gt_iso3200` and `gt_iso12800`.

The `seq0/, seq1/, ...` are folders associated to each sequence, with the images are stored as `.tiff` files.



## Usage: training and validation

Use `train.py` to train the different models. Use `validate.py` for testing and validation.
During training a validation is computed at the end of each epoch.

The training/validation code allows to train and validate all models shown in the article (except for the ones with fastDVDnet), such as
```
FRAME RECURRENT                           : denoised_rgb[t] = net(denoised_rgb[t-1], noisy_raw[t])
FRAME RECURRENT W/ FUTURE FRAME           : denoised_rgb[t] = net(denoised_rgb[t-1], noisy_raw[t], noisy_raw[t+1]) 
FRAME & FEATURE RECURRENT                 : denoised_rgb[t] = net(denoised_rgb[t-1], features[t-1], noisy_raw[t])
FRAME & FEATURE RECURRENT W/ FUTURE FRAME : denoised_rgb[t] = net(denoised_rgb[t-1], features[t-1], noisy_raw[t], noisy_raw[t+1])

```
This code also allows non-recurrent networks that take as input raw noisy frames (from t-1 and optionally from t+1).

The `net` can be any of the two **network architectures** discussed in the paper:
- `convunet` a simple U-Net with standard convolutional layers with a fixed number of channels throughout all layers and scales.
- `ConvNeXtUnet` a U-Net using ConvNeXt blocks based [on this paper](https://arxiv.org/abs/2201.03545).

The RVDD method requires the TV-L1 optical flow to align frames (and features) at `t-1` and `t+1` to `t`. To speed up computation (particularly during training), these optical flows are computed off-line before training/validation begins and stored as part of the dataset. This is done automatically the first time the training code is run (this can take minutes to hours, depending on the size of the dataset). The computed optical flow are stored in the training dataset folders. 

### Training

The following command shows the usage of `train.py`:
```
python3 train.py \
   --netDenoiser convunet \
   --dataroot /path/to/training/dataroot/ \
   --val_dataroot /path/to/validation/dataroot/ \
   --iso 3200 \
   --checkpoints_dir checkpoints \
   --suffix rvdd-basic-iso3200
```
This trains the RVDD-basic network for ISO 3200 (see article), and the following command trains the RVDD network:
```
python3 train.py \
   --feature_rec \
   --netDenoiser convunet+feat \
   --future_patch_depth 1 \
   --dataroot /path/to/training/dataroot/ \
   --val_dataroot /path/to/validation/dataroot/ \
   --iso 3200 \
   --checkpoints_dir checkpoints \
   --suffix rvdd-iso3200
```
In both cases, the recurrency is trained with 4 unrollings. Training commands for the other architectures are provided in the `scripts/` folder.

The training code will create a folder in the `checkpoints_dir` where the produced checkpoints will be stored along with other log files. The name of the folder consists of an automatically generated prefix and `--suffix` given.
These are the contents of the output folder after 32 epochs of training
```
1_net_Denoise.pth           # checkpoint after epoch 1
2_net_Denoise.pth           # checkpoint after epoch 2
...
32_net_Denoise.pth          # checkpoint after epoch 32
latest_net_Denoise.pth      # latest checkpoint saved
loss_log.txt                # log of the training losses
opt_train.txt               # log of all the options of the training
status.pkl                  # dictionary with training status (epoch, lr, etc) 
                            # at the last completed epoch. Used by autoresume.
val_visuals/                # folder with the results of the latest epoch on the validation videos
```


For more information about training and validation see:
```
python3 train.py --help
python3 validate.py --help
```



### Testing

The script `validate.py` can be used to compute JDD results over a dataset of raw images.


### Process the result with the post-processing pipeline (from linear RGB to sRGB)

The post-processing pipeline is based on the invert camera pipeline used for generating the realistic RAW data.
The code is in the folder `dataset`. The following command shows the usage of the script `fwd_ppipe.py`:

```
python3 fwd_ppipe.py --validation_path path/to/validation/dataroot \
                     --result_folder path/to/your/checkpoints/folder/val_visuals \
                     --videos 000,001,002,003,004 \
                     --first 3 --last 267 --step 3 \
                     --ISO 3200 
```

Note that the `--videos` option is used to specify some specific sequences. If you don't use this option, the script expects to process the 30 sequences of the validation set.

It will create the frames after the post-processing pipeline in the same subfolder as the results of the network.


## Organization of the code

Main functions are in the root folder. There are folders for `dataloader` objects,
`models` and command line `options` (though some options might be added in the 
main function files).
```
.
├── libBridge.cpp                        # Python wrapper around C code
├── library.py                           # Frequently used functions and wrappers
├── README.md                            # this README file
├── train.py
├── validate.py
├── run-train-*.sh                       # example scripts to train several networks
├── run-test*.sh                         # example scripts to test several pre-trained networks
├── ...
├── scripts/                             # scripts to reproduce the results in the paper
├── trained-nets/                        # trained networks
├── 3rdparty/                            # TVL1 flow and MLRI demosaicing code for post-processing pipeline
├── dataset/                             # dataset generation and post-processing pipeline
├── data/                                # dataloaders
├── models/                              # models (recurrent only)
├── networks/                            # network architectures
├── options/                             # command line options
└── util/                                # utilities
```


## About

This source code is based on the following repo:
[https://github.com/VSAnimator/stgan](https://github.com/VSAnimator/stgan)

The following libraries are also included as part of the code:
* For computing the optical flow: [the IPOL implementation](http://www.ipol.im/pub/art/2013/26/) of the [TV-L1 optical flow method of Zack et al.](https://link.springer.com/chapter/10.1007/978-3-540-74936-3_22).
* For image I/O: [Enric Meinhardt's iio](https://github.com/mnhrdt/iio).


