#!/bin/bash

iso=3200
#iso=12800

#######################
# DOWNLOAD TINY DATASET
#######################

# NOTE: this is tiny version of the dataset, consisting on 10 training sequences and
#       5 validation sequences of 25 frames each. To reproduce our results you'll need
#       to use the entire training and validation sets.


cd ..
if [ ! -d datasets/tiny_reds ]; then
	mkdir datasets
	cd datasets
	#wget https://github.com/centreborelli/RVDD/releases/download/untagged-663e5e4df5cdebd4e8ee/tiny_reds.zip .
	unzip tiny_reds.zip
    rm tiny_reds.zip
	cd ..
fi

if [ ! -d datasets/tiny_reds/train/gt_iso$iso ]; then
    cd dataset
    python3 generate_raw_from_RGB.py \
      --input_val_dataset ../datasets/tiny_reds/validation/%03d/%08d.png \
      --input_train_dataset ../datasets/tiny_reds/train/%03d/%08d.png \
      --output_val_dataset ../datasets/tiny_reds/validation \
      --output_train_dataset ../datasets/tiny_reds/train \
      --ISO $iso --first 0 --last 72 --step 3 --nb_seq_train 10 --nb_seq_val 5
    cd ..
fi

python3 validate.py \
    --netDenoiser convunet-mode=fixedfeatures \
	--path2epoch trained-nets/recurrent-convunet-iso$iso \
	--val_dataroot datasets/tiny_reds/validation \
    --gtFolder gt_iso$iso --nFolder noisy_iso$iso --gt_linear_RGB_Folder gt_raw_linear_RGB_iso3200 \
	--suffix jdd-4unrollings-iso$iso \
    --checkpoints_dir checkpoints 
