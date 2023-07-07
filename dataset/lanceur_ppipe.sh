#!/bin/bash

root=../checkpoints #change if needed
dossier=$1
iso=$2

for seq in {000..004}; do
    CUDA_VISIBLE_DEVICES="0" python3.6 fwd_ppipe.py --input $root/$dossier/val_visuals/$seq/%08d_denoised.tif --output $root/$dossier/val_visuals/$seq/%08d_processed_pipeline.png --first 3 --last 267 --step 3 --ISO $iso --seq $seq & 
 
    # Wait if too many processes...
    numProcs=`jobs | wc -l`
    if [ "$numProcs" -ge 3 ]; then
      wait
    fi
done
wait

python3.6 compute_psnr_pipeline.py --input $root/$dossier/val_visuals/%03d/%08d_processed_pipeline.png  --ISO $iso --last 264
