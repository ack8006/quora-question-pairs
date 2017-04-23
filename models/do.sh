#!/bin/bash
CMD="python autoencoder_main.py \
	--vocabsize 50000 \
	--max_sentences 10000000  \
	--max_supplement 10000000  \
        --supplement supplemental.csv \
	--lr 0.001 \
	--cuda \
	--din 30 \
	--dhid 100 \
	--demb 100 \
	--epochs 25 --batches 1000 --batchsize 80 \
	--dropout 0.6 \
	--nlayers 1 \
	--squash_size 50 \
	--sloss_factor 1.0 \
	--sloss_slope 0.05 \
	--sloss_shift 100 \
	--dloss_factor 2.5 \
	--dloss_slope 0.05 \
	--dloss_shift 100 \
	--kloss_factor 1.0 \
	--kloss_slope 0.01 \
	--kloss_shift 500 \
	--seed_size 25 \
	--save_to vae_1.pt"


cat do.sh
$CMD
