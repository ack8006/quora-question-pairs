#!/bin/bash
CMD="python autoencoder_main.py \
	--vocabsize 40000 \
	--max_sentences 10000000  \
	--max_supplement 10000000  \
        --supplement supplemental.csv \
	--lr 0.001 \
	--cuda \
	--din 30 \
	--dhid 200 \
	--demb 200 \
	--epochs 25 --batches 1000 --batchsize 80 \
	--dropout 0.4 \
	--nlayers 1 \
	--squash_size 100 \
	--sloss_factor 1.0 \
	--sloss_slope 0.6 \
	--sloss_shift 5 \
	--dloss_factor 1.5 \
	--dloss_slope 1 \
	--dloss_shift 5 \
	--kloss_factor 1.0 \
	--kloss_slope 1 \
	--kloss_shift 7 \
	--seed_size 15 \
	--save_to vae_0.pt"


cat do.sh
$CMD
