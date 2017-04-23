#!/bin/bash
CMD="python autoencoder_main.py \
	--vocabsize 60000 \
	--max_sentences 10000000  \
	--max_supplement 10000000  \
        --supplement supplemental.csv \
	--lr 0.001 \
	--cuda \
	--din 30 \
	--dhid 200 \
	--demb 200 \
	--epochs 25 --batches 1000 --batchsize 80 \
	--dropout 0.5 \
	--nlayers 1 \
	--embed_size 100 \
	--squash_size 50 \
	--sloss_factor 1.0 \
	--sloss_slope 0.02 \
	--sloss_shift 200 \
	--dloss_factor 20 \
	--dloss_slope 0.05 \
	--dloss_shift 100 \
	--kloss_factor 0.1 \
	--kloss_slope 0.01 \
	--kloss_shift 500 \
	--seed_size 20 \
	--save_to vae_2.pt"


cat do.sh
$CMD
