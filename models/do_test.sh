#!/bin/bash
CMD="python autoencoder_main.py \
	--vocabsize 10000 \
	--max_sentences 50000  \
        --supplement ../data/supplemental.csv \
	--max_supplement 27000  \
	--lr 0.001 \
	--din 30 \
	--dhid 60 \
	--demb 50 \
	--epochs 25 --batches 400 --batchsize 20 \
	--dropout 0.5 \
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
	--loginterval 20 \
	--seed_size 10 \
        --debug \
	--save_to weight_decay_5.pt"

cat do_test.sh
$CMD
