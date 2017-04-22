#!/bin/bash
CMD="python autoencoder_main.py \
	--vocabsize 10000 \
	--max_sentences 15000  \
        --supplement ../data/supplemental.csv \
	--max_supplement 7000  \
	--lr 0.001 \
	--din 30 \
	--dhid 50 \
	--demb 50 \
	--epochs 25 --batches 30 --batchsize 20 \
	--dropout 0.5 \
	--nlayers 1 \
	--squash_size 100 \
	--noise_stdev 0.015 \
	--sloss_factor 0.3 \
	--sloss_slope 0.6 \
	--sloss_shift 5 \
	--dloss_factor 1.0 \
	--dloss_slope 1 \
	--dloss_shift 5 \
	--seed_size 15 \
	--optimizer \
        --debug \
	--save_to weight_decay_5.pt"

cat do_test.sh
$CMD
