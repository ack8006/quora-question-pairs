#!/bin/bash
CMD="python autoencoder_main.py \
	--vocabsize 10000 \
	--max_sentences 350000  \
        --supplement ../data/supplemental.csv \
	--max_supplement 750000  \
	--lr 0.001 \
	--cuda \
	--din 30 \
	--dhid 300 \
	--demb 200 \
	--epochs 25 --batches 1000 --batchsize 80 \
	--dropout 0.6 \
	--nlayers 2 \
	--squash_size 150 \
	--noise_stdev 0.01 \
	--sloss_factor 0.6 \
	--sloss_slope 0.6 \
	--sloss_shift 5 \
	--dloss_factor 1.0 \
	--dloss_slope 1 \
	--dloss_shift 2 \
	--seed_size 15 \
	--optimizer \
	--save_to supplemental_5.pt"

cat do.sh
$CMD
