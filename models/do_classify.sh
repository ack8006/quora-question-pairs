#!/bin/bash
# CLASSIFIER
CMD="python autoencoder_classifier_main.py \
	--vocabsize 35000 \
	--max_sentences 500000  \
        --autoencoder ../runs/cpu_anti_collapse_10.pt \
	--lr 0.001 \
	--mode projections \
	--n_projections 50 \
	--projection_dim 3 \
	--demb 200 \
	--epochs 25 --batchsize 20 \
	--dropout 0.5 \
	--loginterval 20 \
	--save_to classify_debug.pt"

cat do_classify.sh
$CMD

