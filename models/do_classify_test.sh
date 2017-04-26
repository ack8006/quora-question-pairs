#!/bin/bash
# CLASSIFIER
CMD="python autoencoder_classifier_main.py \
	--vocabsize 10000 \
	--max_sentences 50000  \
        --autoencoder debug.pt \
	--lr 0.001 \
	--mode projections \
	--n_projections 50 \
	--projection_dim 3 \
	--demb 50 \
	--epochs 25 --batchsize 20 \
	--dropout 0.5 \
	--loginterval 20 \
	--debug \
	--save_to classify_debug.pt"

cat do_classify_test.sh
$CMD
