# quora-question-pairs

TODO (ALEX):
- Fix embedding freeze ability https://discuss.pytorch.org/t/freeze-the-learnable-parameters-of-resnet-and-attach-it-to-a-new-network/949/11
- Experiment with Vocab Size
- Experiment with word adjustments (numbers -> N, removing stop_words, UNK heuristics, word roots)
- Remove all words in common

- Add Batchnorm
- LSTM multilayer MLP
- LSTMs with: cosine similarity, "angle", "distance"
- LSTM with hand crafted features concatenated on top
- Recurrent CNN Model

TODO(Abhishek):

TODO(Cipta):

TODO (ALL):
- Ensemble Best Models
    - Linear Interpolation
    - Hand Crafted Features concatenated with model predictions and run through model


Hand Crafted Features:
- n words q1, n words q2
- difference in word count
- pct word similarity
- number of common words
- cosine similarity of avrage of word2vex embeddings
- pos tags
- Start with same quesiton word


Evaluation:
- Accuracy
- Precision
- Recall
- F1

