"""
A Decomposable Attention Model for Natural Language Inference
ref: https://arxiv.org/pdf/1606.01933.pdf
"""
import argparse

import pandas as pd
from torchtext import data
from nltk.tokenize import word_tokenize

from attend import Attend


def load_data(args):
    train_fraction = (1.0 - args.val_fraction)
    train_data = pd.read_csv(args.train_data)
    train_data = train_data.iloc[:int(len(train_data) * train_fraction)]
    val_data = train_data.iloc[int(len(train_data) * train_fraction):]
    q1 = list(train_data['question1'].map(str).apply(str.lower))
    q2 = list(train_data['question2'].map(str).apply(str.lower))
    y = list(train_data['is_duplicate'])

    # TODO: remove the iteration for only first 1000 elements
    q1 = [word_tokenize(x) for x in q1[:1000]]
    q2 = [word_tokenize(x) for x in q2[:1000]]

    question_field = data.Field(sequential=True, use_vocab=True,
                                lower=True, fix_length=args.d_in)
    # TODO: include the validation data also here
    question_field.build_vocab(q1 + q2)

    if args.cuda:
        device = 1
    else:
        device = -1

    q1_pad_num = question_field.numericalize(question_field.pad(q1), device=device)
    q2_pad_num = question_field.numericalize(question_field.pad(q2), device=device)
    q1_pad_num = q1_pad_num.transpose(1, 0)
    q2_pad_num = q2_pad_num.transpose(1, 0)

    return q1_pad_num, q2_pad_num, y


def main():
    parser = argparse.ArgumentParser(description="parser for Decomposable Attention")
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--d-in", type=int, default=30)
    parser.add_argument("--cuda", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=100)
    args = parser.parse_args()

    # print command line arguments for sanity
    print("-" * 50)
    for arg in args.__dict__:
        print(str(arg).upper() + ": " + str(args.__dict__[arg]))
    print("-" * 50)

    questions1, questions2, labels = load_data(args)
    print(questions1.size(), questions2.size())

    # TODO: Add l2-regularization for weights in the optimizer

    # TODO: Add embedding layer

    attend_module = Attend(max_length=args.d_in,
                           n_input=args.embed_dim,
                           )



if __name__ == "__main__":
    main()