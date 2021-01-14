import argparse
import sys
from models import BERT_Base
import torch

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_portion', default=0.1, type=float)
    parser.add_argument('--k', default=500, type=float)
    parser.add_argument('--model', default="bert-base", type=str)
    parser.add_argument('--base_corpus', default="wikipedia", type=str)
    parser.add_argument('--cased', default=False, type=str2bool)
    parser.add_argument('--path', default="logs/loghe.txt", type=str)
    args = parser.parse_args()

    if args.model == "bert-base":
        b = BERT_Base(args)
        b.findTopK("he")
        # _, _, tokens = b.tokenize("Ma name is jamesd", "nice to meet yo")
        # print(tokens)
        # b.findLen(515)
