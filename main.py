import argparse
import sys
from models import BERT_Base
import torch
import json

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_portion', default=1, type=float)
    parser.add_argument('--k', default=500, type=float)
    parser.add_argument('--model', default="bert-base", type=str)
    parser.add_argument('--base_corpus', default="bookcorpus", type=str)
    parser.add_argument('--cased', default=False, type=str2bool)
    parser.add_argument('--path', default="logs/loghe.txt", type=str)
    args = parser.parse_args()

    if args.model == "bert-base":
        b = BERT_Base(args)
        b.processNames("race_att.json", "black", "white")
        f = open("race_att.json", 'r')
        data = json.load(f)
        stereotypes = data["stereotype"]["pu"]
        names1 = data["names"]["black"]
        names2 = data["names"]["white"]
        b.stereotypeHistogramMulti("race_att.json", stereotypes, "pleasant", "unpleasant", "black", "white", names1, names2, "pu")
