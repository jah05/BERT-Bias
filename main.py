import argparse
import sys
from models import BERT_Base

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_portion', default=0.5, type=float)
    parser.add_argument('--top_k', default=100, type=float)
    parser.add_argument('--model', default="bert-base", type=str)
    parser.add_argument('--base_corpus', default="bookcorpus", type=str)
    parser.add_argument('--cased', default=False, type=str2bool)
    args = parser.parse_args()

    if args.model == "bert-base":
        b = BERT_Base(args)
        attention, words = b.run_model("The cat sat on the mat")
        print(attention[0][0][11][7][7]) # 1 x 12
        print(type(attention[0]))
        print(words)
