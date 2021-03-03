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
    parser.add_argument('--skip', default=3, type=int)
    parser.add_argument('--score_fname', default='', type=str)
    parser.add_argument('--data_dir', default='D:/bert-data/', type=str)
    args = parser.parse_args()

    print(args)
    if True:
        b = BERT_Base(args)
        # b.processNames("race_att.json", "black", "white")
        f = open("race_att.json", 'r')
        data = json.load(f)
        stereotypes = data["stereotype"]["likeability"]
        names1 = data["names"]["black"]
        names2 = data["names"]["white"]
        b.stereotypeHistogramMulti("race_att.json", stereotypes, "unlikeable", "likeable", "black", "white", names1, names2, "likeability")

        # f = open("gender_att.json", 'r')
        # data = json.load(f)
        # stereotype = data["stereotype"]["sci-art"]
        # key1 = "science"
        # key2 = "art"
        # word1 = "she"
        # word2 = "he"
        # sName = "sci-art"
        # f.close()
        # b.stereotypeHistogram(stereotype, key1, key2, word1, word2, sName)

        # b.processNames("gender_att.json", "male-p", "female-p")
        # f = open("gender_att.json", 'r')
        # data = json.load(f)
        # stereotypes = data["stereotype"]["career-family"]
        # names1 = data["names"]["female-p"]
        # names2 = data["names"]["male-p"]
        # b.stereotypeHistogramMulti("gender_att.json", stereotypes, "family", "career", "female", "male", names1, names2, "career-family")
