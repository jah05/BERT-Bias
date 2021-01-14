import sys
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
import numpy as np
import torch
import datetime

class BERT_Base:
    def __init__(self, args):
        self.cased = args.cased
        self.k = args.k
        self.data_portion = args.data_portion

        self.model_version = "bert-base-"
        if self.cased:
            self.model_version += "cased"
        else:
            self.model_version += "uncased"

        self.model = BertModel.from_pretrained(self.model_version, output_attentions=True)
        self.model.eval()
        self.model.to('cuda')
        self.tokenizer = BertTokenizer.from_pretrained(self.model_version, do_lower_case=(not self.cased))
        self.corpus = load_dataset(args.base_corpus)["train"]

    def run_model(self, input_ids, token_type_ids):
        attention = self.model(input_ids, token_type_ids=token_type_ids)[-1]
        return attention

    def tokenize(self, sentence, sentence2=None):
        if sentence2 == None:
            inputs = self.tokenizer.encode_plus(sentence, return_tensors='pt', add_special_tokens=True).to("cuda")
        else:
            inputs = self.tokenizer.encode_plus(sentence, sentence2, return_tensors='pt', add_special_tokens=True).to("cuda")

        token_type_ids = inputs['token_type_ids'].to("cuda")
        input_ids = inputs['input_ids'].to("cuda")
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        return input_ids, token_type_ids, tokens

    def analyze(self, word, index, attentions, tokens, scores):
        # analyze each word
        layer = attentions[-1]
        for head in layer: # go through attention heads
            head[index] = torch.nn.functional.softmax(head[index])

            for i, w in enumerate(tokens):
                try: # {word:[sum_score, appearances]}
                # sum up instead of average
                    scores[w] = [head[index][i].item() + scores[w][0], scores[w][1]+1]
                except KeyError:
                    scores[w] = [head[index][i].item(), 1]


    def findTopK(self, word):
        scores = {} # {"word":[score_sum, occurences]}
        for i, sentence in enumerate(self.corpus):
            sentence = sentence["text"]
            input_ids, token_type_ids, tokens = self.tokenize(sentence)

            if (i+1) % 10000 == 0:
                print(i)
                print(datetime.datetime.now())
                f = open("loghe.txt", 'w')
                f.write("# word avg_score\n")
                for n, key in enumerate(scores):
                    f.write("%d. %s %f\n" %(n+1, key, scores[key][0]/scores[key][1]))
                    if(i==self.k):
                        break
                f.close()

            if len(tokens) <= 512:
                word_index = -1
                for j in range(len(tokens)):
                    if word == tokens[j]: # optimize for search
                        word_index = j
                        break

                if word_index != -1:
                    attention = self.run_model(input_ids, token_type_ids)[0]
                    self.analyze(word, word_index, attention, tokens, scores)

            if int(self.data_portion * len(self.corpus)) == i:
                break
        scores = self.sortDict(scores)
        self.printTopK(scores)

    def sortDict(self, d):
        for key in d:
            d[key] = d[key][0] / d[key][1]

        d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
        return d

    def printTopK(self, d):
        print("# word avg_score")
        for i, key in enumerate(d):
            print("%d. %s %f" %(i+1, key, d[key]))
            if(i==self.k):
                break

    def findLen(self, length):
        for i, sentence in enumerate(self.corpus):
            sentence = sentence["text"]
            input_ids, token_type_ids, tokens = self.tokenize(sentence)
            if(len(tokens) == length):
                print(i)
                break

    def findIndex(self, item, array):
        for i in range(len(array)):
            if array[i] == item:
                return i
