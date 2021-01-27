import sys
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
import numpy as np
import torch
import datetime
import json

class BERT_Base:
    def __init__(self, args):
        self.args = args
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
        if args.base_corpus == "bookcorpus":
            self.corpus = load_dataset(args.base_corpus)["train"]
            self.corpLen = len(self.corpus)
        elif args.base_corpus == "wikipedia":
            self.corpus = open(r"D:\wikipedia_sentences.txt", 'r')
            self.corpLen = 105600162

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

            if int(self.data_portion * self.corpLen) == i:
                break
        scores = self.sortDict(scores)
        self.printTopK(scores)

    def stereotypeHistogram(self, stereotype, key1, key2, word1, word2, sName):
        ster1 = stereotype[key1]
        ster2 = stereotype[key2]

        word1_scores = {key1:{}, key2:{}, "appearances":{}}
        word2_scores = {key1:{}, key2:{}, "appearances":{}}

        for attribute in ster1:
            word1_scores[key1][attribute] = 0
            word2_scores[key1][attribute] = 0
            word1_scores["appearances"][attribute] = 0
            word2_scores["appearances"][attribute] = 0
        for attribute in ster2:
            word1_scores[key2][attribute] = 0
            word2_scores[key2][attribute] = 0
            word1_scores["appearances"][attribute] = 0
            word2_scores["appearances"][attribute] = 0

        for i, sentence in enumerate(self.corpus):
            if i % self.args.skip == 0:
                if self.args.base_corpus == "bookcorpus":
                    sentence = sentence["text"]
                input_ids, token_type_ids, tokens = self.tokenize(sentence)
                index1 = self.findIndex(word1, tokens)
                index2 = self.findIndex(word2, tokens)

                if i % (self.args.skip*10000) == 0:
                    print(i)
                    print(datetime.datetime.now())

                    word1_scores[key1] = self.sortDict(word1_scores[key1])
                    word1_scores[key2] = self.sortDict(word1_scores[key2])
                    f = open("logs/" + word1 + "_scores" + self.args.score_fname + ".json", 'r')
                    try:
                        data = json.load(f)
                        f.close()
                    except json.decoder.JSONDecodeError:
                        f.close()
                        f = open("logs/" + word1 + "_scores" + self.args.score_fname + ".json", 'w')
                        json.dump({}, f)
                        data = {}
                        f.close()
                    f = open("logs/" + word1 + "_scores" + self.args.score_fname + ".json", 'w')
                    data[sName] = word1_scores
                    json.dump(data, f)
                    f.close()

                    word2_scores[key1] = self.sortDict(word2_scores[key1])
                    word2_scores[key2] = self.sortDict(word2_scores[key2])
                    f = open("logs/" + word2 + "_scores" + self.args.score_fname + ".json", "r")
                    try:
                        data = json.load(f)
                        f.close()
                    except json.decoder.JSONDecodeError:
                        f.close()
                        f = open("logs/" + word2 + "_scores" + self.args.score_fname + ".json", 'w')
                        json.dump({}, f)
                        data = {}
                        f.close()
                    f = open("logs/" + word2 + "_scores" + self.args.score_fname + ".json", "w")
                    data[sName] = word2_scores
                    json.dump(data, f)
                    f.close()


                if len(tokens) <= 512:
                    if index1 != -1 and index2 != -1:
                        attention = self.run_model(input_ids, token_type_ids)[0]
                        ll = attention[-1]
                        for head in ll:
                            head[index1] = torch.nn.functional.softmax(head[index1])
                            head[index2] = torch.nn.functional.softmax(head[index2])
                            for j, w in enumerate(tokens):
                                try:
                                    word1_scores[key1][w] += head[index1][j].item()
                                    word1_scores["appearances"][w] += 1
                                except KeyError:
                                    pass
                                try:
                                    word1_scores[key2][w] += head[index1][j].item()
                                    word1_scores["appearances"][w] += 1
                                except KeyError:
                                    pass
                                try:
                                    word2_scores[key1][w] += head[index2][j].item()
                                    word2_scores["appearances"][w] += 1
                                except KeyError:
                                    pass
                                try:
                                    word2_scores[key2][w] += head[index2][j].item()
                                    word2_scores["appearances"][w] += 1
                                except KeyError:
                                    pass
                    elif index1 != -1:
                        attention = self.run_model(input_ids, token_type_ids)[0]
                        ll = attention[-1]
                        for head in ll:
                            head[index1] = torch.nn.functional.softmax(head[index1])
                            for j, w in enumerate(tokens):
                                try:
                                    word1_scores[key1][w] += head[index1][j].item()
                                    word1_scores["appearances"][w] += 1
                                except KeyError:
                                    pass
                                try:
                                    word1_scores[key2][w] += head[index1][j].item()
                                    word1_scores["appearances"][w] += 1
                                except KeyError:
                                    pass
                    elif index2 != -1:
                        attention = self.run_model(input_ids, token_type_ids)[0]
                        ll = attention[-1]
                        for head in ll:
                            head[index2] = torch.nn.functional.softmax(head[index2])
                            for j, w in enumerate(tokens):
                                try:
                                    word2_scores[key1][w] += head[index2][j].item()
                                    word2_scores["appearances"][w] += 1
                                except KeyError:
                                    pass
                                try:
                                    word2_scores[key2][w] += head[index2][j].item()
                                    word2_scores["appearances"][w] += 1
                                except KeyError:
                                    pass
                if int(self.data_portion * self.corpLen) == i:
                    break

        word1_scores[key1] = self.sortDict(word1_scores[key1])
        word1_scores[key2] = self.sortDict(word1_scores[key2])
        f = open("logs/" + word1 + "_scores" + self.args.score_fname + ".json", 'r')
        try:
            data = json.load(f)
            f.close()
        except json.decoder.JSONDecodeError:
            f.close()
            f = open("logs/" + word1 + "_scores" + self.args.score_fname + ".json", 'w')
            json.dump({}, f)
            data = {}
            f.close()
        f = open("logs/" + word1 + "_scores" + self.args.score_fname + ".json", 'w')
        data[sName] = word1_scores
        json.dump(data, f)
        f.close()

        word2_scores[key1] = self.sortDict(word2_scores[key1])
        word2_scores[key2] = self.sortDict(word2_scores[key2])
        f = open("logs/" + word2 + "_scores" + self.args.score_fname + ".json", "r")
        try:
            data = json.load(f)
            f.close()
        except json.decoder.JSONDecodeError:
            f.close()
            f = open("logs/" + word2 + "_scores" + self.args.score_fname + ".json", 'w')
            json.dump({}, f)
            data = {}
            f.close()
        f = open("logs/" + word2 + "_scores" + self.args.score_fname + ".json", "w")
        data[sName] = word2_scores
        json.dump(data, f)
        f.close()

    def stereotypeHistogramMulti(self, src, stereotype, key1, key2, word1, word2, names1, names2, sName):
        ster1 = stereotype[key1]
        ster2 = stereotype[key2]

        word1_scores = {key1:{}, key2:{}, "appearances":{}}
        word2_scores = {key1:{}, key2:{}, "appearances":{}}

        for attribute in ster1:
            word1_scores[key1][attribute] = 0
            word2_scores[key1][attribute] = 0
            word1_scores["appearances"][attribute] = 0
            word2_scores["appearances"][attribute] = 0
        for attribute in ster2:
            word1_scores[key2][attribute] = 0
            word2_scores[key2][attribute] = 0
            word1_scores["appearances"][attribute] = 0
            word2_scores["appearances"][attribute] = 0

        for i, sentence in enumerate(self.corpus):
            if i % self.args.skip == 0:
                if self.args.base_corpus == "bookcorpus":
                    sentence = sentence["text"]
                input_ids, token_type_ids, tokens = self.tokenize(sentence)
                index1s, index1f = self.findIndexL(names1, tokens)
                index2s, index2f = self.findIndexL(names2, tokens)

                if i % (self.args.skip * 10000) == 0:
                    print(i)
                    print(datetime.datetime.now())

                    word1_scores[key1] = self.sortDict(word1_scores[key1])
                    word1_scores[key2] = self.sortDict(word1_scores[key2])
                    f = open("logs/" + word1 + "_scores" + self.args.score_fname + ".json", 'r')
                    try:
                        data = json.load(f)
                        f.close()
                    except json.decoder.JSONDecodeError:
                        f.close()
                        f = open("logs/" + word1 + "_scores" + self.args.score_fname + ".json", 'w')
                        json.dump({}, f)
                        data = {}
                        f.close()
                    f = open("logs/" + word1 + "_scores" + self.args.score_fname + ".json", 'w')
                    data[sName] = word1_scores
                    json.dump(data, f)
                    f.close()

                    word2_scores[key1] = self.sortDict(word2_scores[key1])
                    word2_scores[key2] = self.sortDict(word2_scores[key2])
                    f = open("logs/" + word2 + "_scores" + self.args.score_fname + ".json", "r")
                    try:
                        data = json.load(f)
                        f.close()
                    except json.decoder.JSONDecodeError:
                        f.close()
                        f = open("logs/" + word2 + "_scores" + self.args.score_fname + ".json", 'w')
                        json.dump({}, f)
                        data = {}
                        f.close()
                    f = open("logs/" + word2 + "_scores" + self.args.score_fname + ".json", "w")
                    data[sName] = word2_scores
                    json.dump(data, f)
                    f.close()


                if len(tokens) <= 512:
                    if index1s != -1 and index2s != -1:
                        attention = self.run_model(input_ids, token_type_ids)[0]
                        ll = attention[-1]
                        for head in ll:
                            for ind in range(index1s, index1f+1):
                                head[ind] = torch.nn.functional.softmax(head[ind])
                            for ind in range(index2s, index2f+1):
                                head[ind] = torch.nn.functional.softmax(head[ind])
                            for j, w in enumerate(tokens):
                                if w in word1_scores[key1]:
                                    s = 0
                                    for ind in range(index1s, index1f+1):
                                        s += head[ind][j].item()
                                    s /= (index1f + 1 - index1s)
                                    word1_scores[key1][w] += s
                                    word1_scores["appearances"][w] += 1
                                if w in word1_scores[key2]:
                                    s = 0
                                    for ind in range(index1s, index1f+1):
                                        s += head[ind][j].item()
                                    s /= (index1f + 1 - index1s)
                                    word1_scores[key2][w] += s
                                    word1_scores["appearances"][w] += 1
                                if w in word2_scores[key1]:
                                    s = 0
                                    for ind in range(index2s, index2f+1):
                                        s += head[ind][j].item()
                                    s /= (index2f + 1 - index2s)
                                    word2_scores[key1][w] += s
                                    word2_scores["appearances"][w] += 1
                                if w in word2_scores[key2]:
                                    s = 0
                                    for ind in range(index2s, index2f+1):
                                        s += head[ind][j].item()
                                    s /= (index2f + 1 - index2s)
                                    word2_scores[key2][w] += s
                                    word2_scores["appearances"][w] += 1

                    elif index1s != -1:
                        attention = self.run_model(input_ids, token_type_ids)[0]
                        ll = attention[-1]
                        for head in ll:
                            for ind in range(index1s, index1f+1):
                                head[ind] = torch.nn.functional.softmax(head[ind])
                            for j, w in enumerate(tokens):
                                if w in word1_scores[key1]:
                                    s = 0
                                    for ind in range(index1s, index1f+1):
                                        s += head[index2s][j].item()
                                    s /= (index1f + 1 - index1s)
                                    word1_scores[key1][w] += s
                                    word1_scores["appearances"][w] += 1
                                if w in word1_scores[key2]:
                                    s = 0
                                    for ind in range(index1s, index1f+1):
                                        s += head[index2s][j].item()
                                    s /= (index1f + 1 - index1s)
                                    word1_scores[key2][w] += s
                                    word1_scores["appearances"][w] += 1

                    elif index2s != -1:
                        attention = self.run_model(input_ids, token_type_ids)[0]
                        ll = attention[-1]
                        for head in ll:
                            for ind in range(index2s, index2f+1):
                                head[ind] = torch.nn.functional.softmax(head[ind])
                            for j, w in enumerate(tokens):
                                if w in word2_scores[key1]:
                                    s = 0
                                    for ind in range(index2s, index2f+1):
                                        s += head[index2s][j].item()
                                    s /= (index2f + 1 - index2s)
                                    word2_scores[key1][w] += s
                                    word2_scores["appearances"][w] += 1
                                if w in word2_scores[key2]:
                                    s = 0
                                    for ind in range(index2s, index2f+1):
                                        s += head[index2s][j].item()
                                    s /= (index2f + 1 - index2s)
                                    word2_scores[key2][w] += s
                                    word2_scores["appearances"][w] += 1

                if int(self.data_portion * self.corpLen) == i:
                    break

        word1_scores[key1] = self.sortDict(word1_scores[key1])
        word1_scores[key2] = self.sortDict(word1_scores[key2])
        f = open("logs/" + word1 + "_scores" + self.args.score_fname + ".json", 'r')
        try:
            data = json.load(f)
            f.close()
        except json.decoder.JSONDecodeError:
            f.close()
            f = open("logs/" + word1 + "_scores" + self.args.score_fname + ".json", 'w')
            json.dump({}, f)
            data = {}
            f.close()
        f = open("logs/" + word1 + "_scores" + self.args.score_fname + ".json", 'w')
        data[sName] = word1_scores
        json.dump(data, f)
        f.close()

        word2_scores[key1] = self.sortDict(word2_scores[key1])
        word2_scores[key2] = self.sortDict(word2_scores[key2])
        f = open("logs/" + word2 + "_scores" + self.args.score_fname + ".json", "r")
        try:
            data = json.load(f)
            f.close()
        except json.decoder.JSONDecodeError:
            f.close()
            f = open("logs/" + word2 + "_scores" + self.args.score_fname + ".json", 'w')
            json.dump({}, f)
            data = {}
            f.close()
        f = open("logs/" + word2 + "_scores" + self.args.score_fname + ".json", "w")
        data[sName] = word2_scores
        json.dump(data, f)
        f.close()

    def processNames(self, fname, key1, key2):
        f = open(fname, 'r')
        data = json.load(f)
        f.close()
        names = data["names"]
        name1 = names[key1]
        name2 = names[key2]
        for i in range(len(name1)):
            if isinstance(type(name1[i]), list):
                print("Already processed")
                return
            _, _, tokens = self.tokenize(name1[i])
            name1[i] = tokens[1:-1]

        for i in range(len(name2)):
            _, _, tokens = self.tokenize(name2[i])
            name2[i] = tokens[1:-1]

        f = open(fname, 'w')
        json.dump(data, f)
        f.close()



    def sortDict(self, d):
        # for key in d:
        #     d[key] = d[key][0] / d[key][1]

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

        return -1

    def findIndexL(self, items, array):
        for i in range(len(array)):
            for j in range(len(items)):
                if array[i] == items[j][0]:
                    flag = True
                    wLen = 0
                    for k in range(len(items[j])):
                        if items[j][k] != array[i+k]:
                            flag = False
                            break
                    if flag:
                        return i, i + len(items[j]) - 1


        return -1, -1
