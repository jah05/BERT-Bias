import sys
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
import numpy as np

class BERT_Base:
    def __init__(self, args):
        self.cased = args.cased
        self.k = args.k

        self.model_version = "bert-base-"
        if self.cased:
            self.model_version += "cased"
        else:
            self.model_version += "uncased"

        self.model = BertModel.from_pretrained(self.model_version, output_attentions=True)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_version, do_lower_case=(not self.cased))
        self.corpus = load_dataset(args.base_corpus)["train"]

    def run_model(self, input_ids, token_type_ids):
        attention = self.model(input_ids, token_type_ids=token_type_ids)[-1]
        return attention

    def tokenize(self, sentence):
        inputs = self.tokenizer.encode_plus(sentence, return_tensors='pt', add_special_tokens=True)
        token_type_ids = inputs['token_type_ids']
        input_ids = inputs['input_ids']
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        return input_ids, token_type_ids, tokens

    def analyze(self, word, index, attentions, tokens, scores):
        # analyze each word
        for layer in attentions: # go through layers
            for head in layer: # go through attention heads
                for i, w in enumerate(tokens)




    def findTopK(self, word):
        scores = {} # {"word":[score_sum, occurences]}
        for i, sentence in enumerate(corpus):
            input_ids, token_type_ids, tokens = self.tokenize(sentence)

            word_index = -1
            for i in range(tokens):
                if word == tokens[i]:
                    word_index = i
                    break

            if word_index != -1:
                attention = self.run_model(input_ids, token_type_ids)
                self.analyze(word, word_index, attention, tokens, scores)
