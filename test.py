import sys
from transformers import BertTokenizer, BertModel
import datasets

dataset = datasets.load_dataset('wikipedia','20200501.aa', beam_runner='DirectRunner')
