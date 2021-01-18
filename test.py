from datasets import load_dataset

corpus = load_dataset('wikipedia', '20200501.aa')
print(len(corpus))
