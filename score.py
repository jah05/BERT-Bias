import torch
import argparse
import os
import json

def score(wV, nVs, args):
    nameScores = []
    for nVec in nVs:
        nameScores.append(torch.dot(wV, nVec).item()/(torch.norm(wV).item() * torch.norm(nVec).item()))

    if args.merge == "sum":
        return sum(nameScores)
    elif args.merge == "average":
        return sum(nameScores) / len(nameScores)
    elif args.merge == "max":
        return max(nameScores)
    else:
        print("Bad merge method, quiting...")
        quit()

def sortDict(d):
    d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
    return d

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", default="wikipedia", type=str)
    parser.add_argument("--s", default="6", type=str)
    parser.add_argument("--ste", default="occupation", type=str)
    parser.add_argument("--g1", default="black", type=str)
    parser.add_argument("--g2", default="white", type=str)
    parser.add_argument("--dir", default="D:/", type=str)
    parser.add_argument("--merge", default="average", type=str)
    parser.add_argument("--method", default="z2z", type=str) # other is z to x
    parser.add_argument("--att_file", default="race_att.json", type=str)
    args = parser.parse_args()

    file_path = args.dir + "bert-data/d-%s_s-%s-ste_%s-g1_%s-g2_%s.json" % (args.d, args.s, args.ste, args.g1, args.g2)
    if not os.path.isfile(file_path):
        print(file_path)
        print("File does not exist.")
        quit()

    f = open(file_path, 'r')
    # line = f.readline()[:-59068] + "]}"
    # f.close()
    # f = open(file_path, 'w')
    # f.write(line)
    # f.close()
    # quit()
    data = json.load(f)
    corpus = data["data"]
    f.close()

    f = open(args.att_file, 'r')
    termData = json.load(f)["stereotype"][args.ste]
    f.close()

    word1_scores = {data["key1"]:{}, data["key2"]:{}, "appearances":{}}
    word2_scores = {data["key1"]:{}, data["key2"]:{}, "appearances":{}}

    for attribute in termData[data["key1"]]:
        word1_scores[data["key1"]][attribute] = 0
        word2_scores[data["key1"]][attribute] = 0
        word1_scores["appearances"][attribute] = 0
        word2_scores["appearances"][attribute] = 0
    for attribute in termData[data["key2"]]:
        word1_scores[data["key2"]][attribute] = 0
        word2_scores[data["key2"]][attribute] = 0
        word1_scores["appearances"][attribute] = 0
        word2_scores["appearances"][attribute] = 0

    for i, sentence in enumerate(corpus):
        if sentence["group"] == args.g1:
            if args.method == "z2z":
                if sentence["w"] in termData[data["key1"]]:
                    word1_scores[data["key1"]][sentence["w"]] += score(torch.Tensor(sentence["wzv"]), torch.Tensor(sentence["nzv"]), args)
                    word1_scores["appearances"][sentence["w"]] += 1
                elif sentence["w"] in termData[data["key2"]]:
                    word1_scores[data["key2"]][sentence["w"]] += score(torch.Tensor(sentence["wzv"]), torch.Tensor(sentence["nzv"]), args)
                    word1_scores["appearances"][sentence["w"]] += 1
            else:
                if sentence["w"] in termData[data["key1"]]:
                    word1_scores[data["key1"]][sentence["w"]] += score(torch.Tensor(sentence["wzv"]), torch.Tensor(sentence["nxv"]), args)
                    word1_scores["appearances"][sentence["w"]] += 1
                elif sentence["w"] in termData[data["key2"]]:
                    word1_scores[data["key2"]][sentence["w"]] += score(torch.Tensor(sentence["wzv"]), torch.Tensor(sentence["nxv"]), args)
                    word1_scores["appearances"][sentence["w"]] += 1
        else:
            if args.method == "z2z":
                if sentence["w"] in termData[data["key1"]]:
                    word2_scores[data["key1"]][sentence["w"]] += score(torch.Tensor(sentence["wzv"]), torch.Tensor(sentence["nzv"]), args)
                    word2_scores["appearances"][sentence["w"]] += 1
                elif sentence["w"] in termData[data["key2"]]:
                    word2_scores[data["key2"]][sentence["w"]] += score(torch.Tensor(sentence["wzv"]), torch.Tensor(sentence["nzv"]), args)
                    word2_scores["appearances"][sentence["w"]] += 1
            else:
                if sentence["w"] in termData[data["key1"]]:
                    word2_scores[data["key1"]][sentence["w"]] += score(torch.Tensor(sentence["wzv"]), torch.Tensor(sentence["nxv"]), args)
                    word2_scores["appearances"][sentence["w"]] += 1
                elif sentence["w"] in termData[data["key2"]]:
                    word2_scores[data["key2"]][sentence["w"]] += score(torch.Tensor(sentence["wzv"]), torch.Tensor(sentence["nxv"]), args)
                    word2_scores["appearances"][sentence["w"]] += 1

    word1_scores[data["key1"]] = sortDict(word1_scores[data["key1"]])
    word1_scores[data["key2"]] = sortDict(word1_scores[data["key2"]])
    print("logs/" + "d-%s_s-%s-ste_%s-g_%s" % (args.d, args.s, args.ste, args.g1) + ".json")
    f = open("logs/" + "d-%s_s-%s-ste_%s-g_%s" % (args.d, args.s, args.ste, args.g1) + ".json", 'r')
    try:
        d = json.load(f)
        f.close()
    except json.decoder.JSONDecodeError:
        f.close()
        f = open("logs/" + "d-%s_s-%s-ste_%s-g_%s" % (args.d, args.s, args.ste, args.g1) + ".json", 'w')
        json.dump({}, f)
        d = {}
        f.close()
    f = open("logs/" + "d-%s_s-%s-ste_%s-g_%s" % (args.d, args.s, args.ste, args.g1) + ".json", 'w')
    d[args.ste] = word1_scores
    json.dump(d, f)
    f.close()

    word2_scores[data["key1"]] = sortDict(word2_scores[data["key1"]])
    word2_scores[data["key2"]] = sortDict(word2_scores[data["key2"]])
    f = open("logs/" + "d-%s_s-%s-ste_%s-g_%s" % (args.d, args.s, args.ste, args.g2) + ".json", "r")
    try:
        d = json.load(f)
        f.close()
    except json.decoder.JSONDecodeError:
        f.close()
        f = open("logs/" + "d-%s_s-%s-ste_%s-g_%s" % (args.d, args.s, args.ste, args.g2) + ".json", 'w')
        json.dump({}, f)
        d = {}
        f.close()
    f = open("logs/" + "d-%s_s-%s-ste_%s-g_%s" % (args.d, args.s, args.ste, args.g2) + ".json", "w")
    d[args.ste] = word2_scores
    json.dump(d, f)
    f.close()
