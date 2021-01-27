import numpy as np
import matplotlib.pyplot as plt
import json
import matplotlib.patches as mpatches
import argparse
import os

def plot2Stereotypes(group, args):
    if group == args.g1:
        _, _, data, _, app, _ = getData(args)
    else:
        _, _, _, data, _, app = getData(args)
    key1, key2 = list(data.keys())[0], list(data.keys())[1]
    data1 = data[key1]
    data2 = data[key2]
    x = np.arange(len(data1) + len(data2))
    width = 0.35

    fig, ax = plt.subplots()
    bars = ax.bar(x, list(data1.values()) + list(data2.values()), width)
    ax.set_title(group + " Score")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(list(data1.keys()) + list(data2.keys()))
    fig.legend(handles=[mpatches.Patch(color='red', label=key1), mpatches.Patch(color='blue', label=key2)])
    for i in range(len(data1)):
        ax.get_children()[i].set_color('r')
    for i in range(len(data1), len(data1) + len(data2)):
        ax.get_children()[i].set_color('b')
    fig.tight_layout()
    fig.savefig("graphs/%s/%s_score.png" %(args.folder, group))

def plotStereotypeAveraged(group, args):
    if group == args.g1:
        _, _, data, _, app, _ = getData(args)
    else:
        _, _, _, data, _, app = getData(args)

    key1, key2 = list(data.keys())[0], list(data.keys())[1]

    data1 = data[key1]
    data2 = data[key2]

    average1 = {}
    average2 = {}

    for key in data1:
        try:
            average1[key] = data1[key] / app[key]
        except ZeroDivisionError:
            average1[key] = 0
    for key in data2:
        try:
            average2[key] = data2[key] / app[key]
        except ZeroDivisionError:
            average2[key] = 0

    average1 = sortDict(average1)
    average2 = sortDict(average2)
    x = np.arange(len(average1) + len(average2))
    width = 0.35

    fig, ax = plt.subplots()
    bars = ax.bar(x, list(average1.values()) + list(average2.values()), width)
    ax.set_title(group + " Score (Averaged)")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(list(average1.keys()) + list(average2.keys()))
    fig.legend(handles=[mpatches.Patch(color='red', label=key1), mpatches.Patch(color='blue', label=key2)])
    for i in range(len(average1)):
        ax.get_children()[i].set_color('r')
    for i in range(len(average1), len(average1) + len(average2)):
        ax.get_children()[i].set_color('b')
    fig.tight_layout()
    fig.savefig("graphs/%s/%s_score_average.png" %(args.folder, group))

def plotSideBySide(args):
    _, _, data1, data2, _, _ = getData(args)
    key1, key2 = list(data1.keys())[0], list(data1.keys())[1]
    x1 = np.arange(len(data1[key1]))
    x2 = np.arange(len(data1[key2]))
    width = 0.35
    fig, ax = plt.subplots(1, 2, sharey=True)

    l1 = []
    l2 = []
    l3 = []
    l4 = []

    for key in data1[key1]:
        l1.append(data1[key1][key])
        l2.append(data2[key1][key])

    for key in data1[key2]:
        l3.append(data1[key2][key])
        l4.append(data2[key2][key])

    rects1 = ax[0].bar(x1 - width/2, l1, width, label=args.g1)
    rects2 = ax[0].bar(x1 + width/2, l2, width, label=args.g2)

    rects3 = ax[1].bar(x2 - width/2, l3, width, label=args.g1)
    rects4 = ax[1].bar(x2 + width/2, l4, width, label=args.g2)

    ax[0].set_ylabel('Score')
    ax[0].set_title(key1)
    ax[0].set_xticks(x1)
    ax[0].set_xticklabels(list(data1[key1].keys()))
    ax[0].legend()
    ax[1].set_title(key2)
    ax[1].set_xticks(x2)
    ax[1].set_xticklabels(list(data1[key2].keys()))
    ax[1].legend()

    fig.tight_layout()
    fig.savefig("graphs/%s/%s_vs_%s_score_SBS.png" %(args.folder, args.g1, args.g2))

def plotNormSideBySide(args):
    _, _, data1, data2, _, _ = getData(args)
    key1, key2 = list(data1.keys())[0], list(data1.keys())[1]
    x = np.arange(len(data1[key1]) + len(data1[key2]))
    width = 0.35
    fig, ax = plt.subplots(1)

    word1_score = {}
    word2_score = {}
    word1_sum = 0
    word2_sum = 0
    for key in data1[key1]:
        word1_score[key] = data1[key1][key]
        word1_sum += data1[key1][key]
        word2_score[key] = data2[key1][key]
        word2_sum += data2[key1][key]
    for key in data1[key2]:
        word1_score[key] = data1[key2][key]
        word1_sum += data1[key2][key]
        word2_score[key] = data2[key2][key]
        word2_sum += data2[key2][key]

    for key in word1_score:
        word1_score[key] /= word1_sum
        word2_score[key] /= word2_sum

    rects1 = ax.bar(x - width/2, list(word1_score.values()), width, label=args.g1)
    rects2 = ax.bar(x + width/2, list(word2_score.values()), width, label=args.g2)

    ax.set_ylabel('Score')
    ax.set_title("Score Normalized Over Groups")
    ax.set_xticks(x)
    ax.set_xticklabels(list(word1_score.keys()))
    ax.legend()

    fig.tight_layout()
    fig.savefig("graphs/%s/%s_vs_%s_score_norm_SBS.png" %(args.folder, args.g1, args.g2))

def plotNormPerBar(args):
    _, _, data1, data2, _, _ = getData(args)
    key1, key2 = list(data1.keys())[0], list(data1.keys())[1]
    x = np.arange(len(data1[key1]) + len(data1[key2]))
    width = 0.35
    fig, ax = plt.subplots(1)

    word1_score = {}
    word2_score = {}
    for key in data1[key1]:
        word1_score[key] = data1[key1][key]
        word2_score[key] = data2[key1][key]
    for key in data1[key2]:
        word1_score[key] = data1[key2][key]
        word2_score[key] = data2[key2][key]

    for key in word1_score:
        try:
            word1_score[key] /= (word1_score[key] + word2_score[key])
        except ZeroDivisionError:
            word1_score[key] = 0
        if word1_score[key] != 0:
            word2_score[key] = 1 - word1_score[key]
        else:
            word2_score[key] = 0

    rects1 = ax.bar(x - width/2, list(word1_score.values()), width, label=args.g1)
    rects2 = ax.bar(x + width/2, list(word2_score.values()), width, label=args.g2)

    ax.set_ylabel('Score')
    ax.set_title("Normalized Scores Over Bars")
    ax.set_xticks(x)
    ax.set_xticklabels(list(word1_score.keys()))
    ax.legend()

    fig.tight_layout()
    fig.savefig("graphs/%s/%s_vs_%s_score_npb.png" %(args.folder, args.g1, args.g2))

def plotNormOcc(args):
    _, _, data1, data2, app1, app2 = getData(args)
    x = np.arange(len(app1))
    width = 0.35
    fig, ax = plt.subplots(1)
    data1 = data1[list(data1.keys())[0]]
    data2 = data2[list(data2.keys())[1]]

    d1 = {}
    d2 = {}
    for key in data1:
        d1[key] = app1[key]
        d2[key] = app2[key]
    for key in data2:
        d1[key] = app1[key]
        d2[key] = app2[key]

    sum1 = 0
    sum2 = 0
    for key in d1:
        sum1 += d1[key]
        sum2 += d2[key]

    for key in d1:
        d1[key] /= sum1
        d2[key] /= sum2

    rects1 = ax.bar(x - width/2, list(d1.values()), width, label=args.g1)
    rects2 = ax.bar(x + width/2, list(d2.values()), width, label=args.g2)

    ax.set_ylabel('Score')
    ax.set_title("occ_norm")
    ax.set_xticks(x)
    ax.set_xticklabels(list(d1.keys()))
    ax.legend()

    fig.tight_layout()
    fig.savefig("graphs/%s/%s_vs_%s_occurence_norm.png" % (args.folder, args.g1, args.g2))

def plotAvgSBS(args):
    _, _, data1, data2, app1, app2 = getData(args)
    keys = list(data1.keys())
    key1 = keys[0]
    key2 = keys[1]
    x1 = np.arange(len(data1[key1]))
    x2 = np.arange(len(data1[key2]))
    width = 0.35
    fig, ax = plt.subplots(1, 2, sharey=True)


    for key in data1[key1]:
        try:
            data1[key1][key] = data1[key1][key] / app1[key]
        except ZeroDivisionError:
            data1[key1][key] = 0
        try:
            data2[key1][key] = data2[key1][key] / app2[key]
        except ZeroDivisionError:
            data2[key1][key] = 0
    for key in data1[key2]:
        try:
            data1[key2][key] = data1[key2][key] / app1[key]
        except ZeroDivisionError:
            data1[key2][key] = 0
        try:
            data2[key2][key] = data2[key2][key] / app2[key]
        except ZeroDivisionError:
            data2[key2][key] = 0

    data1[key1] = sortDict(data1[key1])
    data1[key2] = sortDict(data1[key2])

    l1 = []
    l2 = []
    l3 = []
    l4 = []

    for key in data1[key1]:
        l1.append(data1[key1][key])
        l2.append(data2[key1][key])

    for key in data1[key2]:
        l3.append(data1[key2][key])
        l4.append(data2[key2][key])

    rects1 = ax[0].bar(x1 - width/2, l1, width, label=args.g1)
    rects2 = ax[0].bar(x1 + width/2, l2, width, label=args.g2)

    rects3 = ax[1].bar(x2 - width/2, l3, width, label=args.g1)
    rects4 = ax[1].bar(x2 + width/2, l4, width, label=args.g2)

    ax[0].set_ylabel('Score')
    ax[0].set_title(key1 + "(avg)")
    ax[0].set_xticks(x1)
    ax[0].set_xticklabels(list(data1[key1].keys()))
    ax[0].legend()
    ax[1].set_title(key2 + "(avg)")
    ax[1].set_xticks(x2)
    ax[1].set_xticklabels(list(data1[key2].keys()))
    ax[1].legend()

    fig.tight_layout()
    fig.savefig("graphs/%s/%s_vs_%s_average_score_SBS.png" % (args.folder, key1, key2))

def sortDict(d):
    d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
    return d

def getData(args):
    f1 = open(args.f1, 'r')
    j = json.load(f1)
    data1 = j[args.ste]
    app1 = j[args.ste]["appearances"]
    f1.close()

    f2 = open(args.f2, 'r')
    i = json.load(f2)
    data2 = i[args.ste]
    app2 = i[args.ste]["appearances"]
    f2.close()

    return j, i, data1, data2, app1, app2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--f1', default='logs/she_scores.json', type=str)
    parser.add_argument('--f2', default='logs/he_scores.json', type=str)
    parser.add_argument('--ste', default="occupation", type=str)
    parser.add_argument('--folder', default="basic", type=str)
    parser.add_argument('--g1', default="she", type=str)
    parser.add_argument('--g2', default="he", type=str)
    args = parser.parse_args()
    print(args)

    if not os.path.isdir("graphs/%s" %args.folder):
        os.mkdir("graphs/%s" %args.folder)

    plot2Stereotypes(args.g1, args)
    plotStereotypeAveraged(args.g1, args)

    plot2Stereotypes(args.g2, args)
    plotStereotypeAveraged(args.g2, args)

    plotSideBySide(args)
    plotNormSideBySide(args)
    plotNormPerBar(args)
    plotNormOcc(args)
    plotAvgSBS(args)

    plt.show()
