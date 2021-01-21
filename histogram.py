import numpy as np
import matplotlib.pyplot as plt
import json
import matplotlib.patches as mpatches

def plot2Stereotypes(data, key1, key2, name):
    data1 = data[key1]
    data2 = data[key2]
    x = np.arange(len(data1) + len(data2))
    width = 0.35

    fig, ax = plt.subplots()
    bars = ax.bar(x, list(data1.values()) + list(data2.values()), width)
    ax.set_title(name)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(list(data1.keys()) + list(data2.keys()))
    fig.legend(handles=[mpatches.Patch(color='red', label=key1), mpatches.Patch(color='blue', label=key2)])
    for i in range(len(data1)):
        ax.get_children()[i].set_color('r')
    for i in range(len(data1), len(data1) + len(data2)):
        ax.get_children()[i].set_color('b')
    fig.tight_layout()
    fig.savefig('graphs/'+ name + '.png')

def plotStereotypeAveraged(data, key1, key2, name):
    data1 = data[key1]
    data2 = data[key2]

    average1 = {}
    average2 = {}

    for key in data1:
        try:
            average1[key] = data1[key] / data["appearances"][key]
        except ZeroDivisionError:
            average1[key] = 0
    for key in data2:
        try:
            average2[key] = data2[key] / data["appearances"][key]
        except ZeroDivisionError:
            average2[key] = 0

    average1 = sortDict(average1)
    average2 = sortDict(average2)
    x = np.arange(len(average1) + len(average2))
    width = 0.35

    fig, ax = plt.subplots()
    bars = ax.bar(x, list(average1.values()) + list(average2.values()), width)
    ax.set_title(name + " (Averaged)")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(list(average1.keys()) + list(average2.keys()))
    fig.legend(handles=[mpatches.Patch(color='red', label=key1), mpatches.Patch(color='blue', label=key2)])
    for i in range(len(average1)):
        ax.get_children()[i].set_color('r')
    for i in range(len(average1), len(average1) + len(average2)):
        ax.get_children()[i].set_color('b')
    fig.tight_layout()
    fig.savefig("graphs/" + name + "_avg.png")

def plotSideBySide(att1, att2, c1, c2, key1, key2, name):
    x1 = np.arange(len(att1[key1]))
    x2 = np.arange(len(att1[key2]))
    width = 0.35
    fig, ax = plt.subplots(1, 2, sharey=True)

    l1 = []
    l2 = []
    l3 = []
    l4 = []

    for key in att1[key1]:
        l1.append(att1[key1][key])
        l2.append(att2[key1][key])

    for key in att1[key2]:
        l3.append(att1[key2][key])
        l4.append(att2[key2][key])

    rects1 = ax[0].bar(x1 - width/2, l1, width, label=c1)
    rects2 = ax[0].bar(x1 + width/2, l2, width, label=c2)

    rects3 = ax[1].bar(x2 - width/2, l3, width, label=c1)
    rects4 = ax[1].bar(x2 + width/2, l4, width, label=c2)

    ax[0].set_ylabel('Score')
    ax[0].set_title(key1)
    ax[0].set_xticks(x1)
    ax[0].set_xticklabels(list(att1[key1].keys()))
    ax[0].legend()
    ax[1].set_title(key2)
    ax[1].set_xticks(x2)
    ax[1].set_xticklabels(list(att1[key2].keys()))
    ax[1].legend()

    fig.tight_layout()
    fig.savefig("graphs/" + name + ".png")

def plotNormSideBySide(att1, att2, c1, c2, key1, key2, name):
    x = np.arange(len(att1[key1]) + len(att1[key2]))
    width = 0.35
    fig, ax = plt.subplots(1)

    word1_score = {}
    word2_score = {}
    word1_sum = 0
    word2_sum = 0
    for key in att1[key1]:
        word1_score[key] = att1[key1][key]
        word1_sum += att1[key1][key]
        word2_score[key] = att2[key1][key]
        word2_sum += att2[key1][key]
    for key in att1[key2]:
        word1_score[key] = att1[key2][key]
        word1_sum += att1[key2][key]
        word2_score[key] = att2[key2][key]
        word2_sum += att2[key2][key]

    for key in word1_score:
        word1_score[key] /= word1_sum
        word2_score[key] /= word2_sum

    rects1 = ax.bar(x - width/2, list(word1_score.values()), width, label=c1)
    rects2 = ax.bar(x + width/2, list(word2_score.values()), width, label=c2)

    ax.set_ylabel('Score')
    ax.set_title("score_norm")
    ax.set_xticks(x)
    ax.set_xticklabels(list(word1_score.keys()))
    ax.legend()

    fig.tight_layout()
    fig.savefig("graphs/" + name + "_norm.png")

def plotNormPerBar(att1, att2, c1, c2, key1, key2, name):
    x = np.arange(len(att1[key1]) + len(att1[key2]))
    width = 0.35
    fig, ax = plt.subplots(1)

    word1_score = {}
    word2_score = {}
    for key in att1[key1]:
        word1_score[key] = att1[key1][key]
        word2_score[key] = att2[key1][key]
    for key in att1[key2]:
        word1_score[key] = att1[key2][key]
        word2_score[key] = att2[key2][key]

    for key in word1_score:
        try:
            word1_score[key] /= (word1_score[key] + word2_score[key])
        except ZeroDivisionError:
            word1_score[key] = 0
        if word1_score[key] != 0:
            word2_score[key] = 1 - word1_score[key]
        else:
            word2_score[key] = 0

    rects1 = ax.bar(x - width/2, list(word1_score.values()), width, label=c1)
    rects2 = ax.bar(x + width/2, list(word2_score.values()), width, label=c2)

    ax.set_ylabel('Score')
    ax.set_title("bar_norm")
    ax.set_xticks(x)
    ax.set_xticklabels(list(word1_score.keys()))
    ax.legend()

    fig.tight_layout()
    fig.savefig("graphs/" + name + "_barnorm.png")

def plotNormOcc(data1, data2, app1, app2, c1, c2, name):
    x = np.arange(len(app1))
    width = 0.35
    fig, ax = plt.subplots(1)

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

    rects1 = ax.bar(x - width/2, list(d1.values()), width, label=c1)
    rects2 = ax.bar(x + width/2, list(d2.values()), width, label=c2)

    ax.set_ylabel('Score')
    ax.set_title("occ_norm")
    ax.set_xticks(x)
    ax.set_xticklabels(list(d1.keys()))
    ax.legend()

    fig.tight_layout()
    fig.savefig("graphs/" + name + "_norm.png")

def plotAvgSBS(att1, att2, c1, c2, key1, key2, name):
    x1 = np.arange(len(att1[key1]))
    x2 = np.arange(len(att1[key2]))
    width = 0.35
    fig, ax = plt.subplots(1, 2, sharey=True)


    for key in att1[key1]:
        try:
            att1[key1][key] = att1[key1][key] / att1["appearances"][key]
        except ZeroDivisionError:
            att1[key1][key] = 0
        try:
            att2[key1][key] = att2[key1][key] / att2["appearances"][key]
        except ZeroDivisionError:
            att2[key1][key] = 0
    for key in att1[key2]:
        try:
            att1[key2][key] = att1[key2][key] / att1["appearances"][key]
        except ZeroDivisionError:
            att1[key2][key] = 0
        try:
            att2[key2][key] = att2[key2][key] / att2["appearances"][key]
        except ZeroDivisionError:
            att2[key2][key] = 0

    att1[key1] = sortDict(att1[key1])
    att1[key2] = sortDict(att1[key2])

    l1 = []
    l2 = []
    l3 = []
    l4 = []

    for key in att1[key1]:
        l1.append(att1[key1][key])
        l2.append(att2[key1][key])

    for key in att1[key2]:
        l3.append(att1[key2][key])
        l4.append(att2[key2][key])

    rects1 = ax[0].bar(x1 - width/2, l1, width, label=c1)
    rects2 = ax[0].bar(x1 + width/2, l2, width, label=c2)

    rects3 = ax[1].bar(x2 - width/2, l3, width, label=c1)
    rects4 = ax[1].bar(x2 + width/2, l4, width, label=c2)

    ax[0].set_ylabel('Score')
    ax[0].set_title(key1 + "(avg)")
    ax[0].set_xticks(x1)
    ax[0].set_xticklabels(list(att1[key1].keys()))
    ax[0].legend()
    ax[1].set_title(key2+ "(avg)")
    ax[1].set_xticks(x2)
    ax[1].set_xticklabels(list(att1[key2].keys()))
    ax[1].legend()

    fig.tight_layout()
    fig.savefig("graphs/" + name + "_avg.png")

def sortDict(d):
    d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
    return d

if __name__ == "__main__":
    fname1 = "logs/she_scores.json"
    f1 = open(fname1, 'r')
    data1 = json.load(f1)["occupation"]
    key1 = "she"
    key2 = "he"
    f1.close()

    fname2 = "logs/he_scores.json"
    f2 = open(fname2, 'r')
    data2 = json.load(f2)["occupation"]
    key3 = "she"
    key4 = "he"
    f2.close()

    plotNormSideBySide(data1, data2, "female", "male", key1, key2, "she_he_comparison")
    plotNormPerBar(data1, data2, "female", "male", key1, key2, "she_he_comparison")
    # plot2Stereotypes(data1,key1,key2, "she_occupation_graph")
    # plot2Stereotypes(data2,key3,key4, "he_occupation_graph")
    #
    # plotStereotypeAveraged(data1,key1,key2, "she_occupation_graph")
    # plotStereotypeAveraged(data2,key3,key4, "he_occupation_graph")
    #
    # plotSideBySide(data1, data2, "Woman", "Man", key1, key2, "she_he_comparison")
    # plotAvgSBS(data1, data2, "Woman", "Man", key1, key2, "she_he_comparison")
    plotNormOcc(data1["she"], data1["he"], data1["appearances"], data2["appearances"], "female", "male", "she_he_appearance_comp")

    plt.show()
