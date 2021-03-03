import histogram
import argparse
import os
import matplotlib.pyplot as plt
import csv

def evaluate(args):
    _, _, data1, data2, app1, app2 = histogram.getData(args)
    key1, key2 = list(data1.keys())[0], list(data1.keys())[1]
    sum1 = 0
    sum2 = 0
    for key in data1[key1]:
        try:
            data1[key1][key] = data1[key1][key] / app1[key]
        except ZeroDivisionError:
            data1[key1][key] = 0
        try:
            data2[key1][key] = data2[key1][key] / app2[key]
        except ZeroDivisionError:
            data2[key1][key] = 0
        sum1 += data1[key1][key]
        sum2 += data2[key1][key]

    for key in data1[key2]:
        try:
            data1[key2][key] = data1[key2][key] / app1[key]
        except ZeroDivisionError:
            data1[key2][key] = 0
        try:
            data2[key2][key] = data2[key2][key] / app2[key]
        except ZeroDivisionError:
            data2[key2][key] = 0
        sum1 += data1[key2][key]
        sum2 += data2[key2][key]

    for key in data1[key1]:
        data1[key1][key] /= sum1
        data2[key1][key] /= sum2
    for key in data1[key2]:
        data1[key2][key] /= sum1
        data2[key2][key] /= sum2

    with open("graphs/%s/ratios.csv" %args.folder, 'w') as f:
        fieldnames = ["word", "category", "%s score" %args.g1, "%s score" %args.g2, "%s:%s" %(args.g1, args.g2), "%s:%s" %(args.g2, args.g1)]
        f_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        f_writer.writerow(fieldnames)
        for key in data1[key1]:
            f_writer.writerow([key, key1, data1[key1][key], data2[key1][key], data1[key1][key]/data2[key1][key], data2[key1][key]/data1[key1][key]])

        for key in data1[key2]:
            f_writer.writerow([key, key2, data1[key2][key], data2[key2][key], data1[key2][key]/data2[key2][key], data2[key2][key]/data1[key2][key]])

    sumA = 0
    counterA = 0
    for key in data1[key1]:
        try:
            sumA += data1[key1][key] / data2[key1][key]
            counterA+=1
        except:
            pass
    sumB = 0
    counterB = 0
    for key in data1[key2]:
        try:
            sumB += data2[key2][key] / data1[key2][key]
            counterB+=1
        except:
            pass

    beta = (sumA / counterA) + (sumB / counterB)
    print(beta)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--f1', default='logs/she_scores.json', type=str)
    parser.add_argument('--f2', default='logs/he_scores.json', type=str)
    parser.add_argument('--ste', default="occupation", type=str)
    parser.add_argument('--folder', default="basic", type=str)
    parser.add_argument('--g1', default="she", type=str)
    parser.add_argument('--g2', default="he", type=str)
    parser.add_argument('--restrict', default=False, type=bool)
    parser.add_argument('--skip', default=1, type=int)
    args = parser.parse_args()

    if not os.path.isdir("graphs/%s" %args.folder):
        os.mkdir("graphs/%s" %args.folder)

    histogram.plotWordProbabilitiesSBS(args)
    evaluate(args)
    plt.show()
