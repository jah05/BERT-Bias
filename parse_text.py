k = 500
d = {}
f = open("loghe.txt", "r")

counter = 0
for line in f:
    if counter != 0:
        line = line.strip()
        line = line.split(" ")
        d[line[1]] = float(line[2])
    else:
        counter +=1

f.close()


d = dict(sorted(d.items(),  key=lambda item: item[1], reverse=True))

f = open("loghe.txt", "w")
f.write("# word avg_score")
for i, key in enumerate(d):
    f.write("\n%d. %s %f" %(i+1, key, d[key]))
    if(i==k):
        break
f.close()
