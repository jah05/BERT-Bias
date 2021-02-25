import json

f = open("logs/test.json", 'r')
d = json.load(f)
n = d["targ1"]["examples"]
n += d["targ2"]["examples"]
a = d["attr1"]["examples"]
a += d["attr2"]["examples"]
f.close()

f = open("bleached.txt", 'w')
for name in n:
    for att in a:
        f.write(name + " " + att + "\n")

f.close()
