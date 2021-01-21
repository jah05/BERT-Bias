fname = input("fname: ")

f = open(fname, 'r')

words = []
for line in f:
    line = line.strip()
    words = line.split(", ")

for i in range(len(words)):
    words[i] = '"%s"' %words[i]

f.close()

s = ''
for w in words:
    s += w + ", "

print(s)
