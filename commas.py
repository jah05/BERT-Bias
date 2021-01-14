fname = input("fname: ")

f = open(fname, 'r')

words = []
for line in f:
    line = line.strip()
    words.append('"%s"' %line)

f.close()

s = ''
for w in words:
    s += w + ", "

print(s)
