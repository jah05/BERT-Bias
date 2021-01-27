file = open(r"logs/test.txt", "r")

for i, line in enumerate(file):
    print(i)

file.close()
