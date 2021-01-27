import json

def addAttribute():
    fname = input("File Name: ")
    stereotype = input("Stereotype Category: ")
    key1 = input("Type 1: ")

    key2 = input("Type 2: ")
    print("Enter Att1. Ctrl-Z to save it.")
    s1 = ''
    while True:
        try:
            line = input()
        except EOFError:
            break
        s1 += line

    attributes1 = s1.split(", ")
    print("Enter Att2. Ctrl-Z to save it.")
    s2 = ''
    while True:
        try:
            line = input()
        except EOFError:
            break
        s2 += line

    attributes2 = s2.split(", ")
    f = open(fname, 'r')
    data = json.load(f)
    f.close()
    data["stereotype"][stereotype] = {key1:[], key2:[]}
    data["stereotype"][stereotype][key1] = attributes1
    data["stereotype"][stereotype][key2] = attributes2
    f = open(fname, 'w')
    json.dump(data, f)
    f.close()

def addName():
    fname = input("File Name: ")
    key1 = input("Race 1: ")
    print("Enter Names. Ctrl-Z to save it.")
    s1 = ''
    while True:
        try:
            line = input()
        except EOFError:
            break
        s1 += line

    names1 = s1.split(", ")
    key2 = input("Race 2: ")
    print("Enter Names. Ctrl-Z to save it.")
    s2 = ''
    while True:
        try:
            line = input()
        except EOFError:
            break
        s2 += line
    names2 = s2.split(", ")

    f = open(fname, 'r')
    data = json.load(f)
    f.close()
    data["names"][key1] = names1
    data["names"][key2] = names2
    f = open(fname, 'w')
    json.dump(data, f)
    f.close()

if __name__ == '__main__':
    addAttribute()
