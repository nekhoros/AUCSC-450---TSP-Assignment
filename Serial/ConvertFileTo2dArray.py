file = open(r"Data/16Points.txt", "r")

oneLine = file.readlines()
for newLine in oneLine:
    if newLine != "EOF" or newLine == "-1\n":
        newTxt = newLine[:-1].split()
        print(int(float(newTxt[1])), end=" ")
        print(int(float(newTxt[2])))

file.close()