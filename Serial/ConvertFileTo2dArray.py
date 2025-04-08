
file = open(r"Data/HamilitonianCycle.txt", "r")

print("{", end="")
oneLine = file.readlines()
for newLine in oneLine:
    if newLine != "EOF" or newLine == "-1\n":
        print("{", end="")
        newTxt = newLine[:-1].split()
        print(newTxt[0], end="")
        print(", ", end="")
        print(newTxt[1], end="")
        print("},", end="")
print("}")

file.close()