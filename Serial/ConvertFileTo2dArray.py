file = open(r"Data\UsCapitals.txt", "r")

print("{")
oneLine = file.readlines()
for newLine in oneLine:
    if(newLine != "EOF"):
        print("{", end="")
        newTxt = newLine[:-1].split(" ")
        print(newTxt[0], end="")
        print(", ", end="")
        print(newTxt[1], end="")
        print("},", end="")
print("}")

file.close()