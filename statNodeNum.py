File = open("hScoreSearchLog", "r")
lines = File.readlines()
allNum = 0
for line in lines[22:100]:
  num = int(line.split()[1])
  allNum += num
print(allNum)
