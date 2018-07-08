import pickle
import numpy
batch_size = 40
def tokenize():
  File = open("/data/wuning/foursquare/dataset_TSMC2014_NYC.txt", "r")
  rawData = File.readlines()
  userData = {}
  for line in rawData:
#    print(line)
    items = line.split("\t")
#    print user[0],user[1],user[2],user[3],user[4],user[5]
    if(len(items) < 8):
      continue
    record = [items[1], items[4], items[5], items[7]]
    if not items[0] in userData:
      userData[items[0]]=[record]
    else:
      userData[items[0]].append(record)
  return userData
userData = tokenize()
tras = []
count = 0
geoMap = {}
for key, value in userData.items():
  tra = []
  print(count)
  count += 1
  lastDay = 0
  for rec in value:
    if not rec[0] in geoMap:
      geoMap[rec[0]] = [rec[1], rec[2]]
    if len(tra) == 0:
      tra.append([rec[0], rec[3]])
      lastDay = rec[3].split()[2]
      lastTime = rec[3].split()[3].split(":")[0]
    else:
      if rec[0] == tra[-1]:
#        tra.append([rec[0], rec[3]])
        continue  
      if not (int(rec[3].split()[3].split(":")[0]) - int(lastTime) + 24 * (int(rec[3].split()[2].split(":")[0]) - int(lastDay))) < 5:
#        print(rec[3].split()[2], lastDay)
        if len(tra) >= 3:
          tras.append(tra)
        tra = []
      lastDay = rec[3].split()[2]
      lastTime = rec[3].split()[3].split(":")[0]
      tra.append([rec[0], rec[3]])
for i in range(1000):
  print(tras[i])
print(len(tras))
sum_ = 0
for tra in tras:
  sum_ += len(tra)
#  print(len(tra))
print("sum:", sum_)
Set = set([])
for tra in tras:
  for item in tra:
    Set.add(item[0])
print(len(Set))

len_dict = {}

for tra in tras:
  if len(tra) in len_dict:
    len_dict[len(tra)].append(tra)
  else:
    len_dict[len(tra)] = [tra]

batches = []
for key in len_dict:
  len_tra = len_dict[key]
  print(key, len(len_tra))
  for i in range(0, len(len_tra) - batch_size, batch_size):
    batches.append(len_tra[i : i + batch_size])
print(len(batches))
numpy.random.shuffle(batches)
saveFile = open("/data/wuning/foursquare/NYCData", "wb")
pickle.dump(batches, saveFile, -1)
saveMap = open("/data/wuning/foursquare/token2cor", "wb")
pickle.dump(geoMap, saveMap, -1)











