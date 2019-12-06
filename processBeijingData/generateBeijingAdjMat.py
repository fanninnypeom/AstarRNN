import pickle
train_dir = "/data/wuning/AstarBeijing/beijingPolicyTrainSet"       #/data/wuning/mobile trajectory/Q_learning_trainSet
test_dir = "/data/wuning/AstarBeijing/beijingTestSet"

File = open(train_dir, "rb")   #trainSet
train_batches = pickle.load(File)
File = open(test_dir, "rb")    #testSet
test_batches = pickle.load(File)
adj = []
for i in range(0, 26261):
  adj.append([])
count = 0
for batch in train_batches:
#  if count / 1000 == 0:
  print(count)
  count += 1
  for tra in batch:
    for i in range(len(tra) - 1):
      if not tra[i + 1] in adj[tra[i]]:
        adj[tra[i]].append(tra[i + 1])
      if not tra[i] in adj[tra[i + 1]]:
        adj[tra[i + 1]].append(tra[i])

for i in range(len(adj), 0):
  print(len(adj[i]))
pickle.dump(adj, open("/data/wuning/AstarBeijing/adjMat", "wb"))

