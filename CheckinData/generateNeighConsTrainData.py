import pickle
train_batches = pickle.load(open("/data/wuning/LSBN Data/GowallaTrainData", "rb"))
adj = pickle.load(open("/data/wuning/LSBN Data/adjMat", "rb"))
pos_samples = []
neg_samples = []
pre_batches = []
count = 0
for batch in train_batches:
  if len(batch) > 0:
    print(count)
    for i in range(2, len(batch[0]) - 1):
      for j in range(0, len(batch)):
        for act in adj[batch[j][i - 1]]:
          if not act == batch[j][i-1]:
            pos_samples.append([batch[j][:i], batch[j][-1]])
            temp = batch[j][:i - 1]
            temp.append(act)
            neg_samples.append([temp, batch[j][-1]])
      for i in range(100, len(pos_samples) + 100, 100):
        temp_batch = []
        if i > len(pos_samples):
          temp_batch.extend(pos_samples[i - 100 : len(pos_samples)])
          temp_batch.extend(neg_samples[i - 100 : len(neg_samples)])
        else:
          temp_batch.extend(pos_samples[i - 100 : i])
          temp_batch.extend(neg_samples[i - 100 : i])
        pre_batches.append(temp_batch)
      pos_samples = []
      neg_samples = []

    count += 1
pickle.dump(pre_batches, open("/data/wuning/LSBN Data/GowallaPolicyTrainData", "wb"))
