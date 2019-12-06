import pickle
import numpy as np
import sys
import gc
from memory_profiler import profile
from memory_profiler import memory_usage
sys.path.append("/home/wuning/AstarRNN")
from policyNetwork import *
heuristics_batches = []
counter = 0
length = 20
rawData = pickle.load(open('/data/wuning/kalmanrnn/blockData50000_', 'rb'))
print("shape:", np.array(rawData).shape)

trainSet = rawData[0:2000]
testSet = rawData[2000:]
dataset = trainSet

AstarRNN = DAN()

def generatePolicyDataSet():
  policyTrainSet = []
  for batch in trainSet:
    batch = (np.array(batch) - np.array([39.84597, 116.310882])) * np.array([111000, 0.667*111000])
    batch = (np.ceil(batch[:, :, 0]/100).astype(np.int32) - 1)*115 + np.ceil(batch[:, :, 1]/100).astype(np.int64) - 1
    for i in range(0, len(batch[0]), 10):
      seg_batch = np.array(batch)[:, i: i+10]
      final_batch = []
      for tra in seg_batch:
        if len(set(tra)) > 5:
          final_batch.append(tra)
      policyTrainSet.append(final_batch)
  pickle.dump(policyTrainSet, open("/data/wuning/AstarBeijing/beijingPolicyTrainSet", "wb"))

def generateTestDataSet():
  beijingTestSet = []
  for batch in testSet:
    batch = (np.array(batch) - np.array([39.84597, 116.310882])) * np.array([111000, 0.667*111000])
    batch = (np.ceil(batch[:, :, 0]/100).astype(np.int32) - 1)*115 + np.ceil(batch[:, :, 1]/100).astype(np.int64) - 1
    for i in range(0, len(batch[0]), length):
      seg_batch = np.array(batch)[:, i: i + length]
      final_batch = []
      for tra in seg_batch:
        if len(set(tra)) > 15:
          final_batch.append(tra)
      beijingTestSet.append(final_batch)
  pickle.dump(beijingTestSet, open("/data/wuning/AstarBeijing/beijingTestSet", "wb"))

def generateHeuristicsTrainSet():
#  from pympler.tracker import SummaryTracker
#  tracker = SummaryTracker()
  all_batches = []
  heuristics_batches = []
  counter = 0
  for batch in dataset:
    if counter < 501:
      counter += 1
      continue
    batch = (np.array(batch) - np.array([39.84597, 116.310882])) * np.array([111000, 0.667*111000])
    batch = (np.ceil(batch[:, :, 0]/100).astype(np.int32) - 1)*115 + np.ceil(batch[:, :, 1]/100).astype(np.int64) - 1
#  cuted_batch = []
#  for tra in batch:
#    last = -1
#    pre_last = -1
#    for item in tra:
#      if last == pre_last and item == last:
#        continue
#      pre_last = last
#      last = item
#    cuted_batch.append(tra)
#  print("1:", np.array(batch).shape)      
    
    for i in range(0, len(batch[0]), 10):
      seg_batch = np.array(batch)[:, i: i+10]
      final_batch = []
      for tra in seg_batch:
        if len(set(tra)) > 5:
          final_batch.append(tra)
      if len(final_batch) == 0:
        continue
#      print("f shape:", np.array(final_batch).shape)
#      print("hb:", sys.getsizeof(heuristics_batches))
#      print("sb:", sys.getsizeof(seg_batch))
#      print("AR:", sys.getsizeof(AstarRNN))

#      else:
#        print(tra)
#    print(np.array(final_batch).shape)
      for j in range(1, len(final_batch[0])-2):
#        wait_actions = np.random.randint(0, 15870, size=[len(final_batch), 2])
#        wait_actions = np.concatenate((wait_actions, np.array(final_batch)[:, j][:, np.newaxis]), 1)
        eval_policy = AstarRNN.policy.eval(
          feed_dict={
            AstarRNN.p_known_:np.array(final_batch)[:, :j],
            AstarRNN.p_destination_:np.array(final_batch)[:, -1]
          })
        wait_actions = np.argsort(np.array(-eval_policy)[:, -1, :], axis=1)[:,:2]
        y_batch = np.zeros(wait_actions.shape)

        for k in range(np.array(final_batch).shape[0]):
          if(not np.array(final_batch)[k, j] == wait_actions[k, 0]):
            wait_actions[k, 1] = wait_actions[k, 0]
            wait_actions[k, 0] = np.array(final_batch)[k, j]
          y_batch[k, 0] = 1

        for k in range(2):
          heuristics_batches.append([np.array(final_batch)[:, 0:j].tolist(), wait_actions[:, k].tolist(), np.array(final_batch)[:, -1].tolist(), y_batch[:, k].tolist()])
#    tracker.print_diff()
    if counter % 100 == 0:
      print(len(heuristics_batches))
      print("batches generated:{}...".format(counter))
    if counter % 500 == 0:
      pickle.dump(heuristics_batches, open("/data/wuning/AstarBeijing/beijing_Q_learning_policy_serpervised_heuristicsTestSet"+str(counter), "wb"))
      heuristics_batches = []
   
#    gc.collect()
    counter += 1
if __name__ == '__main__':
#generatePolicyDataSet()
  generateTestDataSet()
#  generateHeuristicsTrainSet()
