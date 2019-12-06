import tensorflow as tf
import numpy as np
import random 
import pickle
import copy
from collections import deque
import os
import math  
import datetime
from model import *
from utils import *
from metric import *
os.environ['CUDA_VISIBLE_DEVICES']='3'

def reconstruct_path(model, cameFrom, current):
  total_path = [current]
  while current in cameFrom:
    current = cameFrom[current]
    total_path.append(current)
  return total_path

def insert(model, item, the_list, f_score):  #插入 保持从大到小的顺序
  if(len(the_list) == 0):
    return [item]
  for i in range(len(the_list)):
    if(f_score[the_list[i]] < f_score[item]):
      the_list.insert(i, item)
      break
    if i == len(the_list) - 1:
      the_list.append(item)
  return the_list

def move(model, item, the_list, f_score):
  for it in the_list:
    if(f_score[it] == f_score[item]):
      the_list.remove(it)
      break
  return insert(model, item, the_list, f_score)
def greedyTest(model):
  counters = 0
  all_len = 0
  for tra_bat, hour_bat, day_bat, his_bat, his_hour_bat, his_day_bat, his_mask_bat in generate_batch(maskData[2000:2200], historyData, trainData[2000:2200], trainTimeData[2000:2200], trainUserData[2000:2200]):
    result = []
    tra_bat = np.array(tra_bat[:, :40])
    starttime = datetime.datetime.now()
    NodeNum = 0
    for item_0, item_1, item_2, item_3, start, unknown, end in zip(his_bat, his_hour_bat, his_day_bat, his_mask_bat, np.array(tra_bat)[:, 0], np.array(tra_bat)[:,1:-1], np.array(tra_bat)[:, -1]):
      path = [start]
      for i in range(len(unknown)):
        NodeNum += 1  
#          print("path:", path)
        st_value = model.st_all.eval(
            feed_dict={
              model.st_known_:np.array(path)[np.newaxis, :],
              model.st_destination_:[[end]],
#              model.his_tra:[item_0],
#              model.his_time:[item_1],
#              model.his_day:[item_2],
#              model.his_padding_mask:[item_3]
          })
        policy = np.argmax(st_value, axis=2)[:, -1]
        if policy[0] == end:
          break    
        path.append(policy[0])
      path.append(end)
      result.append(path)
    endtime = datetime.datetime.now()
    for infer, real in zip(result, tra_bat):
#      counters += edit(infer, real)
      try:  
        print("infer:", len(infer), [locList[ite] for ite in infer])
        print("real:", len(real), [locList[ite] for ite in real])
      except:
        print(infer)
      for item in infer[1:-1]:
        if item in real[1:-1]:
          counters += 1
      all_len += len(infer) - 2
#    all_len += len(result) 
#    all_edt = all_edt / all_len
    print ("time:", (endtime - starttime).seconds)
    print("NodeNum:", NodeNum)
    print(counters, float(counters)/all_len)
def beamSearch(model):
  beam_width = 3
  all_len = 0
  counters = 0
  for batch in testData:
    known_seq = [np.array(batch)[:, 0][:, np.newaxis]]
    known_score = []
    for i in range(0, len(batch[0]) - 2):
      if i == 0:
        policy_value = self.policy.eval(
            feed_dict={
              model.p_known_:known_seq[0],
              model.p_destination_:np.array(batch)[: , -1],
        })
        policy = np.argsort(-policy_value, axis=2)[:,-1,:beam_width]
        temp = known_seq
        known_seq = []
        for j in range(0, beam_width):
          known_seq.append(np.concatenate((temp[0], policy[:, j][:, np.newaxis]), 1))
#            print(policy[:, j].shape, np.array(policy_value)[:, -1, :].shape) 
          known_score.append([np.array(policy_value)[:, -1, :][enum, item] for enum, item in enumerate(policy[:, j])])
#            known_score.append(np.choose(policy[:, j], np.array(policy_value)[:, -1, :].T))    
        continue
      policy_known_score = []
      for j in range(0, len(known_seq)):
        policy_value = model.policy_prob.eval(
            feed_dict={
              model.p_known_:known_seq[j],
              model.p_destination_:np.array(batch)[: , -1],
        })
        immedia = (1 - i / len(batch[0]))*np.array(known_score[j])[:, np.newaxis] + np.array(policy_value)[:, -1, :]
        if j == 0:
#            all_policy = np.array(policy_value)[:, -1, :]
          policy_known_score = immedia
        else:
#            all_policy = np.concatenate((all_policy, np.array(policy_value)[:, -1, :]), axis=1)
          policy_known_score = np.concatenate((policy_known_score, immedia), axis=1)
#          policy_value = np.array(policy_value)
      policy = np.argsort(-policy_known_score, axis=1)[:,:beam_width]
      raw_policy = policy
      index = policy // 26261
      policy = policy % 26261
      last_known = []
#        print(known_score[0])
      for k in range(0, beam_width):
        tem_batch = []
        for l in range(0, len(policy)):
          tem = known_seq[index[l][k]][l]
          tem = np.append(tem, policy[l][k])
          tem_batch.append(tem)
          known_score[k][l] = policy_known_score[l][raw_policy[l, k]]
        last_known.append(tem_batch)
      known_seq = last_known
    for infer, real in zip(known_seq[0], batch):
      for item in infer[1:]:
        if item in real[1:-1]:
          counters += 1
      all_len += len(real) - 2
    print(counters, all_len, float(counters)/all_len)

def visGraphAttention(model):

  location_embeddings = model.session.run(tf.get_default_graph().get_tensor_by_name("st_network/location_embedding/embeddings:0"))
  adj = np.matrix(graphData)[:15500, :15500]
  G = nx.from_numpy_matrix(adj, create_using=nx.DiGraph())
  inv_G = nx.from_numpy_matrix(adj.T, create_using=nx.DiGraph())
  end = locList.index("67695")
#  temps = [[45265, 45267,45269,8329,2573], [17243, 97218,97219, 8329,2573]]
# temp = [17243, 97218,97219, 8329,2573]
  temps = [[locList.index("17367"), locList.index("62688"), locList.index("91291"), locList.index("91292"), locList.index("91294"), locList.index("94532")]]

#  temps = [[locList.index(str(item)) for item in temp] for temp in temps]
  for temp in temps:
    waiting = temp[-1]
    src_adj, des_adj, des_emb, src_emb, des_mask, src_mask, src_node, des_node = generate_sub_graph(location_embeddings, G, inv_G, 1000, src=waiting, des=end)
       
    h_score =  model.heuristics.eval(
      feed_dict={
      model.st_known_:np.array(temp)[np.newaxis, :],
      model.st_destination_:[[end]],
      model.src_bias_mat:[src_adj],
      model.des_bias_mat:[des_adj],
      model.src_embedding:[src_emb],
      model.des_embedding:[des_emb],
      model.src_mask:[src_mask],
      model.des_mask:[des_mask]
      }
    )


    src_features = model.src_f.eval(
        feed_dict={
          model.st_known_:np.array(temp)[np.newaxis, :],
          model.st_destination_:[[end]],
          model.src_bias_mat:[src_adj],
          model.des_bias_mat:[des_adj],
          model.src_embedding:[src_emb],
          model.des_embedding:[des_emb],
          model.src_mask:[src_mask],
          model.des_mask:[des_mask]
      }
    )
    des_features = model.des_f.eval(
        feed_dict={
          model.st_known_:np.array(temp)[np.newaxis, :],
          model.st_destination_:[[end]],
          model.src_bias_mat:[src_adj],
          model.des_bias_mat:[des_adj],
          model.src_embedding:[src_emb],
          model.des_embedding:[des_emb],
          model.src_mask:[src_mask],
          model.des_mask:[des_mask]
      }
    )

    print(np.array(src_features).shape, np.array(des_features).shape)
    src_weight = np.dot(np.array(src_features[0]), np.array(src_features[0,0]))
    des_weight = np.dot(np.array(des_features[0]), np.array(des_features[0,0]))
    print(src_weight.shape, des_weight.shape)
    print(len(src_node), len(des_node))
    print("src:", [locList[it] for it in src_node], src_weight.tolist())
    print("des:", [locList[it] for it in des_node], des_weight.tolist())

def admissibleHeu(model):
  results = []
  counters = 0
  all_len = 0
  all_search_count = 0
  print(np.array(trainData[2000:]).shape)
  location_embeddings = model.session.run(tf.get_default_graph().get_tensor_by_name("st_network/location_embedding/embeddings:0"))
  adj = np.matrix(graphData)[:16500, :16500]
  G = nx.from_numpy_matrix(adj, create_using=nx.DiGraph())
  inv_G = nx.from_numpy_matrix(adj.T, create_using=nx.DiGraph())
  all_count = 0
  error = []
  for tra_bat, hour_bat, day_bat, his_bat, his_hour_bat, his_day_bat, his_mask_bat in generate_batch(maskData[0:2000], historyData, trainData[0:2000], trainTimeData[0:2000], trainUserData[0:2000]):
#  for batch in testData:
    print('-------')  
    result = []
    count = 0
    print(np.array(tra_bat).shape)
#    tra_bat = np.array(tra_bat)[:, :7]
    for start, unknown, end in zip(np.array(tra_bat)[:, 0], np.array(tra_bat)[:,1:-1], np.array(tra_bat)[:, -1]):
        time_stamp = 10
        st_value = model.st_all_prob.eval(
          feed_dict={
            model.st_known_:[np.array(tra_bat)[count, :]],
            model.st_destination_:[[end]],
        })
        st_arg = np.argsort(-st_value, axis=2)[0][:, :2]
#        print(st_arg)
#        print(tra_bat[count])
        rewards = [st_value[0][z + time_stamp][tra_bat[count][z + 1 + time_stamp]] for z in range(len(np.array(tra_bat)[count, time_stamp:]) - 1)]
#        print("rewards:", rewards)
        src_adj, des_adj, des_emb, src_emb, des_mask, src_mask, src_node, des_node = generate_sub_graph(location_embeddings, G, inv_G, 500, src=tra_bat[count, time_stamp], des=tra_bat[count, -1])
       
        h_score =  model.heuristics.eval(
            feed_dict={
            model.st_known_:[np.array(tra_bat)[count, :time_stamp]],
            model.st_destination_:[[end]],
            model.src_bias_mat:[src_adj],
            model.des_bias_mat:[des_adj],
            model.src_embedding:[src_emb],
            model.des_embedding:[des_emb],
            model.src_mask:[src_mask],
            model.des_mask:[des_mask]
            }
        )
        print(h_score - np.sum(rewards), h_score, np.sum(rewards))
        error.append(h_score[0][0] - np.sum(rewards))
        count += 1
        all_count += 1
        if(all_count % 1000 == 0):
          print(error)    
#        policy_value = np.exp(policy_value)/np.sum(np.exp(policy_value), axis=2)[:, :, np.newaxis]
#        st_arg = np.argsort(-st_value, axis=2)[0, -1][:2]


def AstarTestSoftmax(model):
  results = []
  counters = 0
  all_len = 0
  all_search_count = 0
  print(np.array(trainData[6000:]).shape)
  location_embeddings = model.session.run(tf.get_default_graph().get_tensor_by_name("st_network/location_embedding/embeddings:0"))
  adj = np.matrix(graphData)[:15208, :15208]
  G = nx.from_numpy_matrix(adj, create_using=nx.DiGraph())
  inv_G = nx.from_numpy_matrix(adj.T, create_using=nx.DiGraph())

  for tra_bat, hour_bat, day_bat, his_bat, his_hour_bat, his_day_bat, his_mask_bat in generate_batch(maskData[0:], historyData, trainData[6000:], trainTimeData[0:], trainUserData[0:]):
#  for batch in testData:
    print('-------')  
    result = []
    count = 0
    print(np.array(tra_bat).shape)
#    tra_bat = np.array(tra_bat)[:, :10]
    for start, unknown, end in zip(np.array(tra_bat)[:, 0], np.array(tra_bat)[:,1:-1], np.array(tra_bat)[:, -1]):
      print(start)
      allNode = []
      closedSet = []
      openSet = [start]
      cameFrom = {}
      pathFounded = {start: [start]}
      gScore = {}
      fScore = {}
      gScore[start] = 0
      fScore[start] = 0
      waitingTra = 0
      bestScore = -10000000
      bestTra = []
      searchCount = 0
      while len(openSet) > 0:
        searchCount += 1
        current = openSet[0]
        allNode.append(current)
        openSet.remove(current)
        closedSet.append(current)
        if gScore[current] > bestScore:# and len(pathFounded[current]) == (len(unknown) + 1):
          bestScore = gScore[current]
          bestTra = copy.deepcopy(pathFounded[current])
          bestTra.append(end)
#        if len(pathFounded[current]) > (len(unknown) + 1) or len(pathFounded[current]) == (len(unknown) + 1):
#          continue
        if current == end:
          bestTra = copy.deepcopy(pathFounded[current])
          bestTra.append(end)
#          result.append(bestTra)  
          print("advance")  
          break    
#        mask = []
#        if not current in OSMadj:
#            continue
#        for nex in OSMadj[current]:
#          mask.append(int(nex))
#        if len(mask) > 70:
#          mask = mask[:70]
#        while len(mask) < 70:
#          mask.append(-1)

        st_value = model.st_all_prob.eval(
          feed_dict={
            model.st_known_:[pathFounded[current]],
            model.st_destination_:[[end]],
#            model.trans_mat:batch[3]
#            model.st_time:hour_bat,
#            model.st_day:day_bat,
#            model.padding_mask:tra_mask_bat,
#            model.his_tra:his_bat,
#            model.his_time:his_hour_bat,
#            model.his_day:his_day_bat,
#            model.his_padding_mask:his_mask_bat
        })
#            model.st_known_:[pathFounded[current]],
#            model.st_destination_:[end],
#            model.trans_mat:[[mask for item in pathFounded[current]]]

#        st_value = np.exp(st_value)/np.sum(np.exp(st_value))
#        policy_value = np.exp(policy_value)/np.sum(np.exp(policy_value), axis=2)[:, :, np.newaxis]
#        st_arg = np.argsort(-st_value, axis=2)[0, -1][:2]
        st_arg = G[current]
        for waiting in st_arg:
          if (waiting in closedSet):
            continue
          one_step_value = st_value[0][-1][waiting]
#          if one_step_value == 0:
#              one_step_value = 0.01
          g_score = one_step_value + gScore[current]#((1 - len(pathFounded[current]) / 20)) * st_value[waiting_count][0] + gScore[current]
#            f_score = np.array(policy_value)[-1, -1, waiting] + fScore[current]           
          temp = copy.deepcopy(pathFounded[current])
          temp.append(waiting)
#          src_adj, des_adj, des_emb, src_emb, des_mask, src_mask, src_node, des_node = generate_sub_graph(location_embeddings, G, inv_G, 500, src=waiting, des=end)
       
          h_score =  model.heuristics.eval(
            feed_dict={
            model.st_known_:np.array(temp)[np.newaxis, :],
            model.st_destination_:[[end]],
#            model.src_bias_mat:[src_adj],
#            model.des_bias_mat:[des_adj],
#            model.src_embedding:[src_emb],
#            model.des_embedding:[des_emb],
#            model.src_mask:[src_mask],
#            model.des_mask:[des_mask]
            }
          )

#          src_features = model.src_f.eval(
#            feed_dict={
#              model.st_known_:np.array(temp)[np.newaxis, :],
#              model.st_destination_:[[end]],
#              model.src_bias_mat:[src_adj],
#              model.des_bias_mat:[des_adj],
#              model.src_embedding:[src_emb],
#              model.des_embedding:[des_emb],
#              model.src_mask:[src_mask],
#              model.des_mask:[des_mask]
#            }
#           )
#          des_features = model.des_f.eval(
#            feed_dict={
#              model.st_known_:np.array(temp)[np.newaxis, :],
#              model.st_destination_:[[end]],
#              model.src_bias_mat:[src_adj],
#              model.des_bias_mat:[des_adj],
#              model.src_embedding:[src_emb],
#              model.des_embedding:[des_emb],
#              model.src_mask:[src_mask],
#              model.des_mask:[des_mask]
#            }
#          )

#          print(np.array(src_features).shape, np.array(des_features).shape)
#          src_weight = np.dot(np.array(src_features[0]), np.array(src_features[0,0]))
#          des_weight = np.dot(np.array(des_features[0]), np.array(des_features[0,0]))
#          print(src_weight.shape, des_weight.shape)
#          print(len(src_node), len(des_node))
#          print("src:", [locList[it] for it in src_node], src_weight.tolist())
#          print("des:", [locList[it] for it in des_node], des_weight.tolist())


#          print(one_step_value, h_score)
          h_score = 1.0
#            f_score = random.uniform(0, 10)

#            f_score = f_scores[waiting_count]
#            f_score = f_score+ fScore[current]
          if (waiting in gScore) and (g_score < gScore[waiting]):
            continue
          gScore[waiting] = g_score
          fScore[waiting] = gScore[waiting] + h_score
          if waiting not in openSet:
            openSet = insert(model, waiting, openSet, fScore)
          else:
            openSet = move(model, waiting, openSet,fScore)
          pathFounded[waiting] =  temp

#          print([locList[item_] for item_ in openSet])
#          for item in openSet:
#            print("loc:", locList[item], "score:", fScore[item], "len:", len(pathFounded[item]), "path:", [locList[item_] for item_ in pathFounded[item]], locList[end])
#          print("----------")

#          print(searchCount)
        if(searchCount >= 200):
          openSet = []
#            print(temp)
      count += 1
#      print(allNode)
      print(count, searchCount, bestScore, bestTra, allNode, [locList[node] for node in allNode])
      all_search_count += searchCount
#        break

#        print("----------------------------------------------")
#        print("count:", count)
#        if(len(openSet) == 0 and len(bestTra) == 0):
#      if len(openSet) == 0:
      result.append(bestTra)
#        print("count:", count)
    results.append(result)
    for infer, real in zip(result, tra_bat):
      print("infer:", infer)
      print("real:", real)
      for item in infer[1:-1]:
        if item in real[1:-1]:
          print(item, real[1:-1])  
          counters += 1
      all_len += len(real) - 2
    print(all_search_count, counters, all_len, float(counters)/all_len)
  return float(counters)/all_len
def AstarTest(model):

  results = []
  counters = 0
  all_len = 0
  for batch in testData:
    result = []
    count = 0

    for start, unknown, end in zip(np.array(batch)[:, 0], np.array(batch)[:,1:-1], np.array(batch)[:, -1]):
      closedSet = []
      openSet = [start]
      cameFrom = {}
      pathFounded = {start: [start]}
      gScore = {}
      fScore = {}
      gScore[start] = 0
      fScore[start] = 0
      waitingTra = 0
      bestScore = -10000000
      bestTra = []
      searchCount = 0
      while len(openSet) > 0:
        searchCount += 1
        current = openSet[0]
#          if current == end and len(pathFounded[current]) == len(unknown) + 1:
#            trajec = pathFounded[current]
#            result.append(trajec)
#            break

        openSet.remove(current)
#          print(len(openSet))
        closedSet.append(current)
        if gScore[current] > bestScore:
          bestScore = gScore[current]
          bestTra = copy.deepcopy(pathFounded[current])
          bestTra.append(end)
        if len(pathFounded[current]) > len(unknown) + 1 or len(pathFounded[current]) == len(unknown) + 1:
          print(len(pathFounded[current]))
          continue
        batch_known = []
        batch_end = []
        try:
            for act in OSMadj[current]:
                temp = copy.deepcopy(pathFounded[current])
                temp.append(act)
                batch_known.append(temp)
                batch_end.append(end)
        except:
            print("not exist")
            continue
        st_value = model.infer_st.eval(
            feed_dict={
              model.st_known_:batch_known,
              model.st_destination_:batch_end
        })
        st_value = np.reshape(np.array(st_value), [-1])
#        st_value = np.exp(st_value)/np.sum(np.exp(st_value))
#        policy_value = np.exp(policy_value)/np.sum(np.exp(policy_value), axis=2)[:, :, np.newaxis]
        st_arg = np.argsort(-st_value, axis=0)[:3]
#        st_arg = np.array([i for i in range(len(st_value))])[:2]
        policy_list = OSMadj[current]
############# new value network         
#          p_known_batch = []
#          for action in policy_list:
#            temp = copy.deepcopy(pathFounded[current])
#            temp.append(action)
#            p_known_batch.append(temp)


#          f_scores = self.heuristics.eval(
#              feed_dict={
#                 self.p_known_:np.array(p_known_batch),
#                 self.p_destination_:[end for i in range(len(policy_list))],
#                }
#            )
#############              
#          print("current:", current)
#          print("policy:", policy_list)
#        st_known_batch = []
#        for action in policy_list:
#          st_known_batch.append(pathFounded[current])
#          f_scores = self.heuristics.eval(
#              feed_dict={
#                self.p_known_:np.concatenate((p_known_batch, np.array(policy_list)[:, np.newaxis]), 1),
#                self.p_destination_:[end for i in range(len(policy_list))]
#                }
#            )

#          print("f_scores:", f_scores)

        for ind in range(len(st_arg)):
          waiting_count = st_arg[ind]
          waiting = OSMadj[current][waiting_count]
          if (waiting in closedSet):
            continue

          g_score = (1 - len(pathFounded[current]) / 10) * (st_value[waiting_count]) + gScore[current]#((1 - len(pathFounded[current]) / 20)) * st_value[waiting_count][0] + gScore[current]
#            f_score = np.array(policy_value)[-1, -1, waiting] + fScore[current]           
          temp = copy.deepcopy(pathFounded[current])
          temp.append(waiting)

#          h_score =  self.heuristics.eval(
#              feed_dict={
#                  self.p_known_:np.array(temp)[np.newaxis, :],
#                  self.p_destination_:[end],
#                }
#            )
#          h_score = 100.0
#            f_score = random.uniform(0, 10)

#            f_score = f_scores[waiting_count]
#            f_score = f_score+ fScore[current]
          if (waiting in gScore) and (g_score < gScore[waiting]):
            continue
          gScore[waiting] = g_score
          fScore[waiting] = gScore[waiting] + h_score
          if waiting not in openSet:
            openSet = insert(model, waiting, openSet, fScore)
          else:
            openSet = move(model, waiting, openSet,fScore)
          pathFounded[waiting] =  temp
#            for item in openSet:
#              print("loc:", item, "score:", fScore[item], "len:", len(pathFounded[item]), "path:", pathFounded[item])
#            print("----------")

#          print(searchCount)
        if(searchCount >= 1500):
          openSet = []
#            print(temp)
      count += 1
      print(count, searchCount, bestScore, bestTra)
#        break

#        print("----------------------------------------------")
#        print("count:", count)
#        if(len(openSet) == 0 and len(bestTra) == 0):
      if len(openSet) == 0:
        print("openSet:", len(openSet))
        result.append(bestTra)
#        print("count:", count)
    results.append(result)
    for infer, real in zip(result, batch):
      print("infer:", infer)
      print("real:", real)
      for item in infer[1:-1]:
        if item in real[1:-1]:
          print(item, real[1:-1])  
          counters += 1
      all_len += len(real) - 2
    print(counters, all_len, float(counters)/all_len)
  return float(counters)/all_len

#OSMadj = pickle.load(open("/data/wuning/map-matching/fmm-master/fmm-master/example/ofoOSMadjMat", "rb"))
#allData = pickle.load(open("/data/wuning/map-matching/fmm-master/fmm-master/example/ofoOSMBeijingtrainData", "rb"))
#testData = []
#for batch in allData[2300:]:
#  for i in range(0, len(batch[0]), 13):
#      if i + 13 < len(batch[0]):
#          testData.append(np.array(batch)[:, i:i+13])



trainData = pickle.load(open("/data/wuning/map-matching/taxiTrainData_", "rb"))
trainTimeData = pickle.load(open("/data/wuning/map-matching/taxiTrainDataTime_", "rb"))
trainUserData = pickle.load(open("/data/wuning/map-matching/taxiTrainDataUser_", "rb"))
historyData = pickle.load(open("/data/wuning/map-matching/userIndexedHistoryAttention", "rb"))
maskData = pickle.load(open("/data/wuning/map-matching/taxiTrainDataMask", "rb"))
graphData = pickle.load(open("/data/wuning/map-matching/allGraph", "rb"))
graphData = pickle.load(open("/data/wuning/map-matching/CompleteAllGraph", "rb"))

locList = pickle.load(open("/data/wuning/map-matching/fmm-master/fmm-master/example/locList", "rb"))

locList = pickle.load(open("/data/wuning/map-matching/fmm-master/fmm-master/example/CompletelocList", "rb"))

#trainData = pickle.load(open("/data/wuning/map-matching/fmm-master/fmm-master/example/portoHDRTrainData", "rb"))
trainData = pickle.load(open("/data/wuning/map-matching/fmm-master/fmm-master/example/OSMBeijingCompleteTrainData_50", "rb"))

#testData = []
#for batch in allData[6000:]:
#  for i in range(0, len(batch[0]), 13):
#      if i + 13 < len(batch[0]):
#          testData.append(np.array(batch)[:, i:i+13])

def main():
    AstarRNN = DAN()
    AstarRNN.all_saver.restore(tf.get_default_session(), "/data/wuning/learnAstar/beijingComplete/pre_all_train_neural_network_epoch16.ckpt")
    #/data/wuning/learnAstar/beijingComplete/pre_all_train_neural_network_epoch16.ckpt
    #/data/wuning/learnAstar/beijingComplete/pre_all_train_neural_network_epoch41.ckpt
    #/AstarRNN/train_complete_heuristics_ST_two_task_epoch37.ckpt
    #/data/wuning/learnAstar/porto/pre_all_train_neural_network_epoch49.ckpt
    #AstarRNN/train_heuristics_TD1_two_task_epoch0.ckpt       train_complete_heuristics_TD1_two_task_epoch0.ckpt
    #learnAstar/pre_train_neural_network_epoch2.ckpt   pre_all_train_neural_network_epoch49.ckpt   pre_all_his_train_neural_network_epoch39.ckpt
    #AstarRNN/train_heuristics_TD1_two_task_epoch0.ckpt  train_heuristics_supver_two_task_epoch2.ckpt
    AstarTestSoftmax(AstarRNN)
#    admissibleHeu(AstarRNN)    
#    visGraphAttention(AstarRNN)
#    greedyTest(AstarRNN)
if __name__ == '__main__':
    main()

