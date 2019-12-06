import tensorflow as tf
import numpy as np
import networkx as nx
import random
import pickle
import copy
from collections import deque
import os
import math
from model import *
from utils import *

os.environ['CUDA_VISIBLE_DEVICES']='0'
# Hyper Parameters for DAN
PRE_TRAIN = False
TEST = True
RESTORE = True
GAMMA = 1.0 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
batch_size = None # size of minibatch
input_steps = None
block_num = 16500
lstm_size = 384
num_layers = 2
TRAIN_BATCH_SIZE = 100 #训练输入的batch 大小
INFERENCE_BATCH_SIZE = 1 #推断的时候输入的batch 大小
PRE_EPISODE = 600
NEG_SAMPLES = 9
NEXT_ACTION_NUM = 3

EPISODE = 100 # Episode limitation
PRE_EPISODE = 300
TRAIN_BATCHES = 300 # Step limitation in an episode

def train_heuristics_network(self):
    self.time_step += 1
# Step 1: obtain random minibatch from replay memory
    minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
    state_batch = [data[0] for data in minibatch]
    action_batch = [data[1] for data in minibatch]
    reward_batch = [data[2] for data in minibatch]
    next_state_batch = [data[3] for data in minibatch]

# Step 2: calculate y
    y_batch = []
    Q_value_batch = self.Q_value.eval(feed_dict={self.state_input:next_state_batch})
    for i in range(0,BATCH_SIZE):
        done = minibatch[i][4]
        if done:
            y_batch.append(reward_batch[i])
        else:
            y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

            self.optimizer.run(feed_dict={
                    self.y_input:y_batch,
                    self.action_input:action_batch,
                    self.state_input:state_batch
                    })
def pre_train(model, PRE_EPISODE):
#    self.all_saver.restore(self.session, "/data/wuning/AstarRNN/pretrain_test_policity_neural_network_epoch0.ckpt")
    OSMadj = pickle.load(open("/data/wuning/map-matching/fmm-master/fmm-master/example/ofoOSMadjMat", "rb"))
    trainData = pickle.load(open("/data/wuning/map-matching/taxiTrainData_", "rb"))
    trainTimeData = pickle.load(open("/data/wuning/map-matching/taxiTrainDataTime_", "rb"))
    trainUserData = pickle.load(open("/data/wuning/map-matching/taxiTrainDataUser_", "rb"))
    historyData = pickle.load(open("/data/wuning/map-matching/userIndexedHistoryAttention", "rb"))
    maskData = pickle.load(open("/data/wuning/map-matching/taxiTrainDataMask", "rb"))
#    testData = pickle.load(open("/data/wuning/map-matching/fmm-master/fmm-master/example/ofoOSMBeijingTestData", "rb"))
    trainData = pickle.load(open("/data/wuning/map-matching/fmm-master/fmm-master/example/portoHDRTrainData", "rb"))   
    trainData = pickle.load(open("/data/wuning/map-matching/fmm-master/fmm-master/example/OSMBeijingCompleteTrainData_", "rb")) 
    for episode in range(PRE_EPISODE):
      for tra_bat, hour_bat, day_bat, his_bat, his_hour_bat, his_day_bat, his_mask_bat in generate_batch(maskData[:700], historyData, trainData[:700], trainTimeData[:700], trainUserData[:700]):
        counter = 0
        if(len(tra_bat[0]) < 5 or len(tra_bat[0]) > 200):
          continue
#        print(tra_bat.shape, tra_mask_bat.shape, hour_bat.shape, day_bat.shape, his_bat.shape, his_hour_bat.shape, his_day_bat.shape, his_mask_bat.shape)
        _, eval_st_loss = model.session.run([model.st_all_optimizer, model.st_all_cost],feed_dict={
          model.st_known_:tra_bat[:, :-1],
          model.st_destination_:tra_bat[:, -1][:, np.newaxis],
          model.st_output_:tra_bat[:, 1:],
#          model.trans_mat:batch[3]
          model.st_time:hour_bat,
          model.st_day:day_bat,
#          model.padding_mask:tra_mask_bat,
          model.his_tra:his_bat,
          model.his_time:his_hour_bat,
          model.his_day:his_day_bat,
          model.his_padding_mask:his_mask_bat
        })
#        eval_st_loss = model.st_cost.eval(feed_dict={
#          model.st_known_:batch[0],
#          model.st_destination_:batch[2],
#          model.st_output_:batch[1],
#          model.trans_mat:batch[3]
#        })

        if counter % 1 == 0:
          print("epoch:{}...".format(episode),
            "batch:{}...".format(counter),
            "loss:{:.4f}...".format(eval_st_loss))
        print(counter)
        counter += 1
      model.all_saver.save(model.session, "/data/wuning/learnAstar/beijingComplete/pre_all_train_neural_network_epoch{}.ckpt".format(episode))  
'''
    pre_batches = []
    pre_test_batches = []
    trans_mask = [0 for i in range(block_num)]
    count = 0
    for batch in trainData:
        print(count)
        count += 1
        if len(batch) > 0 and len(batch[0]) < 100:
            batch_trans_mask = []
            for tra in np.array(batch)[:, :-1]:
              tra_mask = []
              for item in tra:
                  mask = []
                  for nex in OSMadj[item]:
                      mask.append(int(nex))
                  if len(mask) > 70:
                      mask = mask[:70]
                  while len(mask) < 70:
                      mask.append(-1)
                  tra_mask.append(mask)
              batch_trans_mask.append(tra_mask)  
            pre_batches.append([np.array(batch)[:,:-1], np.array(batch)[:,1:], np.array(batch)[:,-1], np.array(batch_trans_mask)])
         
    for test_batch in trainData[0:100]:
        if len(test_batch) > 0 and len(test_batch[0]) < 100:
            batch_trans_mask = []
            for tra in np.array(batch)[:, :-1]:
              tra_mask = []
              for item in tra:
                  mask = []
                  for nex in OSMadj[item]:
                      mask.append(int(nex))
                  if len(mask) > 70:
                      mask = mask[:70]
                  while len(mask) < 70:
                      mask.append(-1)
                  tra_mask.append(mask)
              batch_trans_mask.append(tra_mask)  
            pre_test_batches.append([np.array(test_batch)[:,:-1], np.array(test_batch)[:,1:], np.array(test_batch)[:,-1], np.array(batch_trans_mask)])
'''
#    pre_batches = np.array(pre_batches)
'''
    print(len(pre_batches), len(pre_test_batches))
    for episode in range(PRE_EPISODE):
        counter = 0
        for batch in pre_batches[0:2300]:
            if(batch[0].shape[1] < 5):
                continue

            model.st_optimizer.run(feed_dict={
                model.st_known_:batch[0],
                model.st_destination_:batch[2],
                model.st_output_:batch[1],
                model.trans_mat:batch[3]
                })
            eval_st_loss = model.st_cost.eval(feed_dict={
                model.st_known_:batch[0],
                model.st_destination_:batch[2],
                model.st_output_:batch[1],
                model.trans_mat:batch[3]
                })

            if counter % 1 == 0:
                print("epoch:{}...".format(episode),
                "batch:{}...".format(counter),
                "loss:{:.4f}...".format(eval_st_loss))
            print(counter)
            counter += 1
'''
#        average_loss = 0
#        for test_batch in pre_test_batches:
#            if(test_batch[0].shape[1] < 5):
#                continue
#
#            test_st_loss = model.st_cost.eval(feed_dict={
#            model.st_known_:test_batch[0],
#            model.st_destination_:test_batch[2],
#            model.st_output_:test_batch[1],
#            model.trans_mat:test_batch[3]
#         })
#        average_loss += test_st_loss
#        print("test_loss:", average_loss / len(pre_test_batches))
#        model.all_saver.save(model.session, "/data/wuning/learnAstar/pre_all_train_neural_network_epoch{}.ckpt".format(episode))

def ST(model):

    model.st_saver.restore(model.session, "/data/wuning/learnAstar/beijingComplete/pre_all_train_neural_network_epoch49.ckpt")
    OSMadj = pickle.load(open("/data/wuning/map-matching/fmm-master/fmm-master/example/ofoOSMadjMat", "rb"))
    trainData = pickle.load(open("/data/wuning/map-matching/taxiTrainData_", "rb"))
    trainTimeData = pickle.load(open("/data/wuning/map-matching/taxiTrainDataTime_", "rb"))
    trainUserData = pickle.load(open("/data/wuning/map-matching/taxiTrainDataUser_", "rb"))
    historyData = pickle.load(open("/data/wuning/map-matching/userIndexedHistoryAttention", "rb"))
    maskData = pickle.load(open("/data/wuning/map-matching/taxiTrainDataMask", "rb"))
    graphData = pickle.load(open("/data/wuning/map-matching/allGraph", "rb"))

    trainData = pickle.load(open("/data/wuning/map-matching/fmm-master/fmm-master/example/OSMBeijingCompleteTrainData_", "rb"))
#    variable_names = [v.name for v in tf.trainable_variables()]
#    print(variable_names)
    location_embeddings = model.session.run(tf.get_default_graph().get_tensor_by_name("st_network/location_embedding/embeddings:0"))
    print(np.array(location_embeddings).shape)

    adj = np.matrix(graphData)[:block_num, :block_num]
    print(adj.shape)
    G = nx.from_numpy_matrix(adj, create_using=nx.DiGraph())
    inv_G = nx.from_numpy_matrix(adj.T, create_using=nx.DiGraph())
    train_size = 1500
    losses = []                                    
    for episode in range(PRE_EPISODE):
      for tra_bat, hour_bat, day_bat, his_bat, his_hour_bat, his_day_bat, his_mask_bat in generate_batch(maskData[:train_size], historyData, trainData[:train_size], trainTimeData[:train_size], trainUserData[:train_size]):
        counter = 0
        heuristics_batches = []
        if(len(tra_bat[0]) < 5):
          continue
#        print(tra_bat.shape, tra_mask_bat.shape, hour_bat.shape, day_bat.shape, his_bat.shape, his_hour_bat.shape, his_day_bat.shape, his_mask_bat.shape)
        for k in range(len(tra_bat[0]) - 1, 0, -1):
          item_heu_batch = []
          for ite in tra_bat:
            item_heu_batch.append(float(len(tra_bat[0]) - k))      

          item_known_batch = tra_bat[:, :k]
          item_des_batch = tra_bat[:, -1]

          heuristics_batches.append([item_known_batch, item_des_batch, item_heu_batch])
        feed_data = {}
        for heu_batch in heuristics_batches:

#            print(np.array(heu_batch[0]).shape)
#            print(np.array(heu_batch[1]).shape)
#            print(np.array(heu_batch[2]).shape)
#            print(np.array(heu_batch[3]).shape)
#            print(np.array(heu_batch[4]).shape)
#            print(np.array(heu_batch[5]).shape)
#            print(np.array(heu_batch[6]).shape)
#            print(np.array(heu_batch[7]).shape)
#            print(np.array(heu_batch[8]).shape)

            model.optimizer.run(feed_dict = {
                model.st_known_:heu_batch[0],
                model.st_destination_:np.array(heu_batch[1])[:, np.newaxis],
                model.heuristics_input:heu_batch[2],
#                model.src_bias_mat:heu_batch[3],
#                model.des_bias_mat:heu_batch[4],
#                model.src_embedding:heu_batch[5],
#                model.des_embedding:heu_batch[6],
#                model.src_mask:heu_batch[7],
#                model.des_mask:heu_batch[8]
                }
            )
#                _ = model.st_all_optimizer.run(feed_dict=policy_feed_data)
            heuristics_cost = model.heuristics_cost.eval(feed_dict={
                model.st_known_:heu_batch[0],
                model.st_destination_:np.array(heu_batch[1])[:, np.newaxis],
                model.heuristics_input:heu_batch[2],
#                model.src_bias_mat:heu_batch[3],
#                model.des_bias_mat:heu_batch[4],
#                model.src_embedding:heu_batch[5],
#                model.des_embedding:heu_batch[6],
#                model.src_mask:heu_batch[7],
#                model.des_mask:heu_batch[8]
            })
        heuristics = model.heuristics.eval(feed_dict={
                model.st_known_:heu_batch[0],
                model.st_destination_:np.array(heu_batch[1])[:, np.newaxis],
                model.heuristics_input:heu_batch[2],
#                model.src_bias_mat:heu_batch[3],
#                model.des_bias_mat:heu_batch[4],
#                model.src_embedding:heu_batch[5],
#                model.des_embedding:heu_batch[6],
#                model.src_mask:heu_batch[7],
#                model.des_mask:heu_batch[8]
        })
#        print("heuristics:", heuristics)
        heuristics_batches = []
        losses.append(heuristics_cost)
        print("loss:", heuristics_cost, "counter:", counter)
      print(losses)
      print("heuristics:", heuristics)
      model.all_saver.save(model.session, "/data/wuning/AstarRNN/train_complete_heuristics_ST_two_task_epoch{}.ckpt".format( episode))
def Time_diff(model):
#    model.st_saver.restore(model.session, "/data/wuning/learnAstar/pre_all_train_neural_network_epoch49.ckpt")
#/data/wuning/learnAstar/beijingComplete/pre_all_train_neural_network_epoch
    model.st_saver.restore(model.session, "/data/wuning/learnAstar/beijingComplete/pre_all_train_neural_network_epoch49.ckpt")
    OSMadj = pickle.load(open("/data/wuning/map-matching/fmm-master/fmm-master/example/ofoOSMadjMat", "rb"))
    trainData = pickle.load(open("/data/wuning/map-matching/taxiTrainData_", "rb"))
    trainTimeData = pickle.load(open("/data/wuning/map-matching/taxiTrainDataTime_", "rb"))
    trainUserData = pickle.load(open("/data/wuning/map-matching/taxiTrainDataUser_", "rb"))
    historyData = pickle.load(open("/data/wuning/map-matching/userIndexedHistoryAttention", "rb"))
    maskData = pickle.load(open("/data/wuning/map-matching/taxiTrainDataMask", "rb"))
    graphData = pickle.load(open("/data/wuning/map-matching/allGraph", "rb"))

    trainData = pickle.load(open("/data/wuning/map-matching/fmm-master/fmm-master/example/OSMBeijingCompleteTrainData_", "rb"))
#    variable_names = [v.name for v in tf.trainable_variables()]
#    print(variable_names)
    location_embeddings = model.session.run(tf.get_default_graph().get_tensor_by_name("st_network/location_embedding/embeddings:0"))
    print(np.array(location_embeddings).shape)

    adj = np.matrix(graphData)[:block_num, :block_num]
    print(adj.shape)
    G = nx.from_numpy_matrix(adj, create_using=nx.DiGraph())
    inv_G = nx.from_numpy_matrix(adj.T, create_using=nx.DiGraph())
    train_size = 1500
                                        
    for episode in range(PRE_EPISODE):
      for tra_bat, hour_bat, day_bat, his_bat, his_hour_bat, his_day_bat, his_mask_bat in generate_batch(maskData[:train_size], historyData, trainData[:train_size], trainTimeData[:train_size], trainUserData[:train_size]):
        counter = 0
        heuristics_batches = []
        if(len(tra_bat[0]) < 5):
          continue
#        print(tra_bat.shape, tra_mask_bat.shape, hour_bat.shape, day_bat.shape, his_bat.shape, his_hour_bat.shape, his_day_bat.shape, his_mask_bat.shape)
        eval_policy = model.session.run([model.st_all_prob],feed_dict={
          model.st_known_:tra_bat[:, :-1],
          model.st_destination_:tra_bat[:, -1][:, np.newaxis],
          model.st_output_:tra_bat[:, 1:],
#          model.trans_mat:batch[3]
          model.st_time:hour_bat,
          model.st_day:day_bat,
 #         model.padding_mask:tra_mask_bat,
          model.his_tra:his_bat,
          model.his_time:his_hour_bat,
          model.his_day:his_day_bat,
          model.his_padding_mask:his_mask_bat
        })
        eval_policy = np.array(eval_policy[0])
        wait_next_actions = np.argsort(-eval_policy, axis=2)[:, :, :NEXT_ACTION_NUM]
        policy_value =  -np.sort(-eval_policy, axis=2)[:, :, :NEXT_ACTION_NUM]
        sum_heu_batch = []
        for k in range(wait_next_actions.shape[1], 0, -1):
            item_heu_batch = []
            item_known_batch = []
            item_action_batch = []
            item_des_batch = []
            for l in range(0, NEXT_ACTION_NUM):
                item_heu_batch.extend(policy_value[:, k - 1, l].tolist())
                item_known_batch.extend(tra_bat[:, :-1][:, :k])
                item_action_batch.extend(wait_next_actions[:, k - 1, l].tolist())
                item_des_batch.extend(tra_bat[:, -1])
#                print(np.array(item_known_batch).shape, np.array(item_heu_batch).shape, np.array(item_action_batch).shape, np.array(item_des_batch).shape)
#            print(np.array(item_known_batch).shape, np.array(item_action_batch)[:, np.newaxis].shape)
            item_known_batch_ = np.concatenate((item_known_batch, np.array(item_action_batch)[:, np.newaxis]), axis=1)
            if not k == wait_next_actions.shape[1]:
#                    if len(sum_heu_batch) == 0:
#                        sum_heu_batch = np.zeros(np.array(item_heu_batch).shape)
#                    sum_heu_batch += item_heu_batch
                last_adj = []
                src_adj = []
                des_adj = []
                last_emb = []
                src_emb = []
                des_emb = []
                last_mask = []
                des_mask = []
                src_mask = []
                for last, src, des in zip(np.array(item_known_batch)[:, -1], item_action_batch, item_des_batch):
                  item_1, item_2, item_3, item_4, item_5, item_6, _, _ = generate_sub_graph(location_embeddings, G, inv_G, 10, src=src, des=des)
                  l_item_1, l_item_2, l_item_3 = generate_one_graph(location_embeddings, G, 10, src=last)
                  src_adj.append(item_1)
                  des_adj.append(item_2)
                  des_emb.append(item_3)
                  src_emb.append(item_4)  
                  des_mask.append(item_5)
                  src_mask.append(item_6)
                  last_adj.append(l_item_1)
                  last_emb.append(l_item_2)
                  last_mask.append(l_item_3)
#                print("mask:", np.array(src_mask).shape, np.array(des_mask).shape, np.array(src_emb).shape, np.array(des_emb).shape, np.array(src_adj).shape, np.array(des_adj).shape)  
                item_heu_batch += 0.92 * model.heuristics.eval(
                    feed_dict={
                        model.st_known_:item_known_batch_,
                        model.st_destination_:np.array(item_des_batch)[:, np.newaxis],
                        model.src_bias_mat:src_adj,
                        model.des_bias_mat:des_adj,
                        model.src_embedding:src_emb,
                        model.des_embedding:des_emb,
                        model.src_mask:src_mask,
                        model.des_mask:des_mask
                    }
                )
#                    print(sum_heu_batch)
                heuristics_batches.append([item_known_batch, item_des_batch, item_heu_batch, last_adj, des_adj, last_emb, des_emb, last_mask, des_mask])
        feed_data = {}
        for heu_batch in heuristics_batches:

#            print(np.array(heu_batch[0]).shape)
#            print(np.array(heu_batch[1]).shape)
#            print(np.array(heu_batch[2]).shape)
#            print(np.array(heu_batch[3]).shape)
#            print(np.array(heu_batch[4]).shape)
#            print(np.array(heu_batch[5]).shape)
#            print(np.array(heu_batch[6]).shape)
#            print(np.array(heu_batch[7]).shape)
#            print(np.array(heu_batch[8]).shape)

            model.optimizer.run(feed_dict = {
                model.st_known_:heu_batch[0],
                model.st_destination_:np.array(heu_batch[1])[:, np.newaxis],
                model.heuristics_input:heu_batch[2],
                model.src_bias_mat:heu_batch[3],
                model.des_bias_mat:heu_batch[4],
                model.src_embedding:heu_batch[5],
                model.des_embedding:heu_batch[6],
                model.src_mask:heu_batch[7],
                model.des_mask:heu_batch[8]
                }
            )
#                _ = model.st_all_optimizer.run(feed_dict=policy_feed_data)
            heuristics_cost = model.heuristics_cost.eval(feed_dict={
                model.st_known_:heu_batch[0],
                model.st_destination_:np.array(heu_batch[1])[:, np.newaxis],
                model.heuristics_input:heu_batch[2],
                model.src_bias_mat:heu_batch[3],
                model.des_bias_mat:heu_batch[4],
                model.src_embedding:heu_batch[5],
                model.des_embedding:heu_batch[6],
                model.src_mask:heu_batch[7],
                model.des_mask:heu_batch[8]
            })
        heuristics = model.heuristics.eval(feed_dict={
                model.st_known_:heu_batch[0],
                model.st_destination_:np.array(heu_batch[1])[:, np.newaxis],
                model.heuristics_input:heu_batch[2],
                model.src_bias_mat:heu_batch[3],
                model.des_bias_mat:heu_batch[4],
                model.src_embedding:heu_batch[5],
                model.des_embedding:heu_batch[6],
                model.src_mask:heu_batch[7],
                model.des_mask:heu_batch[8]
        })
#        print("heuristics:", heuristics)
        heuristics_batches = []

        print("loss:", heuristics_cost, "counter:", counter)
      print("heuristics:", heuristics)
      model.all_saver.save(model.session, "/data/wuning/AstarRNN/train_complete_heuristics_TD1_two_task_epoch{}.ckpt".format( episode))

#        eval_st_loss = model.st_cost.eval(feed_dict={
#          model.st_known_:batch[0],
#          model.st_destination_:batch[2],
#          model.st_output_:batch[1],
#          model.trans_mat:batch[3]
#        })

#        if counter % 1 == 0:
#          print("epoch:{}...".format(episode),
#            "batch:{}...".format(counter),
#            "loss:{:.4f}...".format(eval_st_loss))
#        print(counter)
#        counter += 1
#      model.all_saver.save(model.session, "/data/wuning/learnAstar/pre_all_train_neural_network_epoch{}.ckpt".format(episode))  

def Q_learning_train_two_task(model):
    model.st_saver.restore(model.session, "/data/wuning/learnAstar/pre_all_train_neural_network_epoch13.ckpt")

    OSMadj = pickle.load(open("/data/wuning/map-matching/fmm-master/fmm-master/example/ofoOSMadjMat", "rb"))
    trainData = pickle.load(open("/data/wuning/map-matching/taxiTrainData", "rb"))
    trainTimeData = pickle.load(open("/data/wuning/map-matching/taxiTrainDataTime", "rb"))
    trainUserData = pickle.load(open("/data/wuning/map-matching/taxiTrainDataUser", "rb"))
    historyData = pickle.load(open("/data/wuning/map-matching/userIndexedHistoryAttention", "rb"))
    maskData = pickle.load(open("/data/wuning/map-matching/taxiTrainDataMask", "rb"))
    graphData = pickle.load(open("/data/wuning/map-matching/graphData", "rb"))
    location_embeddings = tf.get_default_graph().get_tensor_by_name("location_embedding")
   

#    OSMadj = pickle.load(open("/data/wuning/map-matching/fmm-master/fmm-master/example/ofoOSMadjMat", "rb"))
#    trainData = pickle.load(open("/data/wuning/map-matching/fmm-master/fmm-master/example/ofoOSMBeijingtrainData", "rb"))
#    testata = pickle.load(open("/data/wuning/map-matching/fmm-master/fmm-master/example/ofoOSMBeijingTestData", "rb"))

    pre_batches = []

    for batch in trainData:
        if len(batch) > 0 and len(batch[0]) > 3:
            pre_batches.append([np.array(batch)[:,:-1], np.array(batch)[:,1:], np.array(batch)[:,-1]])
    counter = 0
    for episode in range(EPISODE):
        for batch in pre_batches:
            counter += 1
            heuristics_batches = []
            if len(batch) == 0:
                continue
            policy_feed_data = {
                model.st_known_:batch[0],
                model.st_destination_:batch[2],
                model.st_output_:batch[1]
            }
            eval_policy = model.st_all_prob.eval(feed_dict=policy_feed_data)
            wait_next_actions = np.argsort(-eval_policy, axis=2)[:, :, :NEXT_ACTION_NUM]
            policy_value =  -np.sort(-eval_policy, axis=2)[:, :, :NEXT_ACTION_NUM]
            sum_heu_batch = []
            for k in range(wait_next_actions.shape[1], 0, -1):
                item_heu_batch = []
                item_known_batch = []
                item_action_batch = []
                item_des_batch = []
                for l in range(0, NEXT_ACTION_NUM):
                    item_heu_batch.extend(policy_value[:, k - 1, l].tolist())
                    item_known_batch.extend(batch[0][:, :k])
                    item_action_batch.extend(wait_next_actions[:, k - 1, l].tolist())
                    item_des_batch.extend(batch[2])
#                print(np.array(item_known_batch).shape, np.array(item_heu_batch).shape, np.array(item_action_batch).shape, np.array(item_des_batch).shape)
                item_known_batch_ = np.concatenate((item_known_batch, np.array(item_action_batch)[:, np.newaxis]), axis=1)
                if not k == wait_next_actions.shape[1]:
#                    if len(sum_heu_batch) == 0:
#                        sum_heu_batch = np.zeros(np.array(item_heu_batch).shape)
#                    sum_heu_batch += item_heu_batch
                    item_heu_batch += 0.92 * model.heuristics.eval(
                        feed_dict={
                          model.st_known_:item_known_batch_,
                          model.st_destination_:item_des_batch
                        }
                    )
#                    print(sum_heu_batch)
                    heuristics_batches.append([item_known_batch, item_des_batch, item_heu_batch ])
            feed_data = {}
            for heu_batch in heuristics_batches:
                feed_data = {
                    model.st_known_:heu_batch[0],
                    model.st_destination_:heu_batch[1],
                    model.heuristics_input:heu_batch[2]
                }
                model.optimizer.run(feed_dict=feed_data)
#                _ = model.st_all_optimizer.run(feed_dict=policy_feed_data)
                heuristics_cost = model.heuristics_cost.eval(feed_dict=feed_data)
            heuristics = model.heuristics.eval(feed_dict=feed_data)
#        print("heuristics:", heuristics)
            heuristics_batches = []

            print("loss:", heuristics_cost, "counter:", counter)
        print("heuristics:", heuristics)
        model.all_saver.save(model.session, "/data/wuning/AstarRNN/train_heuristics_TD1_two_task_epoch{}.ckpt".format( episode))
def Q_learning_train(self):
    heuristics_batches = []
    counter = 0
    for episode in range(EPISODE):
        for i in range(500, 12500, 500):
            Q_learning_batches = pickle.load(open("/data/wuning/mobile trajectory/Q_learning_heuristicsTrainSet"+str(i), "rb"))
            for batch in Q_learning_batches:
                policy_feed_data = {
                    self.p_known_:np.concatenate((np.array(batch[0]), np.array(batch[1])[:, np.newaxis]), 1),
                    self.p_destination_:batch[2]                    }
    eval_next_policy = self.policy.eval(feed_dict=policy_feed_data)
    wait_next_actions = np.argsort(eval_next_policy, axis=1)[:, -1, :NEXT_ACTION_NUM]
    heuristics_batch = []  #下一个状态的Q值
    for k in range(NEXT_ACTION_NUM):
        heuristics_batch.append(self.heuristics.eval(
                feed_dict={
                self.known_:np.concatenate((np.array(batch[0]), np.array(batch[1])[:, np.newaxis]), 1),
                self.waiting_:wait_next_actions[:, k],
                self.destination_:batch[2]
                }
                )
            )
    heuristics_batch = np.array(heuristics_batch)
    print("heuristics_batch:", heuristics_batch[2, :], len(heuristics_batch))
    print("heuristics_ave:", np.mean(heuristics_batch[0, ]), np.mean(heuristics_batch[1, :]), np.mean(heuristics_batch[2, :]))
    heuristics_batch = np.max(heuristics_batch, axis = 0)
    batch[3] += GAMMA * heuristics_batch[:, 0]
    heuristics_batches.append([np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), batch[3]])

    for batch in heuristics_batches:
        feed_data = {
            self.known_:batch[0],
            self.waiting_:batch[1],
            self.destination_:batch[2],
            self.heuristics_input:batch[3]
        }
    self.optimizer.run(feed_dict=feed_data)
    heuristics_cost = self.heuristics_cost.eval(feed_dict=feed_data)
    heuristics = self.heuristics.eval(feed_dict=feed_data)
    print("heuristics:", heuristics)
    self.all_saver.save(self.session, "/data/wuning/AstarRNN/train_heuristics_TD1_step{}_epoch{}.ckpt".format(i, episode))
    heuristics_batches = []
    print("loss:", heuristics_cost)
def supervised_train(self):
    for episode in range(EPISODE):
        counter = 0
        for i in range(499, 2000, 500):
#        heuristics_batches = pickle.load(open("/data/wuning/AstarBeijing/beijing_Q_learning_serpervised_heuristicsTrainSet"+str(i), "rb"))
            heuristics_batches = []
    heuristics_policy_batches = pickle.load(open("/data/wuning/AstarBeijing/beijing_Q_learning_policy_surpervised_length_heuristicsTestSet"+str(i), "rb"))
    heuristics_batches.extend(heuristics_policy_batches)
#mobile trajectory/Q_learning_serpervised_heuristicsTrainSet

    for batch in heuristics_batches:
        feed_data = {
            self.p_known_:np.concatenate((np.array(batch[0]), np.array(batch[1])[:, np.newaxis]), 1),
            self.p_destination_:batch[2],
            self.heuristics_input:batch[3]
        }


    self.optimizer.run(feed_dict=feed_data)
    heuristics_cost = self.heuristics_cost.eval(feed_dict=feed_data)
    heuristics_value = self.heuristics.eval(feed_dict=feed_data)
#          heuristics_grad = self.gradients[0].eval(feed_dict=feed_data)
    if counter % 2001 == 0:
#            print("y_batch:", batch[3])
        print("epoch:{}...".format(episode),
            "batch:{}...".format(counter),
            "heuristics{}...".format(heuristics_value),
#                  "grads{}...".format(heuristics_grad),
            "loss:{:.4f}...".format(heuristics_cost))
    counter += 1
    self.all_saver.save(self.session, "/data/wuning/AstarRNN/beijing_supervised_train_heuristics_neural_network_epoch{}.ckpt".format(episode))
def margin_loss_st_train(model):
    st_batches = []
    neg_batches = []
    OSMadj = pickle.load(open("/data/wuning/map-matching/fmm-master/fmm-master/example/ofoOSMadjMat", "rb"))
    trainData = pickle.load(open("/data/wuning/map-matching/fmm-master/fmm-master/example/ofoOSMBeijingtrainData", "rb")) 
#    model.all_saver.restore(tf.get_default_session(), "/data/wuning/learnAstar/only_RNN_satisfication_1.ckpt")

    for episode in range(EPISODE):
        print("generating new training data in epoch:", episode)
        st_batches = []
        neg_batches = []
        for i in range(len(trainData)):
            if i % 100 == 0:
                print(i)
            st_batch = []
            st_batch.append(np.array(trainData[i]))
            st_batch.append(np.array(trainData[i])[:, -1])
            neg_batch = []
            for tra in st_batch[0]:
                neg_tra = [tra[0]]
                for k in range(len(tra) - 2):
                    temp_neig = copy.deepcopy(OSMadj[tra[k]])
                    temp_neig.remove(tra[k+1])
                    if len(temp_neig) == 0:
                        temp_neig.append(random.choice(list(OSMadj.keys())))
                    neg_tra.append(random.sample(temp_neig, 1)[0])
                neg_tra.append(tra[-1])
                neg_batch.append(neg_tra)
            st_batches.append(st_batch)
            neg_batches.append(neg_batch)
        st_feed_data = {}
        counter = 0
        print("start training...")
        for st_batch, neg_batch in zip(st_batches, neg_batches):
            if(st_batch[0].shape[1] < 5):
                continue
            st_feed_data = {
                model.st_known_:np.array(st_batch[0]),
                model.neg_known_:np.array(neg_batch),
                model.st_destination_:np.array(st_batch[1]),
            }
            model.st_optimizer.run(feed_dict=st_feed_data)
            eval_st_loss = model.st_cost.eval(feed_dict=st_feed_data)
            st_value = model.st.eval(feed_dict=st_feed_data)
            neg_value = model.neg.eval(feed_dict=st_feed_data)
#          heuristics_grad = self.gradients[0].eval(feed_dict=feed_data)
            if counter % 2001 == 0:
                print("epoch:{}...".format(episode),
                      "batch:{}...".format(counter),
                      "satisfication{}...".format(st_value),
                      "st_mean{}...".format(np.mean(np.array(st_value))),
                      "neg_satisfication{}...".format(neg_value),
                      "neg_mean{}...".format(np.mean(np.array(neg_value))),
                      "loss:{}...".format(eval_st_loss))
            counter += 1
        model.all_saver.save(model.session, "/data/wuning/learnAstar/only_RNN_satisfication_{}.ckpt".format(episode))
def heuristics_train(self):
# generate samples
#    self.all_saver.restore(self.session, "/data/wuning/AstarRNN/pretrain_policity_neural_network_epoch29.ckpt")

        for episode in range(EPISODE):
#      if counter % 100 == 0:
#        print(len(heuristics_batches))
#        print("batches generated:{}...".format(counter))
#      if counter % 5000 == 0:
            counter = 0
            for i in range(500, 12500, 500):
                heuristics_batches = pickle.load(open("/data/wuning/mobile trajectory/heuristicsTrainSet"+str(i), "rb"))
                for j in range(0, len(heuristics_batches), 2):#:batch in heuristics_batches:
#          print("shape:", np.array(heuristics_batches[j: j+2]).shape)
                                                                batch = heuristics_batches[j]
                                                                neg_batch = heuristics_batches[j + 1]
#          print("batch_size",len(heuristics_batches[0][0]))
        neg_batch[0] = np.concatenate((neg_batch[0], neg_batch[1][:, np.newaxis]), 1)
        neg_batch[0] = neg_batch[0].tolist()

        batch[0] = np.concatenate((batch[0], batch[1][:, np.newaxis]), 1)
        batch[0] = batch[0].tolist()
        batch[0].extend(neg_batch[0])
        batch[1] = batch[1].tolist()
        batch[1].extend(neg_batch[1])
        batch[2] = batch[2].tolist()
        batch[2].extend(neg_batch[2])
        batch[3] = batch[3].tolist()
        batch[3].extend(neg_batch[3])
        feed_data = {
#            self.known_:batch[0],
#            self.waiting_:batch[1],
#            self.destination_:batch[2],
#            self.heuristics_input:batch[3]
            self.p_known_:batch[0],
            self.p_destination_:batch[2],
        }

        self.optimizer.run(feed_dict=feed_data)
        heuristics_cost = self.heuristics_cost.eval(feed_dict=feed_data)
        heuristics_value = self.heuristics.eval(feed_dict=feed_data)
#          heuristics_grad = self.gradients[0].eval(feed_dict=feed_data)
        if counter % 501 == 0:
            print("y_batch:", batch[3])
            print("epoch:{}...".format(episode),
                "batch:{}...".format(counter),
                "heuristics{}...".format(heuristics_value),
#                  "grads{}...".format(heuristics_grad),
                "loss:{:.4f}...".format(heuristics_cost))
        counter += 1
        self.all_saver.save(self.session, "/data/wuning/AstarRNN/train_heuristics_reward_neural_network_epoch{}.ckpt".format(episode))

# Test every 100 episodes
        if (episode + 1) % 1000 == 0:
            accuracy = self.AstarTest()
        print("samples num:", len(heuristics_batches))
        print("samples num:", len(heuristics_batches))
        print('episode: ',episode,'average accuracy:',accuracy)




def main():
    AstarRNN = DAN()
    pre_train(AstarRNN, 50)
#    Time_diff(AstarRNN)
#    ST(AstarRNN)
#    Q_learning_train_two_task(AstarRNN)
#    margin_loss_st_train(AstarRNN)
if __name__ == '__main__':
    main()

