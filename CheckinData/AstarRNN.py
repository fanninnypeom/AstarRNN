import tensorflow as tf
import numpy as np
import random
import pickle
import copy
from collections import deque
import os
import math

train_dir = "/data/wuning/LSBN Data/GowallaTrainData"       #/data/wuning/mobile trajectory/Q_learning_trainSet
test_dir = "/data/wuning/LSBN Data/GowallaTestData"
adj_dir = "/data/wuning/LSBN Data/adjMat"
loc2latlon_dir = "/data/wuning/LSBN Data/loc2latlon_new"

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
block_num = 17187
lstm_size = 1024
num_layers = 1
TRAIN_BATCH_SIZE = 100 #训练输入的batch 大小
INFERENCE_BATCH_SIZE = 1 #推断的时候输入的batch 大小
PRE_EPISODE = 600
NEG_SAMPLES = 9
NEXT_ACTION_NUM = 3

class DAN():
  def __init__(self):
    # init experience replay
    self.train_batches = []
    self.test_batches = []
    self.adj = []
    # init some parameters
    self.token2cor = {}
    self.sigma = 0.001   #高斯核的系数

    self.gradients= None

    self.load_data()
    self.create_policy_network()
#    self.create_heuristics_network()
#    self.create_neigh_constrained_network()
    self.all_saver = tf.train.Saver(max_to_keep=10)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self.session = tf.InteractiveSession(config = config)

  
    self.session.run(tf.global_variables_initializer())

#    self.all_saver = tf.train.import_meta_graph("/data/wuning/AstarRNN/pretrain_test_policity_neural_network_epoch0.ckpt.meta")
# "/data/wuning/AstarRNN/pretrain_test_policity_neural_network_epoch0.ckpt")
    # Init session

    all_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    variables_to_restore = [v for v in all_variables if v.name.split('/')[0]=='policy_network']
#    print("variables:", variables_to_restore)
    self.policy_saver = tf.train.Saver(variables_to_restore, max_to_keep=10)
#    self.all_saver = tf.train.Saver(max_to_keep=10)

  def build_lstm(self, batch_size):
    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
  #BasicLSTMCell    GRUCell
  # 添加dropout

    drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=self.keep_prob)

  # 堆叠
    cell = tf.nn.rnn_cell.MultiRNNCell([drop for _ in range(num_layers)])
    initial_state = cell.zero_state(batch_size, tf.float32)

    return cell, initial_state

  def create_policy_network(self):  
    with tf.variable_scope("policy_network"):
      self.keep_prob = tf.placeholder(tf.float32)
      self.p_output_ = tf.placeholder(tf.int64, [batch_size, input_steps], name = "p_output") 
      self.p_known_ = tf.placeholder(tf.int64, shape=(batch_size, input_steps), name='p_known')
      p_known_embedding = tf.contrib.layers.embed_sequence(self.p_known_, block_num, lstm_size, scope = "location_embedding")
      print("--------------", p_known_embedding)
      self.p_destination_ = tf.placeholder(tf.int64, shape=(batch_size), name='p_destination')
      self.p_destination_embedding = tf.contrib.layers.embed_sequence(self.p_destination_, block_num, lstm_size, scope = "location_embedding", reuse = True)
      cell, initial_state = self.build_lstm(tf.shape(self.p_known_)[0])
      self.outputs, self.final_state = tf.nn.dynamic_rnn(cell, tf.transpose(p_known_embedding, [1, 0, 2]), initial_state = initial_state, dtype=tf.float32, time_major=True)

      with tf.variable_scope('policy_output'):
        w_p = tf.Variable(tf.truncated_normal([lstm_size, block_num], stddev=0.1))
        b_p = tf.Variable(tf.zeros(block_num))

      print(self.p_destination_embedding, self.final_state[0][1])
      self.policy = tf.matmul(tf.reshape(tf.add(tf.expand_dims(self.p_destination_embedding, 0), self.outputs), [-1, lstm_size]), w_p) + b_p
#      self.policy = tf.matmul(tf.reshape(self.outputs, [-1, lstm_size]), w_p) + b_p
      print(self.policy)
      self.policy = tf.transpose(tf.reshape(self.policy, [tf.shape(self.p_known_)[1], tf.shape(self.p_known_)[0], block_num]), [1, 0, 2])
      self.action = tf.argmax(self.policy, axis=2)
      self.policy_prob = tf.nn.softmax(self.policy)
      action_one_hot = tf.one_hot(self.p_output_, block_num)
      self.policy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.policy , labels=action_one_hot)
      self.policy_loss = tf.reduce_mean(self.policy_loss)
      self.policy_optimizer = tf.train.AdamOptimizer(0.001).minimize(self.policy_loss) 


  def create_neigh_constrained_network(self):
    with tf.variable_scope("policy_network"):
      self.keep_prob = tf.placeholder(tf.float32)
      self.half_batch = tf.placeholder(tf.int32)
      self.p_known_ = tf.placeholder(tf.int64, shape=(batch_size, input_steps), name='p_known')
      p_known_embedding = tf.contrib.layers.embed_sequence(self.p_known_, block_num, lstm_size, scope = "location_embedding")
      print("--------------", p_known_embedding)
      self.p_destination_ = tf.placeholder(tf.int64, shape=(batch_size), name='p_destination')
      self.p_destination_embedding = tf.contrib.layers.embed_sequence(self.p_destination_, block_num, lstm_size, scope = "location_embedding", reuse = True)
      cell, initial_state = self.build_lstm(tf.shape(self.p_known_)[0])
      self.outputs, self.final_state = tf.nn.dynamic_rnn(cell, tf.transpose(p_known_embedding, [1, 0, 2]), initial_state = initial_state, dtype=tf.float32, time_major=True)

      with tf.variable_scope('policy_output'):
        w_p = tf.Variable(tf.truncated_normal([lstm_size, 1], stddev=0.1))
        b_p = tf.Variable(tf.zeros(1))

#      with tf.variable_scope('heuristics_output'):
#        w_h = tf.Variable(tf.truncated_normal([lstm_size, 1], stddev=0.1))
#        b_h = tf.Variable(tf.zeros(1))

      self.policy_heuristics = tf.matmul(tf.add(self.p_destination_embedding, self.outputs[-1, :, :]), w_p) + b_p

#      self.heuristics = tf.matmul(tf.add(self.p_destination_embedding, self.outputs[-1, :, :]), w_h) + b_h

#      self.time = tf.nn.relu(tf.matmul(output_state, w_t) + b_t)

#      self.time_input = tf.placeholder(tf.float32, [batch_size], name = "time_input")
      print(self.policy_heuristics)
      margin = 1.0 - tf.slice(self.policy_heuristics, [0, 0], [self.half_batch, 1]) + tf.slice(self.policy_heuristics, [self.half_batch, 0], [self.half_batch, 1])

      condition = tf.less(margin, 0.)

########  supervised learning
#      self.heuristics_input = tf.placeholder(tf.float32, shape=(batch_size), name='heuristics_input')
#      self.f_heuristics_cost = tf.reduce_mean(tf.square(self.heuristics_input - self.heuristics))
########

########### margin loss
      self.heuristics_cost = tf.reduce_mean(tf.where(condition, tf.zeros_like(margin), margin))
###########

      self.policy_heuristics_optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.heuristics_cost)
#      self.heuristics_optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.f_heuristics_cost)


  def create_heuristics_network(self):
    with tf.variable_scope("value_network"):  
      with tf.variable_scope('value_output'):
        w_v1 = tf.Variable(tf.truncated_normal([lstm_size, 2*lstm_size], stddev=0.1))
        b_v1 = tf.Variable(tf.zeros(2*lstm_size))
        w_v2 = tf.Variable(tf.truncated_normal([2*lstm_size, 1], stddev=0.1))
        b_v2 = tf.Variable(tf.zeros(1))
      print("outputs:", self.outputs)
#self.outputs[-1, :, :]
      self.value_layer_1 =tf.nn.relu(tf.matmul(tf.add(self.p_destination_embedding, self.outputs[-1, :, :]), w_v1) + b_v1)
      self.heuristics = tf.matmul(self.value_layer_1, w_v2) + b_v2

#####  margin loss
#      margin = 1.0 - tf.slice(self.heuristics, [0, 0], [TRAIN_BATCH_SIZE, 1]) + tf.slice(self.heuristics, [TRAIN_BATCH_SIZE, 0], [TRAIN_BATCH_SIZE, 1])
#      condition = tf.less(margin, 0.)
#      self.heuristics_cost = tf.reduce_mean(tf.where(condition, tf.zeros_like(margin), margin))
#####

#####  supervised loss
      self.heuristics_input = tf.placeholder(tf.float32, shape=(batch_size), name='heuristics_input')
      self.heuristics_cost = tf.reduce_mean(tf.square(self.heuristics_input - self.heuristics))
#####


#      self.gradients = tf.gradients(self.heuristics_cost, [output_state])

      self.value_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      "value_network")

      self.optimizer = tf.train.AdamOptimizer(0.00001).minimize(self.heuristics_cost, var_list=self.value_variables)


  def old_create_heuristics_network(self):
    with tf.variable_scope("value_network"):
      self.known_ = tf.placeholder(tf.int64, shape=(batch_size, input_steps), name='known')
      known_embedding = tf.contrib.layers.embed_sequence(self.known_, block_num, lstm_size, scope = "value_location_embedding")
      self.waiting_ = tf.placeholder(tf.int64, shape=(batch_size), name='waiting')
      waiting_embedding = tf.contrib.layers.embed_sequence(self.waiting_, block_num, lstm_size, scope = "value_location_embedding", reuse = True)
      self.destination_ = tf.placeholder(tf.int64, shape=(batch_size), name='destination')
      destination_embedding = tf.contrib.layers.embed_sequence(self.destination_, block_num, lstm_size, scope = "value_location_embedding", reuse = True)
    # network weights

      fw_cell, fw_initial_state = self.build_lstm(tf.shape(self.known_)[0])
      bw_cell, bw_initial_state = self.build_lstm(tf.shape(self.known_)[0])
#    print("-------", tf.concat([known_embedding, tf.expand_dims(waiting_embedding, 1)], 1))
      outputs, state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, tf.transpose(tf.concat([known_embedding, tf.expand_dims(waiting_embedding, 1)], 1), [1, 0, 2]), initial_state_fw=fw_initial_state, initial_state_bw=bw_initial_state, dtype=tf.float32, time_major=True)

      initial_state = tf.add(state[0], state[1])
      unstack_state = tf.unstack(initial_state, axis=0)
      tuple_state = tuple([tf.contrib.rnn.LSTMStateTuple(unstack_state[idx][0], unstack_state[idx][1]) for idx in range(num_layers)])

      hidden_states = tf.add(outputs[0], outputs[1])

      distant_embedding = waiting_embedding + destination_embedding
 #   W1 = self.weight_variable([self.state_dim,20])
 #   b1 = self.bias_variable([20])

      with tf.variable_scope('layer1'):
        local_w1 = tf.Variable(tf.truncated_normal([lstm_size, 2*lstm_size], stddev=0.1))
        local_b1 = tf.Variable(tf.zeros(2*lstm_size))

      local_1_layer = tf.nn.relu(tf.matmul(distant_embedding, local_w1) + local_b1)

      with tf.variable_scope('layer2'):
        local_w2 = tf.Variable(tf.truncated_normal([2*lstm_size, lstm_size], stddev=0.1))
        local_b2 = tf.Variable(tf.zeros(lstm_size))

      local_2_layer = tf.nn.relu(tf.matmul(local_1_layer, local_w2) + local_b2)
  
#      output_state = tf.reduce_mean(hidden_states, 0) + local_2_layer

#      print(hidden_states, initial_state.shape, local_2_layer.shape)
      output_state = hidden_states[-1, :, :] + local_2_layer

      with tf.variable_scope('output'):
        w_h = tf.Variable(tf.truncated_normal([lstm_size, 1], stddev=0.1))
        b_h = tf.Variable(tf.zeros(1))
        w_t = tf.Variable(tf.truncated_normal([lstm_size, 1], stddev=0.1))
        b_t = tf.Variable(tf.zeros(1))
      

#      self.heuristics = tf.nn.sigmoid(tf.matmul(output_state, w_h) + b_h)
      self.heuristics = tf.matmul(output_state, w_h) + b_h

      self.time = tf.nn.relu(tf.matmul(output_state, w_t) + b_t)

      self.heuristics_input = tf.placeholder(tf.float32, [batch_size], name = "heuristics_input")

      self.time_input = tf.placeholder(tf.float32, [batch_size], name = "time_input")

#      half_batch_size = tf.div(self.heuristics.shape[0], 2)
      print(tf.slice(self.heuristics, [0, 0], [TRAIN_BATCH_SIZE, 1]).shape, tf.slice(self.heuristics, [TRAIN_BATCH_SIZE, 0], [TRAIN_BATCH_SIZE, 1]).shape)
      margin = 1.0 - tf.slice(self.heuristics, [0, 0], [TRAIN_BATCH_SIZE, 1]) + tf.slice(self.heuristics, [TRAIN_BATCH_SIZE, 0], [TRAIN_BATCH_SIZE, 1])

      condition = tf.less(margin, 0.)

########  supervised learning
      self.heuristics_input = tf.placeholder(tf.float32, shape=(batch_size), name='heuristics_input')
      self.heuristics_cost = tf.reduce_mean(tf.square(self.heuristics_input - self.heuristics))
########

########### margin loss
#      self.heuristics_cost = tf.reduce_mean(tf.where(condition, tf.zeros_like(margin), margin))
###########

#      self.heuristics_cost = tf.reduce_mean(tf.square(self.heuristics_input - self.heuristics))

      self.gradients = tf.gradients(self.heuristics_cost, [output_state])
#      self.heuristics_cost = tf.reduce_mean(tf.square(self.heuristics_input - self.heuristics)) 


  #  y_reshaped = tf.reshape(y_one_hot, logits.get_shape())

    # Softmax cross entropy loss
      self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.heuristics_cost)
 

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

  def process_data(self):   #切勿调用此函数  set会打乱所有的token
    File = open("/data/wuning/mobile trajectory/mobileToken2cor", "rb")
    string2cor = pickle.load(File)
    File = open("/data/wuning/mobile trajectory/mobileTras", "rb")
    data = pickle.load(File)
    locations = set()
    for item in data:
      for tra in item:
        for rec in tra:
          locations.add(rec)      # 这里每个rec可能只有 token  也可能是token+timestamp  rec[0]
    vocabulary = {}
    count = 0
    for key in locations:
      try:
        self.token2cor[count] = string2cor[key] 
      except KeyError:
        continue
      vocabulary[key] = count
      count += 1 
    batches = []
    for batch in data:
      bb = []
      for tra in np.array(batch)[:, :]: #np.array(batch)[:, :, 0]
        aa = []
        flag = True
        for rec in tra:
          if rec not in vocabulary.keys():
            flag = False
            continue
          aa.append(vocabulary[rec])
        if flag:
          bb.append(aa)
      batches.append(bb)  
#    print(len(batches), len(batches[0]), len(batches[0][0])) 
#    print(np.array(batches).shape, "00000")
    self.train_batches = batches[0:20000][:][:]
    self.test_batches = batches[20000:25000][:][:]
    pickle.dump(self.token2cor, open("/data/wuning/mobile trajectory/Q_learning_token2cor", "wb"), -1)
    pickle.dump(self.train_batches, open("/data/wuning/mobile trajectory/Q_learning_trainSet", "wb"), -1)
    pickle.dump(self.test_batches, open("/data/wuning/mobile trajectory/Q_learning_testSet", "wb"), -1)
    print("process data finish")
  def load_data(self):
    File = open(train_dir, "rb")   #trainSet
    self.train_batches = pickle.load(File)
    File = open(test_dir, "rb")    #testSet
    self.test_batches = pickle.load(File)
    File = open(adj_dir, "rb")
    self.adj = pickle.load(File)
    File = open(loc2latlon_dir, "rb")
    self.loc2latlon = pickle.load(File)
    File = open("/data/wuning/mobile trajectory/Q_learning_token2cor", "rb")
    self.token2cor = pickle.load(File)

    print("batch_size:", len(self.train_batches[0]))
  def load_beijing_data(self):
    File = open("/data/wuning/AstarBeijing/beijingPolicyTrainSet", "rb")   #trainSet
    self.train_batches = pickle.load(File)
    File = open("/data/wuning/AstarBeijing/beijingTestSet", "rb")    #testSet
    self.test_batches = pickle.load(File)


  def policy_train(self, PRE_EPISODE):
#    self.all_saver.restore(self.session, "/data/wuning/AstarRNN/pretrain_test_policity_neural_network_epoch0.ckpt")

    pre_test_batches = []
    pre_batches = []
    for train_batch in self.train_batches:
      if len(train_batch) > 0:
        pre_batches.append([np.array(train_batch)[:,:-1], np.array(train_batch)[:,1:], np.array(train_batch)[:,-1]]) 

    for test_batch in self.test_batches:
      if len(test_batch) > 0:
        pre_test_batches.append([np.array(test_batch)[:,:-1], np.array(test_batch)[:,1:], np.array(test_batch)[:,-1]]) 
    print("len pre batch:", len(pre_batches), len(pre_test_batches))
    right_num = 0
    sum_num = 0
    for episode in range(PRE_EPISODE):
      counter = 0
      for batch in pre_batches:
#        print(np.array(batch)[0, 0])
#        print("-----")
#        print(np.array(batch)[:, 1])
#        print(np.array(np.array(batch)[:, 0].tolist()).shape, np.array(np.array(batch)[:, 1].tolist()).shape)
        self.policy_optimizer.run(feed_dict={
          self.p_known_:np.array(batch[0]),
          self.p_output_:np.array(batch[1]),
          self.p_destination_:np.array(batch[2]),
          self.keep_prob:0.2
        })
        eval_policy_loss = self.policy_loss.eval(feed_dict={
          self.p_known_:np.array(batch[0]),
          self.p_output_:np.array(batch[1]),
          self.p_destination_:np.array(batch[2]),
          self.keep_prob:0.2
        })
        
        if counter % 1000 == 0:
          print("epoch:{}...".format(episode),
                "batch:{}...".format(counter),
                "loss:{:.4f}...".format(eval_policy_loss))
        counter += 1
      average_loss = 0
      for test_batch in pre_test_batches:
        pred_policy = self.action.eval(feed_dict={
          self.p_known_:test_batch[0],
          self.p_destination_:test_batch[2],
          self.keep_prob:1.0
        })
        for tra, pred_tra in zip(test_batch[1], pred_policy):
          for item_1, item_2 in zip(tra, pred_tra):
            if item_1 == item_2:
              right_num += 1
            sum_num += 1
#        test_policy_loss = right_num / float(sum_num)
#        average_loss += test_policy_loss
      print("test_loss:", right_num / float(sum_num))
      self.all_saver.save(self.session, "/data/wuning/AstarRNN/gowalla_Q_learning_pre_train_neural_network_epoch{}.ckpt".format(episode))

  def reconstruct_path(self, cameFrom, current):
    total_path = [current]
    while current in cameFrom:
      current = cameFrom[current]
      total_path.append(current)
    return total_path    

  def insert(self, item, the_list, f_score):  #插入 保持从大到小的顺序
    if(len(the_list) == 0):
      return [item]
    for i in range(len(the_list)):
      if(f_score[the_list[i]] < f_score[item]):
        the_list.insert(i, item)
        break
      if i == len(the_list) - 1:
        the_list.append(item)
    return the_list

  def move(self, item, the_list, f_score):
    for it in the_list:
      if(f_score[it] == f_score[item]):
        the_list.remove(it)
        break
    return self.insert(item, the_list, f_score)


  def greedyTest(self):
    counters = 0
    all_len = 0
    for batch in self.test_batches:
      result = []
      for start, unknown, end in zip(np.array(batch)[:, 0], np.array(batch)[:,1:-1], np.array(batch)[:, -1]):
        path = [start]
        for i in range(len(unknown)):
#          print("path:", path)
          policy = self.action.eval(
              feed_dict={
                self.p_known_:np.array(path)[np.newaxis, :],
                self.p_destination_:[end],
                self.keep_prob:1.0
            })[0][-1]
#          policy = np.argmax(policy_value, axis=2)[:,-1]
          path.append(policy)
        path.append(end)
        result.append(path)
      for infer, real in zip(result, batch):
        print("infer:", infer)
        print("real:", real)
        for item in infer[1:-1]:
          if item in real[1:-1]:
            counters += 1
        all_len += len(real) - 2
      print(float(counters)/all_len)

  def beamSearch(self):#batch size 为50
    beam_width = 10
    all_len = 0
    counters = 0
    for batch in self.test_batches:
      known_seq = [np.array(batch)[:, 0][:, np.newaxis]]
      known_score = []
      for i in range(0, len(batch[0]) - 2):
        if i == 0:
          policy_value = self.policy_prob.eval(
              feed_dict={
                self.p_known_:known_seq[0],
                self.p_destination_:np.array(batch)[: , -1],
                self.keep_prob:1.0
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
#        all_policy = []
        policy_known_score = []
        for j in range(0, len(known_seq)):
          policy_value = self.policy_prob.eval(
              feed_dict={
                self.p_known_:known_seq[j],
                self.p_destination_:np.array(batch)[: , -1],
                self.keep_prob:1.0
          })
          immedia = np.array(known_score[j])[:, np.newaxis] + np.array(policy_value)[:, -1, :]
#          print(immedia.shape)
          if j == 0:
#            all_policy = np.array(policy_value)[:, -1, :]
            policy_known_score = immedia
          else:
#            all_policy = np.concatenate((all_policy, np.array(policy_value)[:, -1, :]), axis=1)
            policy_known_score = np.concatenate((policy_known_score, immedia), axis=1)
#          policy_value = np.array(policy_value)
        policy = np.argsort(-policy_known_score, axis=1)[:,:beam_width]
        raw_policy = policy
        index = policy // 17187
        policy = policy % 17187
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
#        print("infer:", len(infer))
#        print("real:", len(real))
        for item in infer[1:]:
          if item in real[1:-1]:
            counters += 1
        all_len += len(real) - 2
      print(float(counters)/all_len)
  
#          temp_known = []
#          for i in range(0, beam_width):
#            item_known = np.concatenate((known_seq[0], policy[:, i][:, np.newaxis]), axis = 1)
#            temp_known.append(item_known)
#          all_policy.extend(policy_value)

               

  def AstarTest(self):    
    results = []
    counters = 0
    all_len = 0
    for batch in self.test_batches:
      result = []
      count = 0
      
      for start, unknown, end in zip(np.array(batch)[:, 0], np.array(batch)[:,1:-1], np.array(batch)[:, -1]):
        closedSet = []
        openSet = [start]
        cameFrom = {}
        pathFounded = {start: [start]}
        fScore = {}
        fScore[start] = 0
        waitingTra = 0
        bestScore = -10000000000
        bestTra = []
        searchCount = 0
####Test Code   compare the score of different trajectory
#        temp_batch = []
#        for i in range(0, 100):
#          temp = [start]
#          temp.extend(unknown)
#          if not i == 0:
#            temp[2] = i
#          temp_batch.append(temp)
      
#        f_score = self.heuristics.eval(
#                feed_dict={
#                   self.p_known_:np.array(temp_batch),
#                   self.p_destination_:[end for i in range(100)],
#                  }
#              )
#        for tra_input, tra_score in zip(temp_batch, f_score):
#          print(tra_input, tra_score)
#        continue
####Test Code 

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
          if len(pathFounded[current]) == len(unknown) + 1: 
            if fScore[current] > bestScore:
              bestScore = fScore[current]
              bestTra = pathFounded[current]
              bestTra.append(end)
            continue

          policy_value = self.policy_prob.eval(
              feed_dict={
                self.p_known_:np.array(pathFounded[current])[np.newaxis, :],
                self.p_destination_:[end],
                self.keep_prob:1.0
          })
#          test_policy_loss = self.policy_loss.eval(feed_dict={
#            self.p_known_:np.array(batch)[:, :-1],
#            self.p_destination_:np.array(batch)[:, -1],
#            self.p_output_:np.array(batch)[:, 1:],
#            self.keep_prob:1.0
#          })

#          predict_output = self.policy.eval(
#              feed_dict={
#                self.p_known_:np.array(batch)[:, :-1],
#                self.p_destination_:np.array(batch)[:, -1],
#                self.keep_prob:1.0
#               })


#          print("predict:", np.argsort(predict_output, axis=2)[0,:,:NEXT_ACTION_NUM])
#          print("predict:", np.argsort(predict_output, axis=2)[1,:,:NEXT_ACTION_NUM])
#          print("predict:", np.argsort(predict_output, axis=2)[2,:,:NEXT_ACTION_NUM])
#          print("predict:", np.argsort(predict_output, axis=2)[3,:,:NEXT_ACTION_NUM])
#          print("predict:", np.argsort(predict_output, axis=2)[4,:,:NEXT_ACTION_NUM])
#          print("predict:", np.argsort(predict_output, axis=2)[5,:,:NEXT_ACTION_NUM])

#          print("p_output:", np.array(batch)[:, 1:])
          policy_value = np.array(policy_value)

#          policy_value = np.exp(policy_value)/np.sum(np.exp(policy_value), axis=2)[:, :, np.newaxis]

          policy = np.argsort(-policy_value, axis=2)[:,-1,:2]
#          print("policy_loss:", test_policy_loss)
#          print("policy_value:", policy_value)
#          print("start:", start)
#          print("end:", end)
#          print("path:", pathFounded[current])
#          print("policy:", policy)
#          print("unknown:", unknown)
#         print("-------------")
          policy_list = policy[0].tolist()

#          if policy_list[0] in pathFounded[current]:
#            policy_list = policy_list[1:]
#          else:
#            policy_list = policy_list[:-1]

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

          p_known_batch = []
          for action in policy_list:
            p_known_batch.append(pathFounded[current])          

#          f_scores = self.heuristics.eval(
#              feed_dict={
#                self.p_known_:np.concatenate((p_known_batch, np.array(policy_list)[:, np.newaxis]), 1),
#                self.p_destination_:[end for i in range(len(policy_list))]
#                }
#            )

#          print("f_scores:", f_scores)

          for waiting_count in range(len(policy_list)):
            waiting = policy_list[waiting_count]  
            if (waiting in closedSet):
              continue
            f_score = np.array(policy_value)[-1, -1, waiting] * fScore[current]

            temp = copy.deepcopy(pathFounded[current])
            temp.append(waiting) 

#            f_score =  self.heuristics.eval(
#                feed_dict={
#                  self.known_:np.array(pathFounded[current])[np.newaxis, :],
#                  self.waiting_:[waiting],
#                  self.destination_:[end]
#---------------------------
#                   self.p_known_:np.array(temp)[np.newaxis, :],
#                   self.p_destination_:[end],
#                  }
#              )

#            f_score = random.uniform(0, 10)
#            f_score = f_scores[waiting_count]
#            f_score = f_score+ fScore[current]
#            print("f_score:", f_score)
            if (waiting in fScore) and (f_score < fScore[waiting]):
              continue
            fScore[waiting] = f_score
            if waiting not in openSet:
#              print("insert!")
              openSet = self.insert(waiting, openSet, fScore)
            else:
              openSet = self.move(waiting, openSet,fScore)
            pathFounded[waiting] =  temp
#          print(searchCount)
          if(searchCount >= 300):
            openSet = []
#            print(temp)
        count += 1
        print(count)
#        print("----------------------------------------------")
#        print("count:", count)
#        if(len(openSet) == 0 and len(bestTra) == 0):
        if len(openSet) == 0:
          result.append(bestTra)
#        print("count:", count)
      results.append(result)
      for infer, real in zip(result, batch):     
        print("infer:", infer)#[self.loc2latlon[item] for item in infer])
        print("real:", real)#[self.loc2latlon[item] for item in real])
        for item in infer[1:-1]:
          if item in real[1:-1]:
            counters += 1 
        all_len += len(real) - 2
      print(float(counters)/all_len) 
    return float(counters)/all_len

  def generate_Q_learning_samples(self):  #将没有经纬度坐标的轨迹都删除掉了
    Q_learning_train_data = []
    Q_learning_test_data = []
    for batch in self.train_batches:
      Q_batch = []
      for tra in batch:
        flag = True
        for item in tra:
          if item not in self.token2cor.keys():
            flag = False
        if flag:
          Q_batch.append(tra)
      Q_learning_train_data.append(Q_batch) 
    print("finish train data.")
    for batch in self.test_batches:
      Q_batch = []
      for tra in batch:
        flag = True
        for item in tra:
          if item not in self.token2cor.keys():
            flag = False
        if flag:
          Q_batch.append(tra)
      Q_learning_test_data.append(Q_batch) 
    print("Q_learning_data:", len(Q_learning_train_data), len(Q_learning_test_data)) 
    pickle.dump(Q_learning_test_data, open("/data/wuning/mobile trajectory/Q_learning_test_data", "wb"))
    pickle.dump(Q_learning_train_data, open("/data/wuning/mobile trajectory/Q_learning_train_data", "wb"))

  def generate_supervised_samples(self):
    heuristics_batches = []
    counter = 0
    for batch in self.train_batches:
      for i in range(1, len(batch[0]) - 2):
        wait_actions = np.random.randint(1, 26261, size=[len(batch), 9])
        wait_actions = np.concatenate((wait_actions, np.array(batch)[:, i][:, np.newaxis]), 1)
        y_batch = np.zeros(wait_actions.shape)
        y_batch[:,-1] = 1 

        for j in range(10):
          heuristics_batches.append([np.array(batch)[:, 0:i], wait_actions[:, j], np.array(batch)[:, -1], y_batch[:, j]])
      if counter % 100 == 0:
        print(len(heuristics_batches))
        print("batches generated:{}...".format(counter))
      if counter % 500 == 0:
        pickle.dump(heuristics_batches, open("/data/wuning/mobile trajectory/Q_learning_serpervised_heuristicsTrainSet"+str(counter), "wb"))
        heuristics_batches = []
      counter += 1

  def generate_heuristics_samples(self):
    heuristics_batches = []
    counter = 1
    for batch in self.train_batches:
      for i in range(1, len(batch[0]) - 2):

        eval_policy = self.policy.eval(
          feed_dict={
            self.p_known_:np.array(batch)[:, :i],
            self.p_destination_:np.array(batch)[:, -1]
          })
#          print("eval_policy_shape", np.array(eval_policy).shape)
        action_value = self.action.eval(
          feed_dict={
            self.p_known_:np.array(batch)[:, :i],
            self.p_destination_:np.array(batch)[:, -1]
        })

        eval_policy_loss = self.policy_loss.eval(feed_dict={
          self.p_known_:np.array(batch)[:, :i],
          self.p_destination_:np.array(batch)[:, -1],
          self.p_output_:np.array(batch)[:, 1:i+1]
        })
#          print(eval_policy.dtype)
        wait_actions = np.argsort(np.array(-eval_policy)[:, -1, :], axis=1)[:,:2]
#          test_action = np.argmax(eval_policy, axis=1)
#          print("eval_policy_loss:", eval_policy_loss)
#          print("action_value:", action_value)
 #         print("test_action:", test_action)
#          print("wait_actions:", wait_actions)
##          print("---------------")
#          print(np.array(batch)[:, i])
#          print(np.array(batch)[:, i-1])
#          print(np.array(batch)[:, i+1])
          # batch_size * (NEG_SAMPLES + 1) * 2      -    batch_size  * 2

        y_batch = np.zeros(wait_actions.shape)
        for k in range(y_batch.shape[0]):
          if(not np.array(batch)[k, i] == wait_actions[k, 0]):
            wait_actions[k, 1] = wait_actions[k, 0]
            wait_actions[k, 0] = np.array(batch)[k, i]
          y_batch[k, 0] = 1
          try:
            item_1 = wait_actions[k, 1]
            item_0 = wait_actions[k, 0]    
            rewards = np.array([float(self.token2cor[item_1][0]), float(self.token2cor[item_1][1])]) - np.array([float(self.token2cor[item_0][0]), float(self.token2cor[item_0][1])])
            y_batch[k, 1] = np.exp(np.sum(- rewards**2) / self.sigma)
          except Exception:
            pass

        for j in range(2):
          heuristics_batches.append([np.array(batch)[:, 0:i], wait_actions[:, j], np.array(batch)[:, -1], y_batch[:, j]])
      if counter % 100 == 0:
        print(len(heuristics_batches))
        print("batches generated:{}...".format(counter))
      if counter % 500 == 0:
        pickle.dump(heuristics_batches, open("/data/wuning/mobile trajectory/Q_learning_reward_heuristicsTrainSet"+str(counter), "wb"))
        heuristics_batches = []
      counter += 1

  def Q_learning_train_two_task(self):
    heuristics_batches = []
    counter = 0
    heu_ave = []
    for episode in range(EPISODE):
      for i in range(0, 12500, 500):
        Q_learning_batches = pickle.load(open("/data/wuning/AstarBeijing/beijing_Q_learning_serpervised_heuristicsTrainSet"+str(i), "rb"))
#mobile trajectory/Q_learning_serpervised_heuristicsTrainSet
        for batch in Q_learning_batches:
          if len(batch) == 0:
            continue
#######test
#          feed_data = {
#            self.p_known_:np.concatenate((np.array(batch[0]), np.array(batch[1])[:, np.newaxis]), 1),
#            self.p_destination_:batch[2],
#          }
#          heuristics = self.heuristics.eval(feed_dict=feed_data)
#          print("heuristics:", np.mean(heuristics))
#          print(heuristics)
#          print("y:", batch[3])
#          continue
#######

          policy_feed_data = {
            self.p_known_:np.concatenate((np.array(batch[0]), np.array(batch[1])[:, np.newaxis]), 1),
            self.p_destination_:batch[2]
          }
          eval_next_policy = self.policy.eval(feed_dict=policy_feed_data)
          wait_next_actions = np.argsort(eval_next_policy, axis=1)[:, -1, :NEXT_ACTION_NUM]
          heuristics_batch = []  #下一个状态的Q值
          for k in range(NEXT_ACTION_NUM):
            heuristics_batch.append(self.heuristics.eval(
              feed_dict={
                self.p_known_:#np.concatenate((wait_next_actions[:, k][:, np.newaxis], wait_next_actions[:, k][:, np.newaxis], wait_next_actions[:, k][:, np.newaxis]), 1),
                np.concatenate((np.array(batch[0]), np.array(batch[1])[:, np.newaxis], wait_next_actions[:, k][:, np.newaxis]), 1),
                self.p_destination_:batch[2]
                }
              )
            )
          heuristics_batch = np.array(heuristics_batch)
          print("heuristics_batch:", heuristics_batch[2, :], len(heuristics_batch))
          heu_ave.append(np.mean(heuristics_batch))
          print("heuristics_ave:", np.mean(heu_ave))
          
          heuristics_batch = np.max(heuristics_batch, axis = 0)
          batch[3] += GAMMA * heuristics_batch[:, 0]
          heuristics_batches.append([np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), batch[3]])
        for batch in heuristics_batches:
          feed_data = {
            self.p_known_:np.concatenate((np.array(batch[0]), np.array(batch[1])[:, np.newaxis]), 1),
            self.p_destination_:batch[2],
            self.heuristics_input:batch[3]
          }
          self.optimizer.run(feed_dict=feed_data)
          heuristics_cost = self.heuristics_cost.eval(feed_dict=feed_data)
        heuristics = self.heuristics.eval(feed_dict=feed_data)
        print("heuristics:", heuristics)
        self.all_saver.save(self.session, "/data/wuning/AstarRNN/train_heuristics_TD1_two_task_step{}_epoch{}.ckpt".format(i, episode))
        heuristics_batches = []
        print("loss:", heuristics_cost)
    

  def Q_learning_train(self):
    heuristics_batches = []
    counter = 0
    for episode in range(EPISODE):
      for i in range(500, 12500, 500):
        Q_learning_batches = pickle.load(open("/data/wuning/mobile trajectory/Q_learning_heuristicsTrainSet"+str(i), "rb"))
        for batch in Q_learning_batches:
#######test
#          feed_data = {
#            self.known_:batch[0],
#            self.waiting_:batch[1],
#            self.destination_:batch[2],
#            self.heuristics_input:batch[3]
#          }
#          heuristics = self.heuristics.eval(feed_dict=feed_data)
#          print("heuristics:", np.mean(heuristics))
#          print(heuristics)
#          print("y:", batch[3])
#          continue
#######
          policy_feed_data = {
            self.p_known_:np.concatenate((np.array(batch[0]), np.array(batch[1])[:, np.newaxis]), 1),
            self.p_destination_:batch[2]
          }
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


#      for batch in self.train_batches:
#        for i in range(1, len(batch[0]) - 2):
#          eval_policy = self.policy.eval(
#            feed_dict={
#              self.p_known_:np.array(batch)[:, :i],
#              self.p_destination_:np.array(batch)[:, -1]
#            })
#          action_value = self.action.eval(
#            feed_dict={
#              self.p_known_:np.array(batch)[:, :i],
#              self.p_destination_:np.array(batch)[:, -1]
#          })

#          eval_policy_loss = self.policy_loss.eval(feed_dict={
#            self.p_known_:np.array(batch)[:, :i],
#            self.p_destination_:np.array(batch)[:, -1],
#            self.p_output_:np.array(batch)[:, 1:i+1]
#          })
#          eval_policy[:, :, 24485:] = -10000
#          wait_actions = np.argsort(np.array(-eval_policy)[:, -1, :], axis=1)[:,:2]
#        print(self.token2cor)
###############  使用距离作为reward
#          try:    
#            reward_batch = np.array([[[float(self.token2cor[item][0]), float(self.token2cor[item][1])] for item in items] for items in wait_actions]) - np.array([[float(self.token2cor[item][0]), float(self.token2cor[item][1])] for item in np.array(batch)[:, i]])[:, np.newaxis,:]
#          except Exception:
#            break
#          y_batch = np.exp(np.sum(- reward_batch**2, axis=2) / self.sigma)   
#          print("y_batch:", y_batch)
###############



#        y_batch = np.zeros(wait_actions.shape)
#          if(i < len(batch[0]) - 3):
#            for j in range(wait_actions.shape[1]):
#              eval_next_policy = self.policy.eval(
#              feed_dict={
#                self.p_known_:np.concatenate((np.array(batch)[:, 0:i], wait_actions[:, j][:, np.newaxis]), 1),
#                self.p_destination_:np.array(batch)[:, -1]
#              })
#              wait_next_actions = np.argsort(-eval_next_policy, axis=1)[:, -1, :NEXT_ACTION_NUM]
#              heuristics_batch = []  #下一个状态的Q值
#              for k in range(NEXT_ACTION_NUM):
#              print("length:", len(self.token2cor))
#              print("dest shape:", np.array(batch)[:, -1].shape)
#              print("waiting shape:", wait_next_actions[:, :].shape)
#              print("known_ shape:", np.array(batch)[:, 0:i].shape)
#                heuristics_batch.append(self.heuristics.eval(
#                  feed_dict={
#                   self.known_:np.array(batch)[:, 0:i],
#                    self.waiting_:wait_next_actions[:, k],
#                    self.destination_:np.array(batch)[:, -1]
#                    }
#                  )
#                )
#            heuristics_batch = np.array(heuristics_batch)
#            heuristics_batch = np.max(heuristics_batch, axis = 0)
#            y_batch[:, j] += heuristics_batch[:, 0]
#            print("y_batch:", y_batch[:, j])
#            heuristics_batches.append([np.array(batch)[:, 0:i], wait_actions[:, j], np.array(batch)[:, -1], y_batch[:, j]])
#        if counter % 100 == 0:
#          print("batches generated:{}...".format(counter))
#        if counter % 500 == 0:
#          print("samples num:", len(heuristics_batches))
#          for batch in heuristics_batches:
#            feed_data = {
#              self.known_:batch[0],
#              self.waiting_:batch[1],
#              self.destination_:batch[2],
#              self.heuristics_input:batch[3]
#            }
#            self.optimizer.run(feed_dict=feed_data)
#            heuristics_cost = self.heuristics_cost.eval(feed_dict=feed_data)
#          heuristics = self.heuristics.eval(feed_dict=feed_data)
#          print("heuristics:", heuristics)  
#          self.all_saver.save(self.session, "/data/wuning/AstarRNN/train_heuristics_Q_neural_network_step{}_epoch{}.ckpt".format(counter, episode))
#          heuristics_batches = []
#          print("loss:", heuristics_cost)
#        counter += 1

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

#  def q_learning_two_task_train(self):

  def margin_loss_two_task_train(self): 
    policy_batches = []
    for batch in self.train_batches:
      if len(batch) > 0:
        policy_batches.append([np.array(batch)[:,:-1], np.array(batch)[:,1:], np.array(batch)[:,-1]])
    print("policy_batches length:", len(policy_batches))
    for episode in range(EPISODE):
      counter = 0
      for i in range(0, 2000, 500):
        heuristics_batches = pickle.load(open("/data/wuning/AstarBeijing/beijing_Q_learning_serpervised_heuristicsTrainSet"+str(i), "rb"))

#heuristicsTrainSet"+str(i), "rb"))
        for j in range(0, len(heuristics_batches), 10):
          print(heuristics_batches[j+9][3])
          batch = heuristics_batches[j]
          neg_batch = heuristics_batches[j + 1]
#          print("batch_size",len(heuristics_batches[0][0]))
#          neg_batch[0] = np.concatenate((neg_batch[0], neg_batch[1][:, np.newaxis]), 1)
#          neg_batch[0] = neg_batch[0].tolist()

#          batch[0] = np.concatenate((batch[0], batch[1][:, np.newaxis]), 1)
          batch[0] = batch[0].tolist()
          batch[0].extend(neg_batch[0])
          batch[1] = batch[1].tolist()
          batch[1].extend(neg_batch[1])
          batch[2] = batch[2].tolist()
          batch[2].extend(neg_batch[2])
          batch[3] = batch[3].tolist()
          batch[3].extend(neg_batch[3])
          print(np.array(batch[0]).shape, np.array(batch[1]).shape)
          feed_data = {
            self.p_known_:np.concatenate((batch[0], np.array(batch[1])[:, np.newaxis]), 1),
            self.p_destination_:batch[2]
          }
          self.optimizer.run(feed_dict=feed_data)
          heuristics_cost = self.heuristics_cost.eval(feed_dict=feed_data)
          heuristics_value = self.heuristics.eval(feed_dict=feed_data)
          policy_batch = policy_batches[counter % 20000]
          policy_feed_data = {
            self.p_known_:policy_batch[0],
            self.p_destination_:policy_batch[2],
            self.p_output_:policy_batch[1]
          }
          self.policy_optimizer.run(feed_dict=policy_feed_data)
          eval_policy_loss = self.policy_loss.eval(feed_dict=policy_feed_data)

#          heuristics_grad = self.gradients[0].eval(feed_dict=feed_data)
          if counter % 2001 == 0:
            print("y_batch:", batch[3])
            print("epoch:{}...".format(episode),
                  "batch:{}...".format(counter),
                  "heuristics{}...".format(heuristics_value),
#                  "grads{}...".format(heuristics_grad),
                  "loss:{:.4f}...".format(heuristics_cost),
                  "policy_loss:{:.4f}...".format(eval_policy_loss))
          counter += 1
      self.all_saver.save(self.session, "/data/wuning/AstarRNN/train_old_network_margin_loss_heuristics_epoch{}.ckpt".format(episode))
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
# ---------------------------------------------------------
EPISODE = 100 # Episode limitation
PRE_EPISODE = 300
TRAIN_BATCHES = 300 # Step limitation in an episode

def main():

  AstarRNN = DAN()  

  if(PRE_TRAIN):
    AstarRNN.policy_train(PRE_EPISODE)
  elif(RESTORE):
    AstarRNN.all_saver.restore(tf.get_default_session(), "/data/wuning/AstarRNN/gowalla_Q_learning_pre_train_neural_network_epoch34.ckpt")

#supervised_train_heuristics_neural_network_epoch1.ckpt")

#train_heuristics_TD1_step2000_epoch1.ckpt")

#/data/wuning/AstarRNN/train_heuristics_Q_neural_network_step500_epoch13.ckpt")
#/data/wuning/AstarRNN/Q_learning_neural_network_epoch112.ckpt


#train_old_network_margin_loss_heuristics_epoch61.ckpt
#    AstarRNN.policy_saver.restore(tf.get_default_session(), "/data/wuning/AstarRNN/train_heuristics_reward_neural_network_epoch99.ckpt")

#supervised_train_heuristics_neural_network_epoch99.ckpt  监督学习的模型
#/data/wuning/AstarRNN/train_heuristics_reward_neural_network_epoch99.ckpt   margin loss 的模型
#    AstarRNN.policy_saver.restore(tf.get_default_session(), "/data/wuning/AstarRNN/pretrain_test_policity_neural_network_epoch37.ckpt")
#"/data/wuning/AstarRNN/train_heuristics_reward_neural_network_epoch30.ckpt") 
  if(TEST):
#    AstarRNN.beamSearch()
    accuracy = AstarRNN.AstarTest()# greedy AstarTest()
    print('average accuracy:',accuracy)
  else:
#    AstarRNN.process_data()
#    AstarRNN.generate_Q_learning_samples()
#     AstarRNN.generate_heuristics_samples()
#    AstarRNN.supervised_train()
#    AstarRNN.generate_supervised_samples()
#    AstarRNN.Q_learning_train_two_task()  
#    AstarRNN.margin_loss_two_task_train()
    AstarRNN.policy_train(PRE_EPISODE) 
if __name__ == '__main__':
  main()
