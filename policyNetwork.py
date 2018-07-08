import tensorflow as tf
import numpy as np
import random
import pickle
import copy
from collections import deque

# Hyper Parameters for DAN
PRE_TRAIN = False
TEST = False
RESTORE = True
GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
batch_size = None # size of minibatch
input_steps = None
block_num = 26261
lstm_size = 512
num_layers = 2
TRAIN_BATCH_SIZE = 100 #训练输入的batch 大小
INFERENCE_BATCH_SIZE = 1 #推断的时候输入的batch 大小
PRE_EPISODE = 600
NEG_SAMPLES = 9
NEXT_ACTION_NUM = 10

class DAN():
  # DQN Agent
  def __init__(self):
    # init experience replay
    self.train_batches = []
    self.test_batches = []
    # init some parameters
    self.token2cor = {}
    self.sigma = 0.01   #高斯核的系数

    self.gradients= None

    self.load_data()
    self.create_policy_network()
    self.create_heuristics_network()
    self.all_saver = tf.train.Saver(max_to_keep=10)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self.session = tf.InteractiveSession(config = config)

  
    self.session.run(tf.global_variables_initializer())

#    self.all_saver = tf.train.import_meta_graph("/data/wuning/AstarRNN/pretrain_test_policity_neural_network_epoch0.ckpt.meta")
# "/data/wuning/AstarRNN/pretrain_test_policity_neural_network_epoch0.ckpt")
    # Init session


#    all_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
#    variables_to_restore = [v for v in all_variables if v.name.split('/')[0]=='policy_network']
#    print("variables:", variables_to_restore)
#    self.policy_saver = tf.train.Saver(variables_to_restore, max_to_keep=10)
#    self.all_saver = tf.train.Saver(max_to_keep=10)

  def build_lstm(self, batch_size):
    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)

  # 添加dropout

    drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=0.5)

  # 堆叠
    cell = tf.nn.rnn_cell.MultiRNNCell([drop for _ in range(num_layers)])
    initial_state = cell.zero_state(batch_size, tf.float32)

    return cell, initial_state

  def create_policy_network(self):  
    with tf.variable_scope("policy_network"):
      self.p_output_ = tf.placeholder(tf.int64, [batch_size, input_steps], name = "p_output") 
      self.p_known_ = tf.placeholder(tf.int64, shape=(batch_size, input_steps), name='p_known')
      p_known_embedding = tf.contrib.layers.embed_sequence(self.p_known_, block_num, lstm_size, scope = "location_embedding")
      print("--------------", p_known_embedding)
      self.p_destination_ = tf.placeholder(tf.int64, shape=(batch_size), name='p_destination')
      p_destination_embedding = tf.contrib.layers.embed_sequence(self.p_destination_, block_num, lstm_size, scope = "location_embedding", reuse = True)
      cell, initial_state = self.build_lstm(tf.shape(self.p_known_)[0])
      outputs, final_state = tf.nn.dynamic_rnn(cell, tf.transpose(p_known_embedding, [1, 0, 2]), initial_state = initial_state, dtype=tf.float32, time_major=True)

      with tf.variable_scope('policy_output'):
        w_p = tf.Variable(tf.truncated_normal([lstm_size, block_num], stddev=0.1))
        b_p = tf.Variable(tf.zeros(block_num))

      print(p_destination_embedding, final_state[0][1])
      self.policy = tf.matmul(tf.reshape(tf.add(tf.expand_dims(p_destination_embedding, 0), outputs), [-1, lstm_size]), w_p) + b_p
      print(self.policy)
      self.policy = tf.transpose(tf.reshape(self.policy, [tf.shape(self.p_known_)[1], tf.shape(self.p_known_)[0], block_num]), [1, 0, 2])
      self.action = tf.argmax(self.policy, axis=2)

      action_one_hot = tf.one_hot(self.p_output_, block_num)
      self.policy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.policy , labels=action_one_hot)
      self.policy_loss = tf.reduce_mean(self.policy_loss)
      self.policy_optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.policy_loss) 
  def create_heuristics_network(self):
    self.known_ = tf.placeholder(tf.int64, shape=(batch_size, input_steps), name='known')
    known_embedding = tf.contrib.layers.embed_sequence(self.known_, block_num, lstm_size, scope = "policy_network/location_embedding", reuse = True)
    self.waiting_ = tf.placeholder(tf.int64, shape=(batch_size), name='waiting')
    waiting_embedding = tf.contrib.layers.embed_sequence(self.waiting_, block_num, lstm_size, scope = "policy_network/location_embedding", reuse = True)
    self.destination_ = tf.placeholder(tf.int64, shape=(batch_size), name='destination')
    destination_embedding = tf.contrib.layers.embed_sequence(self.destination_, block_num, lstm_size, scope = "policy_network/location_embedding", reuse = True)
    # network weights

    with tf.variable_scope("value_network"):
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
  
      output_state = tf.reduce_mean(hidden_states, 0) + local_2_layer

      with tf.variable_scope('output'):
        w_h = tf.Variable(tf.truncated_normal([lstm_size, 1], stddev=0.1))
        b_h = tf.Variable(tf.zeros(1))
        w_t = tf.Variable(tf.truncated_normal([lstm_size, 1], stddev=0.1))
        b_t = tf.Variable(tf.zeros(1))
      

      self.heuristics = tf.nn.sigmoid(tf.matmul(output_state, w_h) + b_h)

      self.time = tf.nn.relu(tf.matmul(output_state, w_t) + b_t)

      self.heuristics_input = tf.placeholder(tf.float32, [batch_size], name = "heuristics_input")

      self.time_input = tf.placeholder(tf.float32, [batch_size], name = "time_input")
      self.heuristics_cost = tf.reduce_mean(tf.square(self.heuristics_input - self.heuristics))
 
      self.gradients = tf.gradients(self.heuristics_cost, [output_state])
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
      vocabulary[key] = count
      try:
        self.token2cor[count] = string2cor[key] 
      except KeyError:
        pass
      count += 1 
    batches = []
    for batch in data:
      bb = []
      for tra in np.array(batch)[:, :]: #np.array(batch)[:, :, 0]
        aa = []
        for rec in tra:
          aa.append(vocabulary[rec])
        bb.append(aa)
      batches.append(bb)  
#    print(len(batches), len(batches[0]), len(batches[0][0])) 
#    print(np.array(batches).shape, "00000")
    self.train_batches = batches[0:20000][:][:]
    self.test_batches = batches[20000:25000][:][:]
    pickle.dump(self.token2cor, open("/data/wuning/mobile trajectory/token2cor", "wb"), -1)
    pickle.dump(self.train_batches, open("/data/wuning/mobile trajectory/trainSet", "wb"), -1)
    pickle.dump(self.test_batches, open("/data/wuning/mobile trajectory/testSet", "wb"), -1)
    print("process data finish")
  def load_data(self):
    File = open("/data/wuning/mobile trajectory/trainSet", "rb")
    self.train_batches = pickle.load(File)
    File = open("/data/wuning/mobile trajectory/testSet", "rb")
    self.test_batches = pickle.load(File)

  def pre_train(self, PRE_EPISODE):
#    self.all_saver.restore(self.session, "/data/wuning/AstarRNN/pretrain_test_policity_neural_network_epoch0.ckpt")
    pre_batches = []
    pre_test_batches = []
    for batch in self.train_batches:
      pre_batches.append([np.array(batch)[:,:-1], np.array(batch)[:,1:], np.array(batch)[:,-1]])    
    for test_batch in self.test_batches:
      pre_test_batches.append([np.array(test_batch)[:,:-1], np.array(test_batch)[:,1:], np.array(test_batch)[:,-1]]) 
#    pre_batches = np.array(pre_batches)
    for episode in range(PRE_EPISODE):
      counter = 0
      for batch in pre_batches:
#        print(np.array(batch[1]).shape)
#        print(np.array(batch[2]).shape)
#        print(np.array(batch[0]).shape)
        self.policy_optimizer.run(feed_dict={
          self.p_known_:batch[0],
          self.p_destination_:batch[2],
          self.p_output_:batch[1]
        })
        eval_policy_loss = self.policy_loss.eval(feed_dict={
          self.p_known_:batch[0],
          self.p_destination_:batch[2],
          self.p_output_:batch[1]
        })
        
        if counter % 100 == 0:
          print("epoch:{}...".format(episode),
                "batch:{}...".format(counter),
                "loss:{:.4f}...".format(eval_policy_loss))
        counter += 1
      average_loss = 0
      for test_batch in pre_test_batches:
        test_policy_loss = self.policy_loss.eval(feed_dict={
          self.p_known_:test_batch[0],
          self.p_destination_:test_batch[2],
          self.p_output_:test_batch[1]
        })
        average_loss += test_policy_loss
      print("test_loss:", average_loss / len(pre_test_batches))
      self.all_saver.save(self.session, "/data/wuning/AstarRNN/pretrain_test_policity_neural_network_epoch{}.ckpt".format(episode))

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
    return the_list

  def move(self, item, the_list, f_score):
    for it in the_list:
      if(f_score[it] == f_score[item]):
        the_list.remove(it)
        break
    return self.insert(item, the_list, f_score)


  def greedyTest(self):
    counters = 0
    for batch in self.test_batches:
      result = []
      for start, end in zip(batch[:, 0], batch[:, -1]):
        pass 
  def AstarTest(self):    
    results = []
    counters = 0
    all_len = 0
    for batch in self.train_batches:
      result = []
      count = 0
      for start, unknown, end in zip(np.array(batch)[:, 0], np.array(batch)[:,1:-1], np.array(batch)[:, -1]):
        closedSet = []
        openSet = [start]
        cameFrom = {}
        pathFounded = {start: [start]}
        fScore = {}
        fScore[start] = 0.001
        while len(openSet) > 0:
          current = openSet[0]
          if current == end:
            result.append(pathFounded[current])
            break
        
          openSet.remove(current)
          closedSet.append(current)
          policy_value = self.policy.eval(
              feed_dict={
                self.p_known_:np.array(pathFounded[current])[np.newaxis, :],
                self.p_destination_:[end]
          })
          policy = np.argsort(policy_value, axis=1)[:,:NEXT_ACTION_NUM]
#          print("policy_value:", policy_value)
#          print("start:", start)
#          print("end:", end)
#          print("path:", pathFounded[current])
#          print("policy:", policy)
#          print("unknown:", unknown)
          for waiting in policy[0]:
            if waiting in closedSet:
              continue
            f_score = self.heuristics.eval(
                feed_dict={
                  self.known_:np.array(pathFounded[current])[np.newaxis, :],
                  self.waiting_:[waiting],
                  self.destination_:[end]
                  }
              )

            if (waiting in fScore) and (f_score < fScore[waiting]):
              continue
            fScore[waiting] = f_score
            if waiting not in openSet:
              openSet = self.insert(waiting, openSet, fScore)
            else:
              openSet = self.move(waiting, openSet,fScore)
            temp = copy.deepcopy(pathFounded[current])
            temp.append(waiting)
            pathFounded[waiting] = temp
          count += 1
        print("count:", count)
      results.append(result)
      for infer, real in zip(result, batch):     
        print("infer:", infer)
        print("real:", real)
        for item in infer[1:-1]:
          if item in real[1:-1]:
            counters += 1 
        all_len += len(infer) - 2
    return float(counters)/all_len

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


#          reward_batch = np.array([[[float(self.token2cor[item][0]), float(self.token2cor[item][1])] for item in items] for items in wait_actions]) - np.array([[float(self.token2cor[item][0]), float(self.token2cor[item][1])] for item in np.array(batch)[:, i]])[:, np.newaxis,:]
#          y_batch = np.exp(np.sum(- reward_batch**2, axis=2) / self.sigma)   
        y_batch = np.zeros(wait_actions.shape)
        for k in range(y_batch.shape[0]):
          if(not np.array(batch)[k, i] == wait_actions[k, 0]):
            wait_actions[k, 1] = wait_actions[k, 0]
            wait_actions[k, 0] = np.array(batch)[k, i]
          y_batch[k, 0] = 1
        for j in range(2):
#            if(i < len(batch[0]) - 3):
#              eval_next_policy = self.policy.eval(
#              feed_dict={
#                self.p_known_:np.concatenate((np.array(batch)[:, 0:i], wait_actions[:, j][:, np.newaxis]), 1),
#                self.p_destination_:np.array(batch)[:, -1]
#              })
#              wait_next_actions = np.argsort(-eval_next_policy, axis=1)[:,:NEXT_ACTION_NUM]
#              heuristics_batch = []  #下一个状态的Q值
#              for k in range(NEXT_ACTION_NUM):
#                heuristics_batch.append(self.heuristics.eval(
#                  feed_dict={
#                    self.known_:np.array(batch)[:, 0:i],
#                    self.waiting_:wait_next_actions[:, k],
#                    self.destination_:np.array(batch)[:, -1]
#                    }
#                  )
#                )
#              heuristics_batch = np.array(heuristics_batch)
#              heuristics_batch = np.max(heuristics_batch, axis = 0)
#              y_batch[:, j] += heuristics_batch[:, 0]

          heuristics_batches.append([np.array(batch)[:, 0:i], wait_actions[:, j], np.array(batch)[:, -1], y_batch[:, j]])
      if counter % 100 == 0:
        print(len(heuristics_batches))
        print("batches generated:{}...".format(counter))
      if counter % 500 == 0:
        pickle.dump(heuristics_batches, open("/data/wuning/mobile trajectory/heuristicsTrainSet"+str(counter), "wb"))
        heuristics_batches = []
      counter += 1

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
        for batch in heuristics_batches:
          feed_data = {
            self.known_:batch[0],
            self.waiting_:batch[1],
            self.destination_:batch[2],
            self.heuristics_input:batch[3]
          }
          self.optimizer.run(feed_dict=feed_data)
          heuristics_cost = self.heuristics_cost.eval(feed_dict=feed_data)
          heuristics_value = self.heuristics.eval(feed_dict=feed_data)
          heuristics_grad = self.gradients[0].eval(feed_dict=feed_data)
          if counter % 501 == 0:
            print("y_batch:", batch[3])
            print("epoch:{}...".format(episode),
                  "batch:{}...".format(counter),
#                  "heuristics{}...".format(heuristics_value),
#                  "grads{}...".format(heuristics_grad),
                  "loss:{:.4f}...".format(heuristics_cost))
          counter += 1
      self.all_saver.save(self.session, "/data/wuning/AstarRNN/train_heuristics_reward_neural_network_epoch{}.ckpt".format(episode))

      # Test every 100 episodes
      if (episode + 1) % 100 == 0:
        accuracy = self.AstarTest()
        print('episode: ',episode,'average accuracy:',accuracy)
# ---------------------------------------------------------
EPISODE = 100 # Episode limitation
PRE_EPISODE = 300
TRAIN_BATCHES = 300 # Step limitation in an episode

def main():

  AstarRNN = DAN()  

  if(PRE_TRAIN):
    AstarRNN.pre_train(PRE_EPISODE)
  elif(RESTORE):
    AstarRNN.all_saver.restore(tf.get_default_session(), "/data/wuning/AstarRNN/pretrain_test_policity_neural_network_epoch37.ckpt") 
  if(TEST):
    accuracy = AstarRNN.AstarTest()
    print('average accuracy:',accuracy)
  else:
#     AstarRNN.generate_heuristics_samples()
    AstarRNN.heuristics_train()
#    AstarRNN.pre_train(PRE_EPISODE) 
if __name__ == '__main__':
  main()
