import tensorflow as tf
import numpy as np
import random
import pickle
import copy
from collections import deque

# Hyper Parameters for DAN
PRE_TRAIN = True
TEST = False
RESTORE = False
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
  train_batches = batches[0:20000][:][:]
  test_batches = batches[20000:25000][:][:]
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
    all_saver.save(self.session, "/data/wuning/AstarRNN/pretrain_test_policity_neural_network_epoch{}.ckpt".format(episode))


