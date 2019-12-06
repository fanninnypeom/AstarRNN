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
    File = open("/data/wuning/mobile trajectory/Q_learning_trainSet", "rb")   #trainSet
    self.train_batches = pickle.load(File)
    File = open("/data/wuning/mobile trajectory/Q_learning_testSet", "rb")    #testSet
    self.test_batches = pickle.load(File)
    File = open("/data/wuning/mobile trajectory/Q_learning_token2cor", "rb")
    self.token2cor = pickle.load(File)

    print("batch_size:", len(self.train_batches[0]))
  def load_beijing_data(self):
    File = open("/data/wuning/AstarBeijing/beijingPolicyTrainSet", "rb")   #trainSet
    self.train_batches = pickle.load(File)
    File = open("/data/wuning/AstarBeijing/beijingTestSet", "rb")    #testSet
    self.test_batches = pickle.load(File)
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

