import random
import copy
import numpy as np
import pickle
from multiprocessing import Process
epoches = 1
def generate_samples(OSMadj, ind, trainData):
    st_batches = []
    neg_batches = []
    for e in range(epoches):
        print(ind, e)
        for i in range(len(trainData)):
            if i % 100 == 0:
                print(ind, i)
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
                    neg_tra.append(random.sample(temp_neig, 1))
                neg_tra.append(tra[-1])
                neg_batch.append(neg_tra)
            st_batches.append(st_batch)
            neg_batches.append(neg_batch)
    pickle.dump(st_batches,open("/data/wuning/learnAstar/st_samples"+str(ind), "wb"))   
    pickle.dump(neg_batches,open("/data/wuning/learnAstar/neg_samples"+str(ind), "wb"))         
    return st_batches, neg_batches
OSMadj = pickle.load(open("/data/wuning/map-matching/fmm-master/fmm-master/example/ofoOSMadjMat", "rb"))
trainData = pickle.load(open("/data/wuning/map-matching/fmm-master/fmm-master/example/ofoOSMBeijingtrainData", "rb"))
if __name__=="__main__":
    n_processes = 8 #number of processes
    n_total = len(trainData)
    length = int(float(n_total) / float(n_processes))
    indices = [i* length for i in range(n_processes)]

    sublists = [trainData[indices[i]:indices[i+1]] for i in range(n_processes - 1)]
    processes = [Process(target=generate_samples,args=(OSMadj, i,x)) for i,x in enumerate(sublists)]

    for p in processes:
        p.start()
    for p in processes:
        p.join()

