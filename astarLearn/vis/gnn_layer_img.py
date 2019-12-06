import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
font = {'family' : 'normal',
        'size'   : 17}

matplotlib.rc('font', **font)

x = np.array([2, 3, 4, 5, 6, 7])
y2 = np.array([0.679, 0.687, 0.693, 0.698, 0.703, 0.699])
y1 = np.array([0.661, 0.661, 0.661, 0.661, 0.661, 0.661])
plt.ylabel("F1")
#plt.figure(figsize=(13, 7))
l1, = plt.plot(x, y1, marker='o', mec='r', mfc='w')
l2, = plt.plot(x, y2, marker='*', ms=10)
plt.legend([l1, l2], ['GA', 'NASR'], loc = 'upper right')
plt.ylim(ymin = 0.65)
plt.ylim(ymax = 0.73)

fig = plt.gcf()
ax = plt.gca()
ax.spines['top'].set_visible(False)  #去掉上边框
ax.spines['right'].set_visible(False) #去掉右边框
#fig.set_size_inches(8.0, 6.0)
fig.savefig('gnn_size.svg',  format='svg')

