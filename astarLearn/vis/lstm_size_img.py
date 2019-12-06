import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
font = {'family' : 'normal',
        'size'   : 17}

matplotlib.rc('font', **font)

x = np.array([128, 256, 384, 512, 640])
y2 = np.array([0.613, 0.662, 0.692, 0.703, 0.689])
y1 = np.array([0.587, 0.587, 0.587, 0.587, 0.587])
plt.ylabel("F1")
#plt.figure(figsize=(13, 7))
l1, = plt.plot(x, y1, marker='o', mec='r', mfc='w')
l2, = plt.plot(x, y2, marker='*', ms=10)
plt.legend([l1, l2], ['RICK', 'NASR'], loc = 'upper left')
plt.ylim(ymin = 0.5)
plt.ylim(ymax = 0.8)  

fig = plt.gcf()
ax = plt.gca()
ax.spines['top'].set_visible(False)  #去掉上边框
ax.spines['right'].set_visible(False) #去掉右边框
#fig.set_size_inches(8.0, 6.0)
fig.savefig('lstm_size.svg',  format='svg')

