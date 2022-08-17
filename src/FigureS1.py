# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import sys


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -

data=pd.read_hdf('../data/FigureS1.h5')

# +
rat_neural = data['rat']['neural']
rat_behavior = data['rat']['behavior']
fig=plt.figure(figsize=(15,5))

ax=plt.subplot(111)
ax.imshow(rat_neural.T, aspect= 'auto', cmap='gray_r', vmax=1)
plt.ylabel('Neurons', fontsize=45)
plt.xlabel('Time (s)', fontsize= 45)
plt.xticks(np.linspace(0, len(rat_neural),5), np.arange(0,45,10))
plt.yticks([0,50,100], [0,50,100])
plt.xticks(fontsize = 45)
plt.yticks(fontsize = 45)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

r=rat_behavior[:,1]==1
l=rat_behavior[:,1]==0
fig = plt.figure(figsize=(15,5), dpi=300)
ax=plt.subplot(111)
ax_r=ax.scatter(np.arange(len(rat_behavior))[r]*0.025, rat_behavior[r,0],  c=rat_behavior[r,0], cmap = 'viridis', s=10)
ax_l=ax.scatter(np.arange(len(rat_behavior))[l]*0.025, rat_behavior[l,0],  c=rat_behavior[l,0], cmap='cool', s=10, alpha = 0.5)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.ylabel('Position (m)', fontsize=45)
plt.xlabel('Time (s)', fontsize= 45)
plt.xticks(fontsize = 45)
plt.yticks(np.linspace(0,1.6,3), fontsize = 45)
plt.xticks(np.linspace(0, len(rat_behavior),5)*0.025, np.arange(0,45,10))
cb_r_axes = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cb_l_axes = fig.add_axes([0.96, 0.15, 0.02, 0.7]) 
cb_r = plt.colorbar(ax_r, cax = cb_r_axes,boundaries=np.linspace(0,1.6, 200))
cb_l = plt.colorbar(ax_l, cax = cb_l_axes, boundaries=np.linspace(0,1.6,200), ticks = np.linspace(0,1.6, 5)) 
cb_r.set_ticks([])

cb_r.ax.set_xlabel('Right', fontsize = 15)
cb_l.ax.set_xlabel('Left', fontsize = 15)


# +
active_target = data['monkey']['behavior']['active']['target']
passive_target = data['monkey']['behavior']['passive']['target']
active_pos = data['monkey']['behavior']['active']['position']
passive_pos = data['monkey']['behavior']['passive']['position']

fig=plt.figure(figsize=(10,5))

ax1 = plt.subplot(1,2,1)
ax1.set_title('Active trials', fontsize=20)
for n,i in enumerate(active_pos.reshape(-1, 600,2)):
    k = active_target[n*600]
    ax1.plot(i[:,0], i[:,1], color=plt.cm.hsv(1/8*k), linewidth = 0.5)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
plt.axis('off')
ax2 = plt.subplot(1,2,2)
ax2.set_title('passive trials', fontsize=20)
for n,i in enumerate(passive_pos.reshape(-1, 600,2)):
    k = passive_target[n*600]
    ax2.plot(i[:,0], i[:,1], color=plt.cm.hsv(1/8*k), linewidth = 0.5)
plt.axis('off')

# +

fig=plt.figure(figsize=(15,5))
ephys = data['monkey']['neural']
ax=plt.subplot(111)
ax.imshow(ephys[:600].T, aspect= 'auto', cmap='gray_r', vmax=1, vmin=0)
plt.ylabel('Neurons', fontsize=20)
plt.xlabel('Time (s)', fontsize= 20)
plt.xticks([0,200,400,600], ['0','200','400', '600'], fontsize = 20)
plt.yticks(fontsize = 20)
plt.yticks([25,50], ['0', '50'])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# +
neuropixel=data['mouse']['neural']['np'] 
ca=data['mouse']['neural']['ca'] 
fig=plt.figure(figsize=(15,10), dpi=300)
plt.subplots_adjust(hspace=0.5)
ax1=plt.subplot(2,1,1)
plt.imshow(neuropixel[:100, :240], aspect= 'auto', vmin = 0, vmax=1.5, cmap = 'gray_r')
plt.ylabel('Neurons', fontsize=45)
plt.xlabel('Time (s)', fontsize= 45)
plt.xticks(np.linspace(5,240,5), np.linspace(0, 2, 5 ) )
plt.yticks([0,50,100], [100,50,0])
plt.xticks(fontsize = 45)
plt.yticks(fontsize = 45)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

ax2=plt.subplot(2,1,2)
ax2.plot(ca.T)
plt.ylabel('dF/F', fontsize=45)
plt.xlabel('Time (s)', fontsize= 45)
plt.ylim(0,6)
#plt.xlim(0,1200)
plt.xticks(np.linspace(0, 1200,5), np.linspace(0, 40, 5 ).astype(int), fontsize=20, )
plt.yticks([0,3,6], [0,3,6], fontsize=45)
plt.xticks(fontsize = 45)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
# -

plt.figure(figsize=(15,5))
for n,i in enumerate(data['mouse']['behavior']):
    ax = plt.subplot(1,3,n+1)
    ax.imshow(i, cmap='gray')
    plt.axis('off')
    


