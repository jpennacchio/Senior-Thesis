# Imports
import sys
import numpy as np
import matplotlib.pyplot as plt ##Imported for the sparsity oversampling/undersampling diagram

##Is there a way to change name of text file without manually changing it each time?
filename_1 = str(sys.argv[1])
filename_2 = str(sys.argv[2])
a = np.loadtxt(filename_1)
b = np.loadtxt(filename_2)

###NEED TO TRANSPOSE FIRST, THEN HORIZONTALLY FLIP TO GET CORRECT GRID FORMAT
a = np.transpose(a)
b = np.transpose(b)
a = np.flipud(a)
b = np.flipud(b)

# Compute ticks values
ticksValue = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])

###Code to plot the sparsity oversampling/undersampling diagram.

##Modify this line when deviate from 5X5, but this is for testing. 
#my_xticks = np.linspace(0.2, 1, a.shape[1]) 
fig, ax = plt.subplots()
im = ax.imshow(a, cmap=plt.get_cmap('jet'), vmin=0.0, vmax=1.2)
ax.grid(False)
ax.set_xlabel(r'$\delta$',fontsize=10)
ax.set_ylabel(r'$\rho$',fontsize=10)
ax.set_xticklabels(ticksValue)
ax.set_yticklabels(ticksValue)
ax.set_xticks(ticksValue*a.shape[0])
ax.set_yticks(ticksValue[::-1]*a.shape[0])
ax.tick_params(direction='out')
fig.colorbar(im)
print('new file: ',filename_1.replace('.out', '.pdf'))
plt.savefig(filename_1.replace('.out', '.pdf'))

fig, ax = plt.subplots()
im = ax.imshow(b, cmap=plt.get_cmap('jet'), vmin=0.0, vmax=1.0)
ax.grid(False)
ax.set_xlabel(r'$\delta$',fontsize=10)
ax.set_ylabel(r'$\rho$',fontsize=10)
ax.set_xticklabels(ticksValue)
ax.set_yticklabels(ticksValue)
ax.set_xticks(ticksValue*a.shape[0])
ax.set_yticks(ticksValue[::-1]*a.shape[0])
ax.tick_params(direction='out')
fig.colorbar(im)
print('new file: ',filename_2.replace('.out', '.pdf'))
plt.savefig(filename_2.replace('.out', '.pdf'))
