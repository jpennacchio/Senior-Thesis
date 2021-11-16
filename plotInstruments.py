import numpy as np
import matplotlib.pyplot as plt

# Plot the instruments
def plotInstruments(file,keyTable,qtyName):
  # Get key table
  kt = np.loadtxt(keyTable,delimiter=',',dtype=str)
  # Get the instruments IDs
  it = np.loadtxt(file,delimiter=',',skiprows=1,dtype=str)

  # Init list
  instrumentList = [[] for i in range(kt.shape[0])]
  for loopA in range(it.shape[0]):
  	currMethod = int(it[loopA,0])
  	currInstr = int(it[loopA,1])
  	instrumentList[currMethod-1].append(currInstr)

  exMarkers = ['o', 'v', 'D', 'P', '>','s', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']

  # Make Scatter Plot
  plt.figure(figsize=(5,3))
  ax  = plt.subplot(1,1,1)
  for loopA in range(kt.shape[0]):
    plt.scatter(instrumentList[loopA],np.ones(len(instrumentList[loopA]))*loopA,marker=exMarkers[loopA],lw=0.2,edgecolors='k')

  # Set fontsize
  fs = 10

  # Set Title
  ax.set_title(qtyName,fontsize=fs)
  ax.set_xlabel('Selected Instruments',fontsize=fs)
  ax.set_ylabel('Method',fontsize=fs)
  ax.set_yticklabels(kt[:,0],rotation=45)
  ax.set_yticks(np.arange(kt.shape[0]))
  ax.tick_params(direction='out',axis='both',which='major',labelsize=fs)
  ax.set_ylim([-0.5,5.5])
  ax.xaxis.grid(False)

  plt.tight_layout()

  plt.savefig(qtyName+'.png',dpi=300)

# ============
# MAIN ROUTINE
# ============
if __name__== "__main__":
  
  # Get File Names
  dataFileNames = ['CS Instruments.csv','FHFA Instruments.csv','GDP Instruments.csv','NM Instruments.csv']
  qtyNames = ['CS','FHFA','GDP','NM']
  keyTable = 'methodID.csv'

  for loopA in range(len(dataFileNames)):
    plotInstruments(dataFileNames[loopA],keyTable,qtyNames[loopA])

