# Imports
import numpy as np
import matplotlib.pyplot as plt

def adjustPD(pdData):
  pdData = np.transpose(pdData)
  return pdData

# Load Result Files
lpRes  = np.loadtxt('02_results/second_LP.out')
ompRes = np.loadtxt('02_results/second_OMP.out')
larRes = np.loadtxt('02_results/second_LASSOLAR.out')
qpRes  = np.loadtxt('02_results/second_LASSOQP.out')

# Adjust Results
lpRes  = adjustPD(lpRes)
ompRes = adjustPD(ompRes)
larRes = adjustPD(larRes)
qpRes  = adjustPD(qpRes)

# Plot 
colIdx = lpRes.shape[1]//2
plt.figure()
plt.plot(lpRes[:,colIdx],label='LP')
plt.plot(ompRes[:,colIdx],label='OMP')
plt.plot(larRes[:,colIdx],label='LAR')
plt.plot(qpRes[:,colIdx],label='QP')
plt.legend()
plt.show()



