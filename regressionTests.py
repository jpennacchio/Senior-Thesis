import numpy as np
from constants import *
from lasso import lassoSolve
from lp import lpSolve
from l1regls import l1regls
from cvxopt import matrix, solvers
from lassoLARS import lassoLarsSolve
from lasso_glmnet import glmLassoSolve

# =============
# MAIN FUNCTION
# =============
if __name__ == "__main__":

  # Set solver type
  #solverType = ipLp
  solverType = ipLasso
  # solverType = ipL1regls
  #solverType = ipGlmnetLasso
  #solverType = ipLassoLARS

  # Set Parameters
  totRows = 8
  totCols = 10
  nnZeros = 2

  # Create a Gaussian Matrix A
  A = np.random.randn(totRows,totCols)
  # Create Sparse Random Vector
  x = np.zeros(totCols)
  supp = np.random.choice(np.arange(totCols),size=nnZeros)
  x[supp] = np.random.normal(5.0,1.0,size=nnZeros)
  # Generate RHS
  b = A.dot(x)

  # Solve
  if(solverType == ipLp):
    sol = lpSolve(A,b).ravel()
  elif(solverType == ipLasso):
    sol = lassoSolve(A,b, num_folds=3)[0].ravel()
  elif(solverType == ipGlmnetLasso):
    sol = glmLassoSolve(A,b)[0].ravel()
  elif(solverType == ipLassoLARS):
    sol = lassoLarsSolve(A,b, num_folds=3)[0].ravel()
  elif(solverType == ipL1regls):
    sol = np.array(l1regls(matrix(A),matrix(b))).ravel()

  print ('--- Difference Norm: %.3e') 
  print (np.linalg.norm(sol-x,2))
  print ('--- Detailed Difference')
  if(solverType == ipLp):
    print ('%-10s %-10s'), print('True','LP')
  elif(solverType == ipLasso):
    print ('%-10s %-10s'), print('True','Lasso')
  elif(solverType == ipL1regls):
    print ('%-10s %-10s'), print('True','L1RegLs')
  elif(solverType == ipLassoLARS):
    print ('%-10s %-10s'), print('True','LassoLARS')
  elif(solverType == ipGlmnetLasso):
    print ('%-10s %-10s'), print('True','GlmnetLasso')
  for loopA in range(len(x)):
    print ('%-10.3e %-10.3e'), print (x[loopA],sol[loopA])
