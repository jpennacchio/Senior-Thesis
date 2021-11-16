# Imports
import sys
import numpy as np
from constants import *

def preProcess(observations,demean,scale):
  """Preprocess observation matrix before starting the OMP iterations
  
  Args:
    observations: rectangular matrix with observations
    demean: Integer flag to demean columns in observation matrix. 
            It can assume the following values:
            - ipDemeanALL:   Subtract the associated mean from all columns
            - ipDemeanFIRST: Will do the same as ALL, but have call separate so easier to work with OMP
                             and switching back to the original observ. after 1st loop.
            - ipDemeanNONE:  Do not subtract the mean.

    scale: Integer flag to scale the columns of the observation matrix.
           It can assume the following values
           - ipScaleYES: Divide each column by its standard deviation
           - ipScaleNO:  Do not scale columns

  Returns:
    newObs: The modified observation matrix with demeaned/scaled columns

  """

  # Find Columns Mean/Std
  column_means = np.sum(observations,axis=0)/float(observations.shape[0])  
  column_sd = np.std(observations,axis=0)

  # If there is an intercept column, which by construction has 0 SD, set it to 1 to avoid any issues
  # This works since dividing by 1 won't scale the intercept at all. 
  if np.abs(column_sd[0]) < kMathZero:
    column_sd[0] = 1.0

  if(demean == ipDemeanALL or demean == ipDemeanFIRST):
    newObs = observations - column_means
  elif(demean == ipDemeanNONE):    
    newObs = observations
  else:
    print("ERROR: Demean Option not Found.")
    sys.exit(-1)

  # Arguments for scale are "YES" and "NO"
  if(scale == ipScaleYES):
    newObs = newObs/column_sd

  # Return the modified observations
  return newObs

def omp(observations, outcomes, tol=1e-5, demean=ipDemeanNONE, scale=ipScaleNO, force_intercept=ipForceInterceptNO, ols_modified_matrix=ipModNO, printMsg=False):
  """Orthogonal Matching Pursuit Algorithm  
  
  Args:
    observations: rectangular matrix with observations
    outcomes: rhs term    
    tol (default 1e-5): numerical tolerance on the OMP residual 2-norm
    demean (default ipDemeanNONE): Option to demean the observation matrix
    scale (default ipScaleNO): Option to scale the columns of the observation matrix
    force_intercept (default ipForceInterceptNO): force selection of column 0 as the first column in OMP
    ols_modified_matrix (default ipModNo): option to use the demeaned and/or scaled matrix throughout the OLS process 
  Returns:
    res: The solution of the underdeterminate linear system found through OMP

  """
##Initializing the counter here.
  num_loops_completed = 0
  # Use A List To Store The Columns in and out of the Model
  if(force_intercept == ipForceInterceptYES):
    columnsInModel = [0]
    ##Count the intercept selection as the first time through the loop.
    num_loops_completed += 1
    columnsOutModel = [i for i in range(1,observations.shape[1])]
  else:
    columnsInModel = []
    columnsOutModel = [i for i in range(observations.shape[1])]
  
  # Initialize Residual using the outcomes
  # Make Sure it is a column vector
  residuals = np.copy(outcomes)

  # Initializing to a number larger than the threshold. 
  resNorm = 100.0
  
  # PRE-PROCESSING. Modified this to be renamed new_observ and to only be used when computing the inner vector product
  new_observ = preProcess(observations,demean,scale)

  # Print Header
  if(printMsg):
    print('--- OMP Iterations')
    print('----------------------------------')
    print('%5s %10s %15s' % ('It.','Sel.','Res Norm'))
    print('----------------------------------')

  # OMP Main Loop  
  while((resNorm > tol)and(num_loops_completed < observations.shape[0])):
    ##Doing this sets the observations equal to the modified observations. Now, OLS will be performed on the modified
    ##observations. Only want to do this once, though.
    if(ols_modified_matrix == ipModYES and num_loops_completed == 0):
        observations = new_observ;  
    ##If only want to demean once, when the counter is 1, set new_observ equal to observations [so now just work with observ].
    if(demean == ipDemeanFIRST and num_loops_completed == 0):
        new_observ = observations;  
    ######CHANGES MADE HERE##########
    # Find Minimum Regression Error
    colSqNorms = np.square(np.linalg.norm(new_observ[:,columnsOutModel],2,axis=0))
    z = np.abs(np.sum(new_observ[:,columnsOutModel] * residuals,axis=0))/colSqNorms
    error = np.square(np.linalg.norm(new_observ[:,columnsOutModel] * z - residuals,2,axis=0))
    ####RESUME ORIGINAL CODE########
    # Find the Column with the minimum error
    addColID = columnsOutModel[np.argmin(error)]

    # Add it to the Index Set of variables in the Model
    columnsInModel.append(addColID)
    # Delete it from the Index Set of variables outside the Model
    columnsOutModel.remove(addColID)

    # Solve the Least Square Problem [on the original observations, not the preprocessed data]
    coeff = np.linalg.lstsq(observations[:,columnsInModel],outcomes)[0] 
    residuals = outcomes - observations[:,columnsInModel].dot(coeff)

    # Calculate 2-Norm of the Residual
    resNorm = np.linalg.norm(residuals,2)

    # Print Messages
    if(printMsg):
      print('%5d %10d %15.3e' % (num_loops_completed,addColID,resNorm))

    # Increment Number of Iterations
    num_loops_completed += 1
    
  # Trying to create escape condition in case no convergence occurs. 
  # Can't have more variables than observations
  if(num_loops_completed == observations.shape[0] and resNorm > tol):
    print("OMP Iterations Not Converged!")
  ##Same return regardless of whether converge or not. Difference is printing the statement. 
  # Have one beta hat for each column in the observations. 
  res = np.zeros(observations.shape[1])
  res[columnsInModel] = coeff.ravel()
  # Return
  return res, len(columnsInModel)


def test1():
  """
  Test on small matrix (5 observations * 6 predictors)
  """
  # Create test observations:
  A = np.array([[1, 0.9, 0.8, 0.2, -0.7, 1.3],[1, -0.1, 0.7, 0.3, -0.2, 0.9], [1,0.5,0.4,0.3,0.9,1.9], [1,0.2,0.9,0.1,0.7,2.3], [1,0.5,1,1.2,1.5,-0.1]])
  X = np.array([[0.5], [0.3], [0], [0], [0], [0]])
  B = A.dot(X)
  omp1betas = omp(A,B,algChoice=1)[0]
  # Compare
  print('Solution Diff: ', np.linalg.norm(omp1betas.flatten()-X.flatten(),2))

# =============
# MAIN FUNCTION
# =============
if __name__ == "__main__":
  
  # Perform Tests
  test1()
