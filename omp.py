# Imports
import sys
import numpy as np
from constants import *
from data_preprocessing import preProcess, transform_data_with_controls
from ols import ols


def omp(observations, outcomes, forced_variables, tol=1e-5, demean=ipDemeanNONE, scale=ipScaleNO, force_intercept=ipForceInterceptNO, ols_modified_matrix=ipModNO, printMsg=False, cutoff = 1.00):
  """Orthogonal Matching Pursuit Algorithm 
  Args:
    observations: rectangular matrix with observations
    outcomes: rhs term    
    forced_variables: allows for a specific set of controls to be forced to be included in the model, before sparse variable selection
    tol (default 1e-5): numerical tolerance on the OMP residual 2-norm
    demean (default ipDemeanNONE): Option to demean the observation matrix
    scale (default ipScaleNO): Option to scale the columns of the observation matrix
    force_intercept (default ipForceInterceptNO): force selection of column 0 as the first column in OMP
    ols_modified_matrix (default ipModNo): option to use the demeaned and/or scaled matrix throughout the OLS process 
  Returns:
    res: The solution of the underdeterminate linear system found through OMP

  """
###Converts 1D outcomes and forced variables to the necessary 2D array for the later steps####
  if(outcomes.ndim == 1):
      outcomes = outcomes[:,np.newaxis] 
  if(len(forced_variables) !=0):
      if(forced_variables.ndim ==1):
          forced_variables = forced_variables[:,np.newaxis] 

##Initializing the counter here.
  num_loops_completed = 0
  num_low_reductions = 0
  finished_first_loop = 0
    
  # Use A List To Store The Columns in and out of the Model
   ###If an array is being used, the length will be nonzero. If [] is used, its length is zero.
  ####################################################################################################
  # Initialize Residual using the outcomes
  # Make Sure it is a column vector
  residuals = np.copy(outcomes)
  # Initializing to a number larger than the threshold.
 
  resNorm = 10000.0
  #####################################################################################################
  ###This section appears to be good. 
  if (len(forced_variables) != 0):
    #######Combine the forced controls with the optional instruments.################### 
    observations = np.concatenate((forced_variables, observations), axis=1)
    ####################################################################################
    ###Mandate that the forced variables enter the model.
    columnsInModel = list(range(0, forced_variables.shape[1]))
    num_loops_completed += forced_variables.shape[1]
    columnsOutModel = [i for i in range(forced_variables.shape[1],observations.shape[1])]
    ####Run regression with forced terms###
    # Solve the Least Square Problem [on the original observations, not the preprocessed data]
    #print(columnsInModel)
    coeff = np.linalg.lstsq(observations[:,columnsInModel],outcomes)[0] 
    residuals = outcomes - observations[:,columnsInModel].dot(coeff)
    # Calculate 2-Norm of the Residual
    resNorm = np.linalg.norm(residuals,2)/(np.linalg.norm(outcomes,2))
    #print('Norm with only the controls')
    #print(resNorm)
    ##Intercept won't be forced if have other forced variables
  elif(force_intercept == ipForceInterceptYES):
    columnsInModel = [0]
    ##Count the intercept selection as the first time through the loop.
    num_loops_completed += 1
    columnsOutModel = [i for i in range(1,observations.shape[1])]
   # Solve the Least Square Problem [on the original observations, not the preprocessed data]
    coeff = np.linalg.lstsq(observations[:,columnsInModel],outcomes)[0] 
    residuals = outcomes - observations[:,columnsInModel].dot(coeff)
    # Calculate 2-Norm of the Residual
    resNorm = np.linalg.norm(residuals,2)/(np.linalg.norm(outcomes,2))
  else:
    columnsInModel = []
    columnsOutModel = [i for i in range(observations.shape[1])]
  

  ################################################################################
  # PRE-PROCESSING. Modified this to be renamed new_observ and to only be used when computing the inner vector product
  new_observ = preProcess(observations,demean,scale)

  # Print Header
  if(printMsg):
    print('--- OMP Iterations')
    print('----------------------------------')
    print('%5s %10s %15s' % ('It.','Sel.','Res Norm'))
    print('----------------------------------')
    print(resNorm)
    print('Norm with controls and some instruments')
  # OMP Main Loop  
  ###Adding threshold of 1% error reduction; if fail on 3 consecutive iterations return columns from 3 rounds prior. 
  while((resNorm > tol)and(num_loops_completed < observations.shape[0]) and (num_loops_completed < observations.shape [1]) and (num_low_reductions<3) ):
    ##Doing this sets the observations equal to the modified observations. Now, OLS will be performed on the modified
    ##observations. Only want to do this once, though.
    if(ols_modified_matrix == ipModYES and finished_first_loop == 0):
        observations = new_observ;  
    ##If only want to demean once, when the counter is 1, set new_observ equal to observations [so now just work with observ].
    if(demean == ipDemeanFIRST and finished_first_loop == 0):
        new_observ = observations;  
    # Find Minimum Regression Error
    colSqNorms = np.square(np.linalg.norm(new_observ[:,columnsOutModel],2,axis=0))
    ###Consider possibility when column square norms are zero. Replace with number close to zero so division can occur####
    colSqNorms[colSqNorms ==0] = 0.00001
    z = np.abs(np.sum(new_observ[:,columnsOutModel] * residuals,axis=0))/colSqNorms
    error = np.square(np.linalg.norm(new_observ[:,columnsOutModel] * z - residuals,2,axis=0))
    # Find the Column with the minimum error
    addColID = columnsOutModel[np.argmin(error)]
    # Add it to the Index Set of variables in the Model
    columnsInModel.append(addColID)
    # Delete it from the Index Set of variables outside the Model
    columnsOutModel.remove(addColID)
    # Solve the Least Square Problem [on the original observations, not the preprocessed data]
    #print("2-3")
    #print(columnsInModel)
    #print(len(columnsInModel))
    coeff = np.linalg.lstsq(observations[:,columnsInModel],outcomes)[0] 
    #coeff = ols(outcomes, [], observations[:,columnsInModel], return_type = 2, dataframe = False)
    #coeff = coeff[:,np.newaxis]
    #print(coeff)
    #print("3-0")
    ###consider possibility that L2 norm of outcomes is zero###
    a = (outcomes - observations[:,columnsInModel].dot(coeff))
    b = (np.linalg.norm(outcomes,2))
    if (a.all()==0 and b==0):
        ###set b=1 so division occurs properly. 
        b=1
    residuals = a/b
    ###Want a percentage improvement comparison.
    resNormOld = resNorm
    # Calculate 2-Norm of the Residual
    resNorm = np.linalg.norm(residuals,2)/(np.linalg.norm(outcomes,2))
    resNormReduction = (resNorm-resNormOld)/resNormOld *100.00
    if(printMsg):
        print(resNorm)
        print(resNormReduction)
    
    ##############################################################################
    ###Work with the 1% threshold. 
    if(abs(resNormReduction)<cutoff):
        num_low_reductions += 1
    else:
        num_low_reductions = 0
    #################################################################################
    # Print Messages
    if(printMsg):
      print('%5d %10d %15.3e' % (num_loops_completed,addColID,resNorm))
    # Increment Number of Iterations
    finished_first_loop = 1
    num_loops_completed += 1
    
  ###If this escape condition occurs, re-run the last good step to obtain correct coefficient estimates and residuals.
  if (num_low_reductions == 3):
    del columnsInModel[-3:]
    coeff = np.linalg.lstsq(observations[:,columnsInModel],outcomes)[0]  
    residuals = outcomes - observations[:,columnsInModel].dot(coeff)

  # Trying to create escape condition in case no convergence occurs. 
  # Can't have more variables than observations
  elif(num_loops_completed == min(observations.shape[0], observations.shape[1]) and resNorm > tol):
    print("OMP Iterations Not Converged!")
    
  ##Same return regardless of whether converge or not. Difference is printing the statement. 
  # Have one beta hat for each column in the observations. 
  res = np.zeros(observations.shape[1])
  res[columnsInModel] = coeff.ravel()
  # Return
  ###Want to see which columns are included in the model, and also obtain the residuals for the residual bootstrap. 
  return res, len(columnsInModel), columnsInModel, resNorm

def omp_controls(observations, outcomes, forced_variables, tol=1e-5, demean=ipDemeanNONE, scale=ipScaleNO, force_intercept=ipForceInterceptNO, ols_modified_matrix=ipModNO, printMsg=False, cutoff = 1.00):
    if(len(forced_variables)!= 0):
        modified_variables = transform_data_with_controls(forced_variables, [], outcomes, observations)
        d = modified_variables [0]
        z = modified_variables [1]
        z = np.asarray(z)
        res = omp(z, d, [], tol, demean, scale, force_intercept, ols_modified_matrix, printMsg, cutoff)
    else:
        res = omp(observations, outcomes, [], tol, demean, scale, force_intercept, ols_modified_matrix, printMsg, cutoff)

    return res  



def test1():
  """
  Test on small matrix (5 observations * 6 predictors)
  """
  # Create test observations:
  A = np.array([[1, 0.9, 0.8, 0.2, -0.7, 1.3],[1, -0.1, 0.7, 0.3, -0.2, 0.9], [1,0.5,0.4,0.3,0.9,1.9], [1,0.2,0.9,0.1,0.7,2.3], [1,0.5,1,1.2,1.5,-0.1]])
  X = np.array([[0.5], [0.3], [0], [0], [0], [0]])
  B = A.dot(X)
  bla = []
  #omp1betas = omp(A,B, bla,algChoice=1)[0]
  omp1betas = omp(A,B, bla)[0]
  #omp2betas = omp_old(A,B)[0]
  # Compare
  #print('Solution Diff: ', np.linalg.norm(omp1betas.flatten()-X.flatten(),2))
  #print('Solution Diff: ', np.linalg.norm(omp2betas.flatten()-X.flatten(),2))
  print(omp1betas)
  #print(type(omp1betas))
  ###Same result; the new argument does not change the program when no terms are forced in. 

def test2():
  """
  Test on slightly larger matrix (5 observations * 10 predictors)
  """
  # Create test observations:
  A = np.array([[1, 0.9, 0.8, 0.2, -0.7, 1.3, 2.3, -0.5, 1.4, -0.2],[1, -0.1, 0.7, 0.3, -0.2, 0.9, 0.7, 1.3, 1.8, -2.5], [1,0.5,0.4,0.3,0.9,1.9, 2.3, -3.1, -2.9, -0.7], [1,0.2,0.9,0.1,0.7,2.3, 0.5, 2.5, -1.5, -2.9], [1,0.5,1,1.2,1.5,-0.1, 0.3, 0.7, 1.9, -0.6]])
  X = np.array( [[0.5], [0], [0], [0], [-0.2], [0], [0], [3.1], [0], [0]])
  B = A.dot(X)
  bla = []
    
  omp1betas = omp(A,B,[], tol=0.01, demean=ipDemeanFIRST, scale=ipScaleYES)[0]
  #omp2betas = omp_old(A,B, tol=0.01, demean=ipDemeanFIRST, scale=ipScaleYES)[0]

  # Compare
  print('Solution Diff: ', np.linalg.norm(omp1betas.flatten()-X.flatten(),2))
  #print('Solution Diff: ', np.linalg.norm(omp2betas.flatten()-X.flatten(),2))
 

  # If I force the intercept in, this runs perfectly. 
  omp1betas = omp(A,B,[], tol=0.01, demean=ipDemeanFIRST, scale=ipScaleYES, force_intercept=ipForceInterceptYES)[0]
  #omp2betas = omp_old(A,B, tol=0.01, demean=ipDemeanFIRST, scale=ipScaleYES, force_intercept=ipForceInterceptYES)[0]

  # Compare
  print('Solution Diff: ', np.linalg.norm(omp1betas.flatten()-X.flatten(),2))
  #print('Solution Diff: ', np.linalg.norm(omp2betas.flatten()-X.flatten(),2))

def test3():
  """
  Test on slightly larger matrix (5 observations * 10 predictors)
  """
  ###Note: Lasso did not converge with this test case.
  # Create test observations:
  A = np.array([[1, 0.9, 0.8, 0.2, -0.7, 1.3, 2.3, -0.5, 1.4, -0.2],[1, -0.1, 0.7, 0.3, -0.2, 0.9, 0.7, 1.3, 1.8, -2.5], [1,0.5,0.4,0.3,0.9,1.9, 2.3, -3.1, -2.9, -0.7], [1,0.2,0.9,0.1,0.7,2.3, 0.5, 2.5, -1.5, -2.9], [1,0.5,1,1.2,1.5,-0.1, 0.3, 0.7, 1.9, -0.6]])
  X = np.array( [[0.5], [0], [0], [0], [-0.2], [0], [0], [3.1], [0], [0]])
  B = A.dot(X)
  B = B.ravel()
  controls2 = np.array([[0.3], [0.7], [-0.9], [0.4], [2.3]])
  A_controls = np.concatenate((controls2, A), axis=1)
  X_controls = np.array( [[0.5], [0.5], [0], [0], [0], [-0.2], [0], [0], [3.1], [0], [0]])
  B_controls = A_controls.dot(X_controls)
  B_controls = B_controls.ravel()

  omp1 = omp(A, B_controls, controls2, cutoff = 0.001)  
  omp11 = omp(A, B_controls, controls2)  
  ###Using the residual method to show results are identical for OMP###
  omp22 = omp_controls(A, B_controls, controls2)
  ###cutoff ended up not making a difference in this application
  omp3 = omp(A_controls, B_controls, [],  cutoff = 0.001)  
  ###Forcing variables, however, did.
  print(omp1)
  print(omp11)
  print(omp22)
  print(omp3)
    
    
# =============
# MAIN FUNCTION
# =============
if __name__ == "__main__":
  
  # Perform Tests
  #print("")
  #print("=== Test 1 ===")
  #print("")
  #test1()
  #print("")
  #print("=== Test 2 ===")
  #print("")
  #test2()
  #print("")
  #print("=== Test 3 ===")
  #print("")
  test3()  
