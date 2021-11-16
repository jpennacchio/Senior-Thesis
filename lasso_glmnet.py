##Imports
import numpy as np
import glmnet_python
from glmnet import glmnet; from glmnetCoef import glmnetCoef
from glmnetPredict import glmnetPredict; from cvglmnet import cvglmnet
from cvglmnetCoef import cvglmnetCoef
from data_preprocessing import transform_data_with_controls
#from runAnyMethod import *


###alpha in this is the equivalent of lambda in the other lasso method; alpha often refers to the shifting of weight between ridge
###regression and elastic net
def glmLassoSolve(A,b, perform_cv= True):
  ###The default argument for the glmnet Lasso includes an intercept. 
  if (perform_cv == True):
      clf = cvglmnet(x = A,y = b, intr= False)
      lambda_to_use = clf['lambda_min']
      coef = cvglmnetCoef(clf, s = lambda_to_use)[1:].ravel()  
  else:
      ####shouldn't be forcing lambda here, but for the phase diagram we should; otherwise it will take forever.####
      temp_lambda = np.array([0.30])
      clf = glmnet(x = A,y = b, intr= False, lambdau = temp_lambda)
      ###This is an nXnX1 grid--->only need 1 coefficient set, and exclude the intercept
      coef = glmnetCoef(clf)[1:,1].ravel()
      lambda_to_use = 0.30
  ###Must exclude intercept here as well
  return coef, lambda_to_use

####Default to perform 10-fold Cross Validation.####
def glmLassoSolveControls(A,b, controls, perform_cv= True):
    ###This is using the method mirroring Chernuzokov
    if(len(controls) !=0):
        modified_variables = transform_data_with_controls(controls, [], b, A)
        d_lasso = modified_variables [0]
        z_lasso = modified_variables [1]
        ###d is an array, z is a matrix. Don't know why.###
        ###Use Cross Validation here#####
        if (perform_cv == True):
            clf = cvglmnet(x = z_lasso,y = d_lasso, intr= False)
            lambda_to_use = clf['lambda_min']
            coef = cvglmnetCoef(clf, s = lambda_to_use)[1:].ravel() 
            lambda_to_use = lambda_to_use[0]
        else:
            temp_lambda = np.array([0.30])
            clf = glmnet(x = z_lasso,y = d_lasso, intr= False, lambdau = temp_lambda)
            ###This is an nXnX1 grid--->only need 1 coefficient set, and exclude the intercept
            coef = glmnetCoef(clf)[1:,1].ravel()
            lambda_to_use = 0.30
    else:
        res = glmLassoSolve(A,b,perform_cv)
        coef = res[0].ravel()
        lambda_to_use = res[1]
    return coef, lambda_to_use

def glmLassoSolveControls_old(A,b, controls, perform_cv = True):
  ###The default argument for the glmnet Lasso includes an intercept. 
  #clf = glmnet(x = A,y = b, intr= False)
  ###If no controls, want the other function to be called. 
  if (len(controls) != 0):
      num_vars_to_force = controls.shape[1]
      A = np.concatenate((controls, A), axis=1)
      num_instruments = A.shape[1]-num_vars_to_force
      ###penalty factor argument; penalty factor of zero means variable will always be in the model####
      pen_fac = np.array([0,1])
      pen_fac = np.repeat(pen_fac, (num_vars_to_force, num_instruments))
      if (perform_cv ==True):  
          clf = cvglmnet(x = A,y = b, intr= False, penalty_factor = pen_fac)
          lambda_to_use = clf['lambda_min']
          coef= cvglmnetCoef(clf, s = lambda_to_use)[1:].ravel()
          nonzero_coef = np.abs(coef[num_vars_to_force:])
          num_nonzero_inst = sum(i>0.001 for i in nonzero_coef)
          if(num_nonzero_inst ==0):
             all_lambdas = clf['lambdau']
             counter = 0
             while (num_nonzero_inst == 0):
                 print("Entered While Loop")
                 lambda_to_use = np.atleast_1d(np.array(all_lambdas[counter]))
                 coef = cvglmnetCoef(clf, s = lambda_to_use)[1:].ravel()
                 nonzero_coef = np.abs(coef[num_vars_to_force:])
                 num_nonzero_inst = sum(i>0.001 for i in nonzero_coef)
                 counter = counter + 1   
      else:  
        ####shouldn't be forcing lambda here, but for the phase diagram we should; otherwise it will take forever.####
          temp_lambda = np.array([0.30])
          clf = glmnet(x = A,y = b, intr= False, penalty_factor = pen_fac, lambdau = temp_lambda)
          ###This is an nXnX1 grid--->only need 1 coefficient set, and exclude the intercept
          coef= glmnetCoef(clf)[1:,1].ravel()
          lambda_to_use = 0.30
      ###Don't print the intercept; it will make things a lot more confusing
      nonzero_coef = np.nonzero(coef)[0]
      #Throw out the forced variables. Can't simply subset in case 1 or 2 of the controls are excluded even with zero penalty. 
      nonzero_inst = nonzero_coef[np.logical_and(nonzero_coef>(num_vars_to_force-1),nonzero_coef>0)]
  else:
      object = glmLassoSolve(A,b, perform_cv)
      coef = object[0]
      lambda_to_use = object [1]
      nonzero_coef= np.nonzero(coef)[0]
      ###Since no controls, all nonzeros can be classified as "instruments."
      nonzero_inst = nonzero_coef
  

  return coef, lambda_to_use, nonzero_coef, nonzero_inst

def test1():
  """
  Test on slightly larger matrix (5 observations * 10 predictors)
  Also test with one additional predictor, and forcing that one predictor into the Lasso
  """
  ###Note: Lasso did not converge with this test case.
  # Create test observations:
  A = np.array([[1, 0.9, 0.8, 0.2, -0.7, 1.3, 2.3, -0.5, 1.4, -0.2],[1, -0.1, 0.7, 0.3, -0.2, 0.9, 0.7, 1.3, 1.8, -2.5], [1,0.5,0.4,0.3,0.9,1.9, 2.3, -3.1, -2.9, -0.7], [1,0.2,0.9,0.1,0.7,2.3, 0.5, 2.5, -1.5, -2.9], [1,0.5,1,1.2,1.5,-0.1, 0.3, 0.7, 1.9, -0.6]])
  controls2 = np.array([[0.3], [0.7], [-0.9], [0.4], [2.3]])
  A_controls = np.concatenate((controls2, A), axis=1)
  X = np.array( [[0.5], [0], [0], [0], [-0.2], [0], [0], [3.1], [0], [0]])
  X_controls = np.array( [[0.1], [0.5], [0], [0], [0], [-0.2], [0], [0], [3.1], [0], [0]])
  B = A.dot(X)
  B_controls = A_controls.dot(X_controls)
  ####Don't double-include controls. Need to include only once. 
  #x_lasso=glmLassoSolveControls(A,B, [])
  #print(x_lasso)
  #x_lasso2 = run_choice_method_once (4, A, B)
  #print(x_lasso2)
  #x_lasso=glmLassoSolveControls(A,B_controls, controls2)
  #x_lasso2=run_choice_method_once(4, A, B_controls, controls=controls2)
  #print("First Run")
  #print(x_lasso[0])
####Note: This is different since OLS is now being run on the selected variables.#######
  #print("Second Run")
  #print(x_lasso2)
  ####Do some debugging here with the fixed alpha value. #####
  #x_lasso=glmLassoSolveControls(A,B_controls, controls2, perform_cv = False)
  #x_lasso2=run_choice_method_once(4, A, B_controls, controls=controls2, perform_cv= False)
  #print("First Run")
  #print(x_lasso[0])
####Note: This is different since OLS is now being run on the selected variables.#######
  #print("Second Run")
  x_lasso = glmLassoSolveControls(A,B_controls, controls2, perform_cv = True)
  #x_lasso2 = run_choice_method_once(4, A, B_controls, controls=controls2, perform_cv= False, num_folds=10)
  x_lasso3 = glmLassoSolveControls(A_controls, B_controls, [], perform_cv = True)
  print(x_lasso)
  #print(x_lasso2)
  print(x_lasso3)
# =============
# MAIN FUNCTION
# =============
if __name__ == "__main__":
  
  # Perform Tests
  #print("")
  #print("=== Test 1 ===")
  #print("")
  test1()
  #print("")
  #print("=== Test 2 ===")
  #print("")
  #test2() 