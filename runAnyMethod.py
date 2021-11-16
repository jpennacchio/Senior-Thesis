###The functions in this file are used to test the various different methods, compare true solutions to solutions from the algorithm,
###and to run an algorithm of the user's choice. 

# Imports
import sys
import numpy as np
from constants import *
from random import randint ##Imported for Random Numbers
from random import sample ##Select list of unique random integers
from data_preprocessing import transform_data_with_controls

# Solvers
from omp import omp, omp_controls
###Work this into the solving options####
from lasso_glmnet import glmLassoSolve, glmLassoSolveControls, glmLassoSolveControls_old
from lasso import lassoSolve, lasso_force_controls
from lassoLARS import lassoLarsSolve, lassoLARS_force_controls
from lp import lpSolve
from l1regls import l1regls
from cvxopt import matrix, solvers

###OMP Cross Validation
from crossValidation import n_fold_cross_validation


###This function runs the desired solver once. Can define this in makePhaseDiagrams to avoid repetitiveness. 
###Empty controls by default, but can use controls with any of these###
###Will make performing cv the default. For phase diagrams, need to use different argument
def run_choice_method_once(solver, A, b, controls = [], num_folds = 10, num_iterations = 1000, perform_cv = True, ols_solution = True, instrument_list= False, chern_ols = False):
    ###In these cases, don't transform to OLS; already done.####
    possible_cutoffs= [0.01, 0.05, 0.1, 0.5, 1, 2, 3, 4, 5, 10, 15]
    if (solver == ipOmp):
        ###This is the OMP Non-Chernozukov Cross Validation [should run correctly]. Other method isn't done yet.###
        min_cutoff = n_fold_cross_validation(A, b, controls, shuffle=True, cutoff_list = possible_cutoffs, omp_control_method = False)       
        print(min_cutoff)
        sol = omp(A,b,controls, tol=1e-5, cutoff = min_cutoff)[0]
        support = np.asarray(np.where(np.abs(sol) > 1.0e-5)).flatten()
    elif (solver == ipLp):
        sol = lpSolve(A,b).ravel()
        support = np.asarray(np.where(np.abs(sol) > 1.0e-5)).flatten()
    elif (solver == ipGlmnetNoControls):
        tmpSol = glmLassoSolveControls_old(A, b, controls, perform_cv)[0].ravel()
        support = np.asarray(np.where(np.abs(tmpSol) > 1.0e-5)).flatten()
        A = np.concatenate((controls, A), axis=1)  
        sol = np.zeros(A.shape[1])
        sol[support]= np.linalg.lstsq(A[:,support],b)[0].ravel()
    else:
        if(solver == ipL1regls):
            tmpSol = np.array(l1regls(matrix(A),matrix(b))).ravel()  
        elif (solver ==ipOmpControls):
            ###First, need to perform Cross Validation to ensure the correct cutoff is used###
            ###Also, should there be a guarantee that at least one instrument is selected?###   
            min_cutoff = n_fold_cross_validation(A, b, controls, shuffle = True, cutoff_list = possible_cutoffs, omp_control_method = True)
            print(min_cutoff)
            tmpSol = omp_controls(A, b, controls, tol=1e-5, cutoff = min_cutoff)[0]
        elif (solver == ipLasso):
            tmpSol = lasso_force_controls(A, b, controls, num_folds, num_iterations, perform_cv)[0].ravel() 
        elif (solver == ipGlmnetLasso):
            tmpSol = glmLassoSolveControls(A,b,controls, perform_cv)[0].ravel()
        elif (solver == ipLassoLARS):
            tmpSol = lassoLARS_force_controls(A, b, controls, num_folds, num_iterations, perform_cv)[0].ravel()
        ###now, using instruments selected by Lasso/OMP, perform OLS, and, eventually, 2SLS. #####
        if(ols_solution == True):
            ###prints positions of indexes, rather than T/F###
            support = np.asarray(np.where(np.abs(tmpSol) > 1.0e-5)).flatten()
            if(chern_ols==True):
                if(len(controls)!=0):
                    ##Even though this was computed earlier on, need to compute again since not returned in earlier functions.
                    modified_data = transform_data_with_controls(controls,[],b,A)
                    d_mod = modified_data[2]
                    z_mod = modified_data[3]
                    y_mod = modified_data[4]
                    ###Initialize sol###
                    sol = np.zeros(z_mod.shape[1])
                    if(np.any(support)):
                    ###This is the setup used in the Chernuzokov paper. 
                       sol[support] = np.linalg.lstsq(z_mod[:,support],d_mod)[0].ravel()                         
                else:                  
                    sol = np.zeros(A.shape[1])
                    if(np.any(support)):
                        sol[support] = np.linalg.lstsq(A[:,support],b)[0].ravel() 
            ###This is the more logical way of performing OLS, unless I know how to transform the coef. back###
            else:
            ###Alternate sol method d~x+z, with z as the subset of instruments picked[no modification]
                if(len(controls) !=0):  
                    num_controls = controls.shape[1]
                    support[:] = [q + num_controls for q in support]
                    control_indexes = np.asarray(range(num_controls))
                    support = np.concatenate((control_indexes, support))
                    A = np.concatenate((controls, A), axis=1)  
                    sol = np.zeros(A.shape[1])
                    if(np.any(support)):
                        sol[support] = np.linalg.lstsq(A[:,support],b)[0].ravel() 
                else:
                    sol = np.zeros(A.shape[1])
                    if(np.any(support)):
                        sol[support]= np.linalg.lstsq(A[:,support],b)[0].ravel() 
        ###If want to keep the Lasso/OMP/non-OLS coefficients, use ols_solution = False argument. ###
        else:
            sol = tmpSol
    if(instrument_list == False): ##Default is False; makes it easier with the other functions. ##
        return sol
    else:
        return sol, support

def compare(x,y):
    '''
    This function compares the result from the orthogonal matching pursuit algorithm to any pre-populated result 
    used in the testing and development of this analysis. 
    
    Returns:
    difference_12_norm: norm of the difference between the true betas and the OMP solution
    correct_sln_norm: norm of the true solution
    fractional error: difference_12_norm/correct_sln_norm (as defined in the OMP paper).
    '''
    difference_12_norm = np.linalg.norm(x.flatten()-y.flatten(),2)
    correct_sln_norm = np.linalg.norm(y.flatten(),2)
    ##Alleviate issue of dividing by zero when there are no nonzero x's. 
    if(correct_sln_norm == 0):
       fractional_error = 0
    else:
       fractional_error = difference_12_norm/float(correct_sln_norm)
    return(difference_12_norm, correct_sln_norm, fractional_error)

def generateAXB(delta,rho, nvars=200, force_intercept=ipForceInterceptNO, print_detail=ipPrintNO):
    """Creates AX=B form. Randomly generates the observations and the betas, then back out the outcomes.
    OMP takes in the observations and the outcomes. 
    Args:
    delta: number of observations/number of variables
    rho: number of nonzeroes (in variables)/number of observations
    nvars: number of variables, defaulted to 200. 
    
    From the 3 above arguments, can back out the number of observations and the number of nonzeroes.
    
    force_intercept (default ipForceInterceptNO): force selection of column 0 as the first column in OMP
    print_detail (default ipPrintNO): print more detailed output

  Returns:
    res: The solution of the underdeterminate linear system found through OMP
    """
        
    nobs = int(nvars * delta)
    number_of_nonzeroes = int(rho * nobs)
    if(print_detail==ipPrintYES):
       print(nobs)
       print(number_of_nonzeroes)
    
    ##Generate A [observations]
    if (force_intercept == ipForceInterceptYES):
        ##First, implement it in A. Then, require it to present. 
        A = np.random.rand(nobs,nvars-1)
        ones= np.ones((nobs, 1))
        A = np.concatenate((ones, A), axis=1)
    else:
        A = np.random.rand(nobs,nvars)
    #######################################################################
    ##Now generate the x's. Generate from N(6,1)
    X = np.random.normal(6, 1, (nvars,1))
    # X = np.random.normal(0, 1, (nvars,1))
    ##Now, based on the number of nonzeroes specified early, pick which x's to set to zero. 
    num_zeros = nvars - number_of_nonzeroes;
    ###Now pick where the zeros come from####

    if (force_intercept == ipForceInterceptYES):
        s = sample(range(1,nvars), num_zeros)
    else:
        s = sample(range(0,nvars), num_zeros) 
    ##Now loop through "s" and for each element of "s" set corresponding
    ##value in x to be zero.
    for i in range(0, len(s)):
        b = s[i]
        X[b]=0
    #######################################################################
    ##Calculate AX=B
    A1 = np.matrix(A)
    X1 = np.matrix(X)
    B1 = A1 * X1
    B = np.array(B1) 
    return (A, X, B);

#####Self-Contained Test Code######
def test1():
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

  ####NOTE: USE A 1D ARRAY INSTEAD OF A COLUMN VECTOR USING RAVEL. 
  #x_lasso=lassoSolve(A,B, num_iterations = 2000, num_folds = 2)
  #x_lasso2=lasso_force_controls(A,B, [], num_iterations = 2000, num_folds = 2)
  ###work with CV, now try no CV
  print('Lasso Force Lasso Solution')
  x_lasso = lasso_force_controls(A, B_controls, controls2, num_folds=2, perform_cv = False)[0].flatten()
  print(x_lasso)
  print('Lasso Force OLS Solution')
  x_lasso2 = run_choice_method_once(2, A, B_controls, controls=controls2, perform_cv= False)
  print(x_lasso2)
  print('Lasso No Force Solution')
  x_lasso3 = lasso_force_controls(A_controls, B_controls, [], num_folds = 2, perform_cv = False)[0].flatten()
  print(x_lasso3)
  print('Lasso No Force OLS Solution')
  x_lasso4 = run_choice_method_once(2, A_controls, B_controls, [],perform_cv = False)
  print(x_lasso4)
    
    
  #print('Result 1')
  #x_lasso = glmLassoSolveControls(A,B_controls, controls2, perform_cv = False)
  #print('Result 2')
  #x_lasso2 = run_choice_method_once(4, A, B_controls, controls=controls2, perform_cv= False)
  #print('Result 3')
  #x_lasso3 = glmLassoSolveControls(A_controls, B_controls, [], perform_cv = False)
  #print(x_lasso)
  #print(x_lasso2)
  #print(x_lasso3)
  #omp1 = omp(A, B_controls, controls2, cutoff = 0.001)  
  #omp11 = omp(A, B_controls, controls2)  
  #omp2 = run_choice_method_once(0, A, B_controls, controls=controls2, perform_cv= False) 
  ###Using the residual method to show results are identical for OMP###
  #omp22 = omp_controls(A, B_controls, controls2)
  ###cutoff ended up not making a difference in this application
  #omp3 = omp(A_controls, B_controls, [],  cutoff = 0.001)  
  ###Forcing variables, however, did.
  #print(omp1)
  #print(omp11)
  #print(omp2)
  #print(omp22)
  #print(omp3)
    
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