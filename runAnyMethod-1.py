###The functions in this file are used to test the various different methods, compare true solutions to solutions from the algorithm,
###and to run an algorithm of the user's choice. 

# Imports
import sys
import numpy as np
from constants import *
from random import randint ##Imported for Random Numbers
from random import sample ##Select list of unique random integers

# Solvers
from omp import omp
###Work this into the solving options####
from lasso_glmnet import *
###############################################
from lasso import lassoSolve, lasso_force_controls
from lassoLARS import lassoLarsSolve, lassoLARS_force_controls
from lp import lpSolve
from l1regls import l1regls
from cvxopt import matrix, solvers


###This function runs the desired solver once. Can define this in makePhaseDiagrams to avoid repetitiveness. 
###Empty controls by default, but can use controls with any of these###
###Will make performing cv the default. For phase diagrams, need to use different argument
def run_choice_method_once(solver, A, b, controls = [], perform_cv = True):
    if (solver == ipOmp):
        sol = omp(A,b,controls, tol=1e-5)[0]
        ###Work on putting OMP CV here###
    elif (solver == ipLp):
        sol = lpSolve(A,b).ravel()
    else:
        if (solver == ipLasso):
        # Find support with Lasso
            tmpSol = lasso_force_controls(A,b, controls, perform_cv)[0].ravel() 
        elif (solver == ipGlmnetLasso):
            tmpSol = glmLassoSolveControls(A,b,controls, perform_cv)[0].ravel()
        elif (solver == ipLassoLARS):
            tmpSol = lassoLars_force_controls(A,b,controls, perform_cv)[0].ravel()
        elif (solver == ipL1regls):
        # Find support with quadratic programming
            tmpSol = np.array(l1regls(matrix(A),matrix(b))).ravel()  
        ###############################################################################################################
        ###Combine the controls with the original dataframe before performing Least Squares####
        if (len(controls) !=0):
            A = np.concatenate ((controls, A), axis=1)
        ########################################################################################
        ####Now continue with the same procedure###
        sol = np.zeros(A.shape[1])
        ###I think this should only be needed for l1regls. Otherwise, we're re-running selected lasso variables with OLS 
        support = (np.abs(tmpSol) > 1.0e-5)
        if(np.any(support)):
             sol[support] = np.linalg.lstsq(A[:,support],b)[0].ravel() 
        ##################################################################################################################
    return sol


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