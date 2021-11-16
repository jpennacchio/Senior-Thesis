# coding: utf-8
# Imports
import sys
import numpy as np
from constants import *
from random import randint ##Imported for Random Numbers
from random import sample ##Select list of unique random integers
from runAnyMethod import generateAXB, compare, run_choice_method_once

###Solvers are present within runAnyMethod. But need omp for testing in this file.
from omp import omp
##This is for parallel programming
from joblib import Parallel, delayed


def solve_n_times(solver, n, delta, rho, nvars=200, tol=1e-5, solution_comp_norm=1e-3, ols_solution = True, controls = [], num_folds = 10, num_iterations = 1000, perform_cv = False):
    '''
    This function repeats the OMP algorithm "n" times.
    Arguments are the same as those discussed before.
    n: number of iterations
    tol: tolerance used in the OMP Algorithm
    solution_comp_norm: when looking at the fractional errors, this is threshold to determine if above/below. 
    
    
    Returns:
    nonzeroes_present: counts the number of nonzeroes in the returned beta hats. Ideally this will perfectly match
                        the number in the sparse signal.
    residual_norms: these were the norms returned when comparing correct solution to OMP solution.
    fractional_error: replicates the residual norms/norm of correct solution coming from the OMP Paper
    mean_fractional_error: takes the mean of the fractional errors
    prop_exceed_threshold: looks at the proportion of the fractional errors above a certain threshold.
                           low proportion implies good signal recovery; high proportion implies the reverse. 
    '''
    nonzeroes_present = []
    residual_norms    = []
    fractional_error  = []
    for i in range (0,n):
        ###Printing Iteration Number
        #print(i)
        AXB = generateAXB(delta,rho, nvars)
        ###Calls a function that runs the desired method once. ####NO CV HERE#####
        sol = run_choice_method_once(solver, AXB[0], AXB[2], controls, num_folds, num_iterations, perform_cv, ols_solution)
        result = compare(sol, AXB[1])
        ##Create a vector with sol[1], which contains number of nonzeroes
        #nonzeroes_present = np.append(nonzeroes_present,sol[1])
        #residual_norms = np.append(residual_norms,result[0])
        fractional_error = np.append(fractional_error, result[2])
        
    mean_fractional_error = np.mean(fractional_error)
    ##This computes the proportion of fractional errors exceeding a given tolerance.
    prop_exceed_threshold = sum(i>solution_comp_norm for i in fractional_error)/float(n)
    print('Completed - delta %.3f, rho %.3f' % (delta, rho))
    return (mean_fractional_error, prop_exceed_threshold)
     
def makePhaseDiagrams(solver,num_iterations=100, nvars=200, num_subdiv=20, tol=1e-5, solution_comp_norm=1e-3, nproc=8, ols_solution = True):
    '''
    This function is the final test loop for OMP. This will look at rho and delta in the appropriate increments.
    Output from this will be used to create the final sparsity sampling diagrams. 
    
    Arguments:
    num_iterations: number of times OMP will be executed on each of the (delta,rho) combinations.
    num_subdiv: determines the number of total increments at which delta and rho increase. 
    
    Returns:
    stored_fractional_errors: grid containing the mean fractional errors from num_iterations runs through each
                              (delta,rho) combination
    stored_prop_exceed_threshold: similar grid, but printing proportion of those mean fractional errors above a given threshold
    '''
     
    stored_fractional_errors= np.zeros((num_subdiv,num_subdiv), dtype=np.float)
    stored_prop_exceed_threshold= np.zeros((num_subdiv,num_subdiv), dtype=np.float)

    omp_deltarho_results= Parallel(n_jobs = nproc, verbose = 150)(delayed(solve_n_times)(solver,num_iterations, i/float(num_subdiv), \
                                 j/float(num_subdiv), nvars, tol, solution_comp_norm, ols_solution) \
                                 for i in range(1,num_subdiv+1) for j in range(1, num_subdiv+1))

    ####Output is returned as tuples; need to subdivide
    stored_fractional_errors = [x[0] for x in omp_deltarho_results]
    stored_prop_exceed_threshold = [x[1] for x in omp_deltarho_results]
    ###Convert to array, then rearrange in correct shape. 
    stored_fractional_errors = np.asarray(stored_fractional_errors)
    stored_prop_exceed_threshold = np.asarray(stored_prop_exceed_threshold)
    stored_fractional_errors= stored_fractional_errors.reshape(num_subdiv, num_subdiv)
    stored_prop_exceed_threshold= stored_prop_exceed_threshold.reshape(num_subdiv, num_subdiv)
    
    return(stored_fractional_errors, stored_prop_exceed_threshold)
            

# =======================
# TESTING FUNCTIONALITIES
# =======================

def test2():
  """
  Test on slightly larger matrix (5 observations * 10 predictors)
  """
  A = np.array([[1, 0.9, 0.8, 0.2, -0.7, 1.3, 2.3, -0.5, 1.4, -0.2],[1, -0.1, 0.7, 0.3, -0.2, 0.9, 0.7, 1.3, 1.8, -2.5], [1,0.5,0.4,0.3,0.9,1.9, 2.3, -3.1, -2.9, -0.7], [1,0.2,0.9,0.1,0.7,2.3, 0.5, 2.5, -1.5, -2.9], [1,0.5,1,1.2,1.5,-0.1, 0.3, 0.7, 1.9, -0.6]])
  X = np.array( [[0.5], [0], [0], [0], [-0.2], [0], [0], [3.1], [0], [0]])
  B = A.dot(X)
  # Solve OMP
  sol = omp(A,B,[], tol=0.01, demean=ipDemeanFIRST, scale=ipScaleYES)[0]
  # Compare
  print('Solution Diff: ', compare(sol, X))

  # If I force the intercept in, this runs perfectly. 
  sol = omp(A,B,[], tol=0.01, demean=ipDemeanFIRST, scale=ipScaleYES, force_intercept=ipForceInterceptYES)[0]
  # Compare
  print('Solution Diff: ', compare(sol, X))
  
def test3():
  """
  Try playing around with the same test and no intercept
  """
  A = np.array([[1, 0.9, 0.8, 0.2, -0.7, 1.3, 2.3, -0.5, 1.4, -0.2],[1, -0.1, 0.7, 0.3, -0.2, 0.9, 0.7, 1.3, 1.8, -2.5], [1,0.5,0.4,0.3,0.9,1.9, 2.3, -3.1, -2.9, -0.7], [1,0.2,0.9,0.1,0.7,2.3, 0.5, 2.5, -1.5, -2.9], [1,0.5,1,1.2,1.5,-0.1, 0.3, 0.7, 1.9, -0.6]])
  X = np.array( [[0], [0], [0], [0], [-0.2], [0], [1.5], [3.1], [0], [0]])
  B = A.dot(X)
  # Want the solution to have 4, 6, 7.
  # DEMEAN=ALL, SCALE=YES, FORCE_INTERCEPT=NO
  # Selected the first 2 correctly but didn't select the 3rd unfortunately. 2/3 isn't horrible though. 
  sol = omp(A,B, [], tol=0.01, demean=ipDemeanALL, scale=ipScaleYES)[0]
  print('Solution Diff: ', compare(sol, X))

def test4():
  """
Performing one iteration of what will be in test 5. 
  """
  #result = generateAXB(0.05, 0.05, 400, force_intercept, print_detail)
  #omp1betas = omp(result[0],result[2], tol, demean, scale, force_intercept, ols_modified_matrix)
  result = generateAXB(0.05, 0.05, 200)
  omp1betas = omp(result[0],result[2], [])
  omp_result = compare(omp1betas[0], result[1])
  #print("Printing AX=B")
  #print(result[0])
  #print(result[1])
  #print(result[2])
  #print("Printing algorithm coefficients and number of nonzeros")
  #print(omp1betas[0]) 
  #print(omp1betas[1])
  print("Printing norms and errors")
  print(omp_result[0])
  print(omp_result[1])
  print(omp_result[2])

def test5(solver):
    """
    Repeating test4 above 100 times with solve_n_times function
    """
    a= solve_n_times(solver, 100, 0.40, 0.05, 200)
    #a= solve_n_times(solver, 100, 0.05, 0.05, 799)
    #a= solve_n_times(solver, 100, 0.05, 0.05, 400)
    #print("FINAL OUTPUT")
    #print(a[0])
    #print(a[1])
    print("Mean Fractional Error")
    print(a[0])
    print("Proportion Exceeding Threshold")
    print(a[1])

def test6():
    """
    This function is testing the function that creates the matrix in AX=B form first.
    """
    result = generateAXB(1.0,1.0, 200, force_intercept=ipForceInterceptNO, print_detail=ipPrintNO)
    print(result[0], result[1], result[2])
    print(result[0].shape)
    a = omp(result[0],result[2], [], tol=0.00001, demean=ipDemeanFIRST, scale=ipScaleYES, force_intercept=ipForceInterceptYES)
    return (a)
    
def main(showMsgs=False):
    """
    This is the final test function. Test the creation of the 5X5 grid with both sets of results. 
    In makePhaseDiagrams, demeaning is defaulted to ALL, and scale is defaulted to YES. Can easily change both of these.    
    """
    solver     = int(sys.argv[1])
    nproc      = int(sys.argv[2])
    param      = int(sys.argv[3])
    file_name1 = str(sys.argv[4])
    file_name2 = str(sys.argv[5])

    # Echo command line parameters
    if(showMsgs):
      print('Solver: ',sys.argv[1]);
      print('n Proc: ',sys.argv[2]);
      print('Params: ',sys.argv[3]);
      print('File 1: ',sys.argv[4]);
      print('File 2: ',sys.argv[5]);

    # Make Phase Diagrams
    ##Trying with non-OLS solution.###
    b = makePhaseDiagrams(solver, 100, 100, param, nproc=nproc, ols_solution = True)
    
    # Save results
    np.savetxt(file_name1,b[0])
    np.savetxt(file_name2,b[1])

# MAIN FUNCTION
if __name__ == "__main__":
  # Perform Tests
  #print("")
  #print("=== Test 2 ===")
  #print("")
  #test2()
  #print("")
  #print("=== Test 3 ===")
  #print("")
  #test3()
  #print("")
  #print("=== Test 4 ===")
  #print("")
  #test4()
  #print("=== Test 6 ===")
  #print("")
  #test6()

  #print("=== Test 5 ===")
  #print("")
  #test5(ipOmp)
  #test5(ipLp)
  #test5(ipLasso)
  #test5(ipL1regls)
  #test5(ipGlmnetLasso)
  #test5(ipLassoLARS)

  print("Computing Phase Diagrams...")
  print("")
  main()

