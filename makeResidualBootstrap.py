###The goal of the residual bootstrap is to run multiple sparse regressions, 
###and to take samples of the residuals each time.

# Imports
import sys
import numpy as np
import pandas as pd
from constants import *
from random import randint ##Imported for Random Numbers
from random import sample ##Select list of unique random integers
from runAnyMethod import generateAXB, compare, run_choice_method_once

###Solvers are present within runAnyMethod

##This is for parallel programming
from joblib import Parallel, delayed

###Step 1: The method (OMP, Lasso, LP, QP) is run, and the coefficients are obtained.
###For now, allow these parameters to be chosen manually. 
#AXB = generateAXB(0.8,0.2, 100)

###This function takes the coefficients from run_choice_method_once (first line of code), and then generates the bootstrap samples.
###Default to having no controls, but have ability to include controls
def generate_bootstrap_samples(solver, A, b, controls=[], num_bootstrap_samples = 1000, ols=False, treatment= []):
    ###Generate the coefficients from the method of choice. 
    if ols==True:
        ###Arguments are outcome, treatment, x_vars
        fitted_B = np.concatenate(ols(b,treatment,A)[1])
    else:
        coefficients = run_choice_method_once(solver, A, b, controls)
        ###Compute the fitted values from the coefficients from OMP/Lasso/LP/QP
        fitted_B = A.dot(coefficients)
    ###Convert b to a vector
    b = np.concatenate (b)
    ####Compute the residuals to use for the bootstrap####
    residuals = np.subtract(b, fitted_B)
    ###Demean the residuals
    centered_residuals = residuals - np.mean(residuals)
    ###Step 3: Now generate "n" bootstrap samples of the residuals.
    adjusted_y_values = np.zeros((len(centered_residuals), num_bootstrap_samples))
    for i in range (0,num_bootstrap_samples):
        #print(centered_residuals)
        bootstrap_sample = np.random.choice(centered_residuals, len(centered_residuals))
        ###sol contains the coefficients in a vector. Compute A*sol + bootstrap_sample = adjusted y-values.
        adjusted_y_values[:,i] = A.dot(coefficients)+bootstrap_sample 
        #adjusted_y_values[:,i] = fitted_B +bootstrap_sample 

    return adjusted_y_values
    
    
    
###Residual Bootstrap is generated using this function
###A[observations] and B[outcomes] are taken in as inputs using this function. CONTROLS are also taken in as well
###At this point, hold off on the forced inclusion of the controls, and the variable selection only occurring with the instruments
###Providing an option to import the y-values.
def ResidualBootstrap(solver, A, b, controls=[], num_bootstrap_samples=1000, nproc=8, tol=1e-5, ols=False, pregenerated_bootstrap=False):
    
    # Step 1: Solve AX=B with various methods and obtain the coefficients [called within Step 2]
    # Step 2: Generate a large number of bootstrap samples of residuals, and use those to generate new y_i values
    if (pregenerated_bootstrap==False):
        y_values = generate_bootstrap_samples (solver, A, b, controls, num_bootstrap_samples)
    ###The latter option calls generate_bootstrap_samples earlier on in the process. Might clean this up later. 
    else:
        y_values = b
    ###Step 3: Run the sparse regression method once on each set of y-values, and store the betas from each iteration in a matrix.
    bootstrap_coefficients = np.zeros((A.shape[1],num_bootstrap_samples))
    #bootstrap_coefficients_test = np.zeros((A.shape[1],5))
    ###Check in serial for 5 samples. It worked; results same for both; comment out. 
    #for j in range(0, 5):
    #bootstrap_coefficients_test[:,j] = run_choice_method_once(solver, A, y_values[:,j])
    if (ols==False):
        bootstrap_coefficients= Parallel(n_jobs = nproc, verbose = 150)(delayed(run_choice_method_once)(solver,A, y_values[:,i], controls, num_folds = 10, num_iterations = 1000, perform_cv = True, ols_solution = False, instrument_list= False, chern_ols = False) \
                                 for i in range(0,num_bootstrap_samples))
    else:
        bootstrap_coefficients= Parallel(n_jobs = nproc, verbose = 150)(delayed(ols)(y_values[:,i], A, controls, return_type=2) \
                                 for i in range(0,num_bootstrap_samples))
    ###Save results to text file to examine more closely
    #np.savetxt('bootstrap_result.txt', bootstrap_coefficients)
    ###This is a list.
    bootstrap_coefficients = np.asarray(bootstrap_coefficients)
    bootstrap_coefficients_df = pd.DataFrame(bootstrap_coefficients)
    #print(bootstrap_coefficients)
    
    ###Step 4: Summary Statistics of the Betas
    summ_stats = bootstrap_coefficients_df.describe()
    #np.savetxt('bootstrap_sum_stats.txt', summ_stats)
    ##1 is mean, 2 is sd
    key_stats = np.array(summ_stats.iloc[1:3,:])
    
    ###Step 5: Fitted Values. These can be useful for the Multi-Step Residual Bootstrap. 
    A_matrix= np.asmatrix(A)
    ###Matrix
    bootstrap_coefficients_matrix= np.asmatrix(bootstrap_coefficients)
    bootstrap_coefficients_matrix = bootstrap_coefficients_matrix.transpose()
    fitted_values = np.array(A_matrix* bootstrap_coefficients_matrix)
    #return key_stats
    return key_stats, fitted_values


def Belloni_Bootstraps(showMsgs=False):
    """
    This is the final test function. Test the creation of the 5X5 grid with both sets of results. 
    In makePhaseDiagrams, demeaning is defaulted to ALL, and scale is defaulted to YES. Can easily change both of these.    
    """
    num_samples = int(sys.argv[1])
    nproc = str(sys.argv[2])
    OMPTextFile= int(sys.argv[3])
    GlmLassoTextFile= int(sys.argv[4])

    # Echo command line parameters
    if(showMsgs):
      print('Number of Bootstrap Samples: ',sys.argv[1]);
      print('n proc: ',sys.argv[2]);
      print('File 1: ',sys.argv[3]);
      print('File 2: ',sys.argv[4]);

    # Run Residual Bootstrap with each method
    omp_result = ResidualBootstrap(0, A, b, controls, num_samples, nproc=nproc)
    glmlasso_result = ResidualBootstrap(4, A, b, controls, num_samples, nproc=nproc)

    # Save results
    np.savetxt(file_name1,omp_result)
    np.savetxt(file_name2,glmlasso_result)


def test5():
    """
    Executing a test of the residual bootstrap. Will test this as ongoing.
    """
    ##solver     = int(sys.argv[1])
    ###OMP, LP, QP Converge; Lasso does not converge
    ###Look at the conditions for each of the different approaches to converge
    A = np.array([[1, 0.9, 0.8, 0.2, -0.7, 1.3, 2.3, -0.5, 1.4, -0.2],[1, -0.1, 0.7, 0.3, -0.2, 0.9, 0.7, 1.3, 1.8, -2.5], [1,0.5,0.4,0.3,0.9,1.9, 2.3, -3.1, -2.9, -0.7], [1,0.2,0.9,0.1,0.7,2.3, 0.5, 2.5, -1.5, -2.9], [1,0.5,1,1.2,1.5,-0.1, 0.3, 0.7, 1.9, -0.6]])
    X = np.array( [[0.5], [0], [0], [0], [-0.2], [0], [0], [3.1], [0], [0]])
    B = A.dot(X)
    lasso_boot= ResidualBootstrap(2, A, B)
    key_stats = lasso_boot[0]
    fitted_values = lasso_boot[1]
    print('Key Stats Above')
    print(key_stats)
    print('Fitted Values')
    print(fitted_values)

# =============
# MAIN FUNCTION
# =============
if __name__ == "__main__":
  
  # Perform Tests
  print("")
  print("=== Test 5 ===")
  print("")
  test5()
  #print("")
  #print("=== Belloni Bootstraps ===")
  #print("")
  #Belloni_Bootstraps()
  