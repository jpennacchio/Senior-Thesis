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

###Note: Can modify this function to take in arguments in terms of delta and rho. 
def test_bootstrap(solver, nobs=100, nvars=50, num_nonzeros=15, num_bootstraps = 1000, nproc=8):
    ###Step 0: Generate observations matrix A
    A = np.random.rand(nobs,nvars)
    ####Step 1: Randomly select n non-zero components of a coefficient vector beta.
    ##Now generate the x's. Generate from N(6,1). These will be the means
    X_means = np.random.normal(6, 1, (nvars,1))
    ###Generate the sd's as well; use N(0.6,0.1) for the sd's. Want coefficients close together, so try narrow band. 
    X_sds = np.random.normal(0.6, 0.1, (nvars,1))
    
    ##Now, based on the number of nonzeroes specified early, pick which x's to set to zero. 
    num_zeros = nvars - num_nonzeros;
    
    ####Now randomly sample from normal distribution with corresponding means and variances.
    random_coefficients = np.random.normal(X_means, X_sds, (nvars, num_bootstraps))
    
    ###Double loop through the random coefficients. 
    ###These are the different coefficient samples
    for j in range(random_coefficients.shape[1]):
        ###These are individual coefficients within the samples. 
        ##Now pick where the zeros come from####
        s = sample(range(0,nvars), num_zeros) 
        ###Inner Loop: Change coefficient values
        for i in range(0, len(s)):
            b = s[i]
            random_coefficients[b,j] = 0
    
    #test_result = np.concatenate((X_means, X_sds, random_coefficients), axis=1)
    #np.savetxt('check_coefficients.txt', test_result)
    #######################################################################################################################
    ###Generate the y-values corresponding to these coefficients and observations. 
    gen_realizations = A.dot(random_coefficients)
    
    
    ###Step 3: Run the sparse regression method once on each set of y-values, and store the betas from each iteration in a matrix.
    bootstrap_coefficients = np.zeros((A.shape[1],num_bootstraps))
    bootstrap_coefficients= Parallel(n_jobs = nproc, verbose = 150)(delayed(run_choice_method_once)(solver,A, gen_realizations[:,i]) \
                                 for i in range(0,num_bootstraps))
    bootstrap_coefficients = pd.DataFrame(np.asarray(bootstrap_coefficients))
    
    ###Step 4: Summary Statistics of the Betas
    summ_stats = bootstrap_coefficients.describe(percentiles = [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99])
    #np.savetxt('bootstrap_sum_stats.txt', summ_stats)
    ###Want to compare the means and standard deviations of the bootstrap coefficients to the pre-generated ones.
    mean_boot = np.array(summ_stats.iloc[1,:])
    sd_boot = np.array(summ_stats.iloc[2,:])
    ###Transpose to have coefficients as the rows and statistics as the columns### Exclude last column [max]###
    percentiles_boot = np.transpose(np.array(summ_stats.iloc[4:11, :]))

    ###Note: There is one coef. column for each bootstrap sample; must transpose so the columns are the different coefficients.
    random_coefficients = pd.DataFrame(np.transpose(random_coefficients))
    ###Now compare the results from the pre-generated coefficients
    random_stats = random_coefficients.describe(percentiles = [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99])
    mean_random = np.array(random_stats.iloc[1,:])
    sd_random = np.array(random_stats.iloc[2,:])
    ###Transpose to have coefficients as the rows and statistics as the columns###
    percentiles_random = np.transpose(np.array(random_stats.iloc[4:11, :]))

    ###Convert from vectors to arrays to allow for easier comparison
    mean_boot = mean_boot[:,np.newaxis]
    sd_boot = sd_boot[:,np.newaxis]
    mean_random = mean_random[:,np.newaxis]
    sd_random = sd_random[:,np.newaxis]

    ###Store the results in a dataframe to allow for easy side-by-side comparison. 
    result_frame = np.concatenate((mean_random, mean_boot, sd_random, sd_boot), axis=1)
    percentiles_frame  = np.concatenate((percentiles_random, percentiles_boot), axis=1)
    ###Rearrange percentiles_frame to be alternating
    index_list = [0,7,1,8,2,9,3,10,4,11,5,12,6,13]
    percentiles_frame = percentiles_frame[:,index_list]
    ###Save the dataframes
    np.savetxt('noboot_boot_mean_sd_sidebyside.txt', result_frame)
    np.savetxt('noboot_boot_percentiles.txt', percentiles_frame)
    return result_frame, percentiles_frame


# =============
# MAIN FUNCTION
# =============
if __name__ == "__main__":
  
  # Perform Tests
  #print("")
  #print("=== Test 5 ===")
  #print("")
  ###also vary zero/nonzero selections since not always full recovery. Testing from recovery region should match perfectly. 
  ###Test on OMP First
  res = test_bootstrap(0, num_bootstraps=1000)
  
    