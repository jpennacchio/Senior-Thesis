# Imports
import sys
import numpy as np
import pandas as pd
from constants import *
from random import randint ##Imported for Random Numbers
from random import sample ##Select list of unique random integers
from runAnyMethod import generateAXB, compare, run_choice_method_once
from regression_formulas import ols
from Eminent_Domain_AdHoc import union, ad_hoc_single_selection, ad_hoc_double_selection, test_variables_to_drop, step_F

###Solvers are present within runAnyMethod
##This is for parallel programming
from joblib import Parallel, delayed

print("Started Multistep Residual Ad Hoc Double Selection Compiling")


###Note: The imports should be in array format. 
###Creating this function from scratch; no dependencies on other functions.
def Residual_Bootstrap_2SLS_DoubleSelection(x, y, d, z, num_bootstrap_samples = 1000, nproc=8, tol=1e-5):
    ###Run with controls and instruments, in Stage 1 on outcome
    ###This is the only line that changed.###
    res_double_selection = ad_hoc_double_selection (x, y, d, z, return_type = 1)
    controls = res_double_selection[0]
    print("Controls")
    print(controls)
    instruments = res_double_selection [1]
    print("Instruments")
    print(instruments)
    x_subset = x[:,controls]
    z_subset = z[:,instruments]
    print(x_subset.shape)
    print(z_subset.shape)
    xz_subset = np.concatenate((x_subset, z_subset), axis=1)
    ########STEP 1A:#########
    ###Solve d~z+x with OMP or Lasso
    ###Arguments are solver, A, b, controls
    ###This should be OLS now. No variable selection here.### Return Type = 2 corresponds to the coefficients###
    coefficients = ols(d, [], xz_subset, return_type = 2, dataframe=False, stage1 = True)
    ####STEP 1B:#########
    ###Compute the centered residuals.
    ###Compute the fitted values from the coefficients from OMP/Lasso/LP/QP
   ###Combine instruments and controls into a dataframe. The coefficients for the controls go first per omp and lasso setup. 
    d_hat = xz_subset.dot(coefficients)
    ###Convert d to a vector
    d = np.concatenate (d)
    ####Compute the residuals to use for the bootstrap####
    u_res = np.subtract(d, d_hat)
    ###Demean the residuals
    u_res_centered = u_res - np.mean(u_res)
    ####STEP 2A: Solve y~d_hat + x with OLS---Variable Selection isn't occurring here####
    ###Arguments are outcome, treatment, x_vars, return_type
    step2_result = ols(y, d_hat, x, return_type=3, dataframe = False)
    step2_coef = step2_result[2]
    d_hat_coef = step2_coef[0]
    step2_fitted_values = step2_result[1]
    ###STEP 2B: Determine revised residuals. 
    y_vec = y.flatten()
    step2_unaltered_residuals = y_vec - step2_fitted_values
    ###Add back d_hat*coefficient, subtract d*coefficient. [add coefficient*(d_hat-d)] 
    e_res = step2_unaltered_residuals + d_hat_coef * (d_hat - d) 
    #print(e_res.shape)
    ###Demean the residuals
    e_res_centered = e_res - np.mean(e_res)
    ####STEPS 1C and 2C; resample from u_res_centered and e_res_centered.####
    ###This will be the first of two loops. Only the second loop will be parallelized.####
    adjusted_d_values = np.zeros((len(u_res_centered), num_bootstrap_samples))
    adjusted_y_values = np.zeros((len(e_res_centered), num_bootstrap_samples))    
    
    ###Now generate "n" bootstrap samples of the residuals, and the corresponding adjusted d and y values.
    u_res_centered = u_res_centered[:,np.newaxis]
    e_res_centered = e_res_centered[:,np.newaxis]
    centered_residuals = np.concatenate((u_res_centered, e_res_centered), axis=1)
    for i in range (0,num_bootstrap_samples):
        ###Sample these together, then split up###
        ###Select the appropriate indexes.
        index = np.random.randint(low = 0, high = centered_residuals.shape[0], size = centered_residuals.shape[0])
        ###Then, pull the selected rows of residuals.###
        centered_sample = centered_residuals[index,:]
        u_sample = centered_sample[:,0]                     
        e_sample = centered_sample[:,1]
                                                    
        adjusted_d_values[:,i] = d_hat + u_sample               
        ### Adjusted y-values     
        temp_d = adjusted_d_values[:,i]
        temp_d = temp_d[:,np.newaxis]
        df = np.concatenate ((temp_d, x), axis=1)
        ###Here, di* is used instead of dhat, to ensure accurate results. 
        y_hat = df.dot(step2_coef)
        adjusted_y_values[:,i] = y_hat + e_sample    
    ###All sections below this have been modified###
    print("1") ###This section also changed###
    treatment_coefficients_2SLS= Parallel(n_jobs = nproc, verbose = 150)(delayed(ad_hoc_double_selection)(x, adjusted_y_values[:,i], adjusted_d_values[:,i], z) \
                                 for i in range(0,num_bootstrap_samples)) 
    print("2")
    treatment_coefficients_2SLS = np.asarray(treatment_coefficients_2SLS)
    treatment_coefficients_df = pd.DataFrame(treatment_coefficients_2SLS)
       
    ###Step 4: Summary Statistics of the Betas
    summ_stats = treatment_coefficients_df.describe()
    print(summ_stats)
    key_stats= np.array(summ_stats.iloc[1:3,:])
    return key_stats
    
    
    

    
    