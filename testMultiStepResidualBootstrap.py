# Imports
import sys
import numpy as np
import pandas as pd
from constants import *
from random import randint ##Imported for Random Numbers
from random import sample ##Select list of unique random integers
from runAnyMethod import generateAXB, compare, run_choice_method_once
from makeResidualBootstrap import generate_bootstrap_samples, ResidualBootstrap
from regression_formulas import ols

###Solvers are present within runAnyMethod
##This is for parallel programming
from joblib import Parallel, delayed

###Note: The imports should be in array format. 
###Creating this function from scratch; no dependencies on other functions.
  
def test_Residual_Bootstrap_2SLS(solver, nobs=200,  num_controls=50, num_instruments=50, num_nonzero_instruments=15, num_bootstraps = 1000, nproc=8, tol=1e-5):
    ###Randomly generate control, instrument, treatment, and outcome matrices. 
    ###1. Generate controls matrix x
    Controls = np.random.rand(nobs,num_controls)
    ###Generate following similar parameters to before. ######
    Step1_Control_Means = np.random.normal(6, 1, (num_controls,1))
    Step1_Control_Sds = np.random.normal(0.6, 0.1, (num_controls,1))
    random_Step1_Control_coefficients = np.random.normal(Step1_Control_Means, Step1_Control_Sds, (num_controls, num_bootstraps))
    ##2. Generate instrument
    Instruments = np.random.rand(nobs,num_instruments)
    ###Generate following similar parameters to before. ######
    Step1_Inst_Means = np.random.normal(6, 1, (num_instruments,1))
    Step1_Inst_Sds = np.random.normal(0.6, 0.1, (num_instruments,1))
    random_Step1_inst_coefficients = np.random.normal(Step1_Inst_Means, Step1_Inst_Sds, (num_instruments, num_bootstraps))
    #Now, based on the number of nonzeroes specified early, pick which instrument coefficients to set to zero. 
    num_zero_inst = num_instruments - num_nonzero_instruments;
    ###Double loop through the random coefficients. 
    ###These are the different coefficient samples
    for j in range(random_Step1_inst_coefficients.shape[1]):
        ###These are individual coefficients within the samples. 
        ##Now pick where the zeros come from####
        s = sample(range(0,num_instruments), num_zero_inst) 
        ###Inner Loop: Change coefficient values
        for i in range(0, len(s)):
            b = s[i]
            random_Step1_inst_coefficients[b,j] = 0           
    ###Now, compute the treatment values. Won't be in [0,1] here, but the testing is still valid###
    ###Must combine both the instruments and controls dataframes
    A = np.concatenate((Controls, Instruments), axis=1)
    random_Step1_coefficients = np.concatenate((random_Step1_Control_coefficients, random_Step1_inst_coefficients), axis=0)
    Treatment_Values = A.dot(random_Step1_coefficients)
    ####Now generate beta1 and beta2, which will be the factors by which the Step 2 coefficients are related to Step 1.
    ###Can generate at once, with beta1 being the first coefficient
    ####Trying diffferent distribution here for effect. ###
    ###Only have one treatment coefficient: use that first.
    Step2_coefficient_Means = np.random.normal(1.2, 0.2, (num_controls + 1,1))
    Step2_coefficient_Sds = np.random.normal(0.2, 0.05, (num_controls + 1,1))
    random_Step2_coefficients = np.random.normal(Step2_coefficient_Means, Step2_coefficient_Sds, (num_controls + 1, num_bootstraps))
    ###y = B1gamma1 * treatmenthat + (B1gamma2 + B2) controls + (B1u+e). But don't need to do that here. 
    ####Generate the appropriate outcome values. dimensions are nobs X num_bootstraps.####
    Outcome_Values = np.zeros((nobs, num_bootstraps))
    for k in range(num_bootstraps):
        Selected_Treatment = Treatment_Values[:,k]
        Selected_Treatment = Selected_Treatment[:,np.newaxis]
        AA = np.concatenate((Selected_Treatment, Controls), axis=1)
        Outcome_Values[:,k] = AA.dot(random_Step2_coefficients[:,k])
    #####Now follow through on the testing procedure. #######
    ###We already have Outcome and Treatment Values that vary based on the coefficients [replace "adjusted d" and "adjusted y" values]
    ####Now use these bootstrapped values to complete the procedure
    ###These two sections will be performed in parallel
    
    ###Arguments are solver, A, b, controls[force]: d~x+z
    print(Instruments.shape)
    print(Treatment_Values[:,0].shape)
    print(Controls.shape)
    step1_bootstrap_coefficients= Parallel(n_jobs = nproc, verbose = 150)(delayed(run_choice_method_once)(solver,Instruments, Treatment_Values[:,i], Controls) \
                                 for i in range(0,num_bootstraps))
    ###Arguments are outcome, treatment, x_vars, return_type [2=controls]: y~dhat + x
    ###Convert to Pandas DataFrames.
    step2_bootstrap_coefficients= Parallel(n_jobs = nproc, verbose = 150)(delayed(ols)(Outcome_Values[:,i], Treatment_Values[:,i], Controls, return_type=2, dataframe = False) \
                                 for i in range(0,num_bootstraps))    
    
    ###Now need to compile some summary statistics. 
    print("Computed both bootstrap loops")
    step1_bootstrap_coefficients = pd.DataFrame(np.asarray(step1_bootstrap_coefficients))
    step2_bootstrap_coefficients = pd.DataFrame(np.asarray(step2_bootstrap_coefficients))
    ###Step 4: Summary Statistics of the Betas
    step1_summ_stats = step1_bootstrap_coefficients.describe(percentiles = [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99])
    step2_summ_stats = step2_bootstrap_coefficients.describe(percentiles = [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99])

    ###Want to compare the means and standard deviations of the bootstrap coefficients to the pre-generated ones.
    step1_mean_boot = np.array(step1_summ_stats.iloc[1,:])
    step1_sd_boot = np.array(step1_summ_stats.iloc[2,:])
    step1_percentiles_boot = np.transpose(np.array(step1_summ_stats.iloc[4:11, :]))
    step2_mean_boot = np.array(step2_summ_stats.iloc[1,:])
    step2_sd_boot = np.array(step2_summ_stats.iloc[2,:])
    step2_percentiles_boot = np.transpose(np.array(step2_summ_stats.iloc[4:11, :]))

    ###Note: There is one coef. column for each bootstrap sample; must transpose so the columns are the different coefficients.
    random_Step1_coefficients = pd.DataFrame(np.transpose(random_Step1_coefficients))
    ###Now compare the results from the pre-generated coefficients
    random_Step1_stats = random_Step1_coefficients.describe(percentiles = [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99])
    mean_Step1_random = np.array(random_Step1_stats.iloc[1,:])
    sd_Step1_random = np.array(random_Step1_stats.iloc[2,:])
    percentiles_Step1_random = np.transpose(np.array(random_Step1_stats.iloc[4:11, :]))
    
    ###Note: There is one coef. column for each bootstrap sample; must transpose so the columns are the different coefficients.
    random_Step2_coefficients = pd.DataFrame(np.transpose(random_Step2_coefficients))
    ###Now compare the results from the pre-generated coefficients
    random_Step2_stats = random_Step2_coefficients.describe(percentiles = [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99])
    mean_Step2_random = np.array(random_Step2_stats.iloc[1,:])
    sd_Step2_random = np.array(random_Step2_stats.iloc[2,:])
    percentiles_Step2_random = np.transpose(np.array(random_Step2_stats.iloc[4:11, :]))

    ###Convert from vectors to arrays to allow for easier comparison
    step1_mean_boot, step2_mean_boot = step1_mean_boot[:,np.newaxis], step2_mean_boot[:,np.newaxis]
    step1_sd_boot, step2_sd_boot = step1_sd_boot[:,np.newaxis], step2_sd_boot[:,np.newaxis]
    mean_Step1_random, mean_Step2_random = mean_Step1_random[:,np.newaxis], mean_Step2_random[:,np.newaxis]
    sd_Step1_random, sd_Step2_random = sd_Step1_random[:,np.newaxis], sd_Step2_random[:,np.newaxis]

    ###Store the results in a dataframe to allow for easy side-by-side comparison.###
    result_frame1 = np.concatenate((mean_Step1_random, step1_mean_boot, sd_Step1_random, step1_sd_boot), axis=1)
    percentiles_frame1  = np.concatenate((percentiles_Step1_random, step1_percentiles_boot), axis=1)
    result_frame2 = np.concatenate((mean_Step2_random, step2_mean_boot, sd_Step2_random, step2_sd_boot), axis=1)
    percentiles_frame2  = np.concatenate((percentiles_Step2_random, step2_percentiles_boot), axis=1)
    ###Rearrange percentiles_frame to be alternating
    index_list = [0,7,1,8,2,9,3,10,4,11,5,12,6,13]
    percentiles_frame1, percentiles_frame2 = percentiles_frame1[:,index_list], percentiles_frame2[:,index_list]

    ###Save the dataframes
    np.savetxt('6_multistep_1_mean_sd_sidebyside.txt', result_frame1)
    np.savetxt('6_multistep_1_percentiles.txt', percentiles_frame1)
    np.savetxt('6_multistep_2_mean_sd_sidebyside.txt', result_frame2)
    np.savetxt('6_multistep_2_percentiles.txt', percentiles_frame2)
    return result_frame1, percentiles_frame1, result_frame2, percentiles_frame2


# =============
# MAIN FUNCTION
# =============
if __name__ == "__main__":
  
  # Perform Tests
  ###also vary zero/nonzero selections since not always full recovery. Testing from recovery region should match perfectly. 
  res = test_Residual_Bootstrap_2SLS(6)
    