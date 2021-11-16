import pandas as pd
import statsmodels.formula.api as sm
import numpy as np
import itertools as it

from regression_formulas import ols_formula, ols, two_stage_least_squares, standard_errors_2SLS

from Eminent_Domain_Data_Setup import data_prep_GDP, data_prep_FHFA, data_prep_CS, data_prep_NonMetro

from joblib import Parallel, delayed


###############################################################################################################
def union(a, b):
    """ return the union of two lists """
    return list(set(a) | set(b))
####Code to Perform Ad-Hoc Variable Selection.
###Focus on the first stage (so, use ols(outcome, treatment, x_vars)). For Stage 1, use "treatment, instruments, controls"

###Return type of 0 is coef/se; return type of 1 is list of instruments###
def ad_hoc_single_selection(x, y, d, z, full_results = True, return_type = 0):
    ##Create empty solution vector###
    sol = np.zeros(x.shape[1] + z.shape[1])
    ###Store all of the t-stats for the univariate case. If t-stat>=1 in magnitude, include in later processes.
    t_stats = np.zeros(z.shape[1])
    for i in range(z.shape[1]):
        t_stats[i] = ols(d, z[:,i] , x, return_type = 6, dataframe = False)
    instruments_to_consider = np.where(abs(t_stats)>1)[0]
    num_possible_combinations = (2** len(instruments_to_consider)) - 1
    f_stats = np.zeros(num_possible_combinations)
    counter = 0
    for k in range(1,len(instruments_to_consider)+1):
        for variables in it.combinations(instruments_to_consider, k):
            predictors = list(variables)
            f_stats[counter] = ols(d, [], np.concatenate((x, z[:,predictors]), axis=1), return_type = 5, dataframe = False)
            counter += 1
    ###Now find the minimum F-stat p-value here.###
    selected_combination = np.argmin(f_stats)
    ###Now go back and find which predictors were included###
    counter = 0
    for k in range(1,len(instruments_to_consider)+1):
        for variables in it.combinations(instruments_to_consider, k):
            if(counter == selected_combination):
                predictors = list(variables)
                final_model = ols(d, [], np.concatenate((x, z[:,predictors]), axis=1), return_type = 4, dataframe = False)
                if(full_results == True):
                    print(final_model)
                k = len(instruments_to_consider) + 1
                break
            counter += 1
    ###Return the coefficient vector solution from ad-hoc single selection.###
    num_controls = x.shape[1]
    support = [q + num_controls for q in predictors]
    print(support)
    control_indexes = np.asarray(range(num_controls))
    support = np.concatenate((control_indexes, support))
    A = np.concatenate((x,z), axis=1)     
    sol[support] = np.linalg.lstsq(A[:,support], d)[0].ravel()
    ###Now I have the Step 1 result. Now obtain fitted values and proceed to Step 2###
    fitted_values = A.dot(sol)
    ###Step 2###
    if(full_results == True):
        step2_model = ols(y, fitted_values, x, return_type = 4, dataframe = False)
        print(step2_model)
    ### Returning the treatment coefficient and standard error###Note: This isn't the correct standard error.###
    treatment_coef = ols(y, fitted_values, x, return_type = 0, dataframe = False)
    if(return_type==0):
        return treatment_coef
    else:
        return predictors

def test_variables_to_drop(outcome, variables, var_to_drop, outcome_is_treatment = True):
    ###Drop the variable. ###
    ###array, column index, 0 for row (1 for column)
    temp_variables = np.delete(variables, var_to_drop, 1) 
    if(outcome_is_treatment == True):
          F = ols(outcome, [], temp_variables, return_type = 5, dataframe = False)
    else:
          F = ols(outcome, [], temp_variables, return_type = 5, dataframe = False)
    return F

###Currently designed to work with smallest F p-value [5] in sequence, but can modify to go by largest overall F-statistic [7].###
###AIC [8] could also be an option.###
def step_F(outcome, variables, outcome_is_treatment = True, nproc = 4):
    num_variables = variables.shape[1]
    num_variables_remaining = variables.shape[1] ###This one will change###
    min_F_pvalue = 1.00
    variable_ids = list(range(num_variables))
    ###This loops through all variables in the original dataframe.
    for j in range (num_variables):
        F = np.zeros(num_variables_remaining)
        ###Escape condition###
        if(num_variables_remaining == 1):
            if(outcome_is_treatment == True):
                    ###Return Type 4 is summary
                    res = ols(outcome, [], variables, return_type = 4, dataframe = False)
            else:
                    res = ols(outcome,[], variables, return_type = 4, dataframe = False)            
            return res  
        ###This loops through the remaining variables and determines which one to dump.###
            #F = Parallel(n_jobs = nproc, verbose = 150)(delayed(test_variables_to_drop)(outcome, variables, i, outcome_is_treatment) \
                                 #for i in range(num_variables_remaining))
            #F = np.asarray(F)       
            #F_min = max(F)
        for i in range(num_variables_remaining):
            F[i] = test_variables_to_drop (outcome, variables, i, outcome_is_treatment)
        ###Since going off of p-value, need to select the minimum here.###
        F_min = min(F)

        if (F_min < min_F_pvalue and num_variables_remaining >1):
            ###Make the current F-stat the new maximum.###
            min_F_pvalue = F_min
            var_to_drop = np.argmin(F)
            variables = np.delete(variables, var_to_drop, 1)
            ###Delete variable id from list###
            del variable_ids[var_to_drop]
            num_variables_remaining = variables.shape[1] ###Now this should be smaller.
        else: ##If only one variable remaining, must fit this here###
            if(outcome_is_treatment == True):
                res = ols(outcome, [], variables, return_type = 4, dataframe = False)
            else:
                res = ols(outcome, [], variables, return_type = 4, dataframe = False)   
            #variables_to_include = list(variables)
            #print(variables_to_include)            
            return res, variable_ids
    return 0

def ad_hoc_double_selection(x, y, d, z, return_type = 0):
    num_total_controls = x.shape[1]
    xz = np.concatenate([x,z], axis=1)
    #print("STARTED STEPWISE F PROCEDURE")
    step_f_test_1 = step_F (y, xz, outcome_is_treatment = False)
    step_f_test_2 = step_F (d, xz, outcome_is_treatment = True)
    variables_1 = step_f_test_1[1]
    variables_2 = step_f_test_2[1]
    controls_instruments = union(variables_1, variables_2)
    subsetted_controls = [i for i in controls_instruments if i < num_total_controls]
    subsetted_instruments = [i - num_total_controls for i in controls_instruments if i >= num_total_controls]
    print(subsetted_controls)
    print(subsetted_instruments)
    if(return_type ==0):
    ###Now run 2SLS with the desired attributes.####Note: This WILL have the correct standard errors.###
        res = two_stage_least_squares(y, d, x[:,subsetted_controls], z[:,subsetted_instruments], dataframe=False)
        return res
    else:
        return subsetted_controls, subsetted_instruments
    

def test(data):
    x, y, d, z = np.array(data[0]), np.array(data[1]), np.array(data[2]), data[3]    
    ####1st stage regressions
    #z = z.iloc[:,[0 ,1 ,3 ,7 ,8 ,9 ,15 ,18 ,24 ,31]]
    z = np.array(z)
    #gdp_res = ad_hoc_single_selection(x, y, d, z)
    gdp_res2 = ad_hoc_double_selection(x, y, d, z)  
    ###This is where the appropriate variables should be selected.###
    #print(gdp_res)
    print(gdp_res2)

print("GDP")
data1= data_prep_GDP()
test(data1)
print("NonMetro")
data2= data_prep_NonMetro()
test(data2)
print("CS")
data3= data_prep_CS()
test(data3)
print("FHFA")
data4= data_prep_FHFA()
test(data4)




