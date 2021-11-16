###This file contains the functions to create the OLS regression and the 2SLS regression. 
###There may be some redundancy compared to the OMP Stepwise Regression Process

###Imports
import pandas as pd
import statsmodels.formula.api as sm
import numpy as np
from data_preprocessing import preprocess_arrays, transform_data_with_controls
from runAnyMethod import run_choice_method_once

###############################################################################################################
###CONSTANTS
# Arguments for demean are "ALL," "FIRST", and "NONE."
ipNoINTERCEPT  = 0
ipYesINTERCEPT =  1
################################################################################################################

####Found code on stack overflow to try to replicate R-style formulas for linear regression
###https://stackoverflow.com/questions/22388498/statsmodels-linear-regression-patsy-formula-to-include-all-predictors-in-model
def ols_formula(df, dependent_var, intercept = ipNoINTERCEPT, *excluded_cols):
    '''
    Generates the R style formula for statsmodels (patsy) given
    the dataframe, dependent variable and optional excluded columns
    as strings
    
    Args:
    df: dataframe
    dependent variable: y-variable in the regression
    intercept: Integer flag to determine whether intercept is included in regression. Belloni paper 
    does not include intercept.
           It can assume the following values
           - ipNoINTERCEPT: Don't include intercept in regression formula
           - ipYesINTERCEPT: Include intercept in regression formula
    *excluded_cols: choose not to include some of the columns in the regression

    Returns:
    The proper OLS formula with all variables and with/without an intercept is returned, to use for the regressions

    '''
    df_columns = list(df.columns.values)
    df_columns.remove(dependent_var)
    for col in excluded_cols:
        df_columns.remove(col)
    if (intercept == ipNoINTERCEPT):
        return dependent_var + ' ~ ' + ' + '.join(df_columns)+ ' -1'
    else:
        return dependent_var + ' ~ ' + ' + '.join(df_columns)
#############################################################################################    
    
    
###Runs Ordinary Least Squares Regression y~d+x(or xz)
def ols(outcome, treatment, x_vars, return_type = 0, dataframe=True, stage1 = False): 
    if (dataframe==False):
        temp = preprocess_arrays(outcome, treatment, x_vars)
        outcome = temp[0]
        treatment = temp[1]
        x_vars = temp[2]
        ###automatically change here since the transformation gives "d" a "y"-label. 
        stage1 = False
    if(len(x_vars) !=0 and len(treatment) !=0):
        ols_frame = pd.concat([outcome, treatment, x_vars], axis=1)
    elif(len(x_vars) !=0 and len(treatment) ==0):
        ols_frame = pd.concat([outcome, x_vars], axis=1)
    else:
        ols_frame = pd.concat([outcome, treatment], axis=1)
    #ols1 = ols_formula(ols_frame, 'y')
    ###This will allow OLS formula to be used when treatment is the dependent variable###
    if(stage1 == False):
        ols1 = ols_formula(ols_frame, 'y')
    else:
        ols1 = ols_formula(ols_frame, 'd')
    #print(ols1)
    ###Note: stage1 is NOT the data source; the data source is ols_frame!!!!
    model = sm.ols(formula = ols1, data= ols_frame)
    ols_results = model.fit() ###To work around SVD issues###
    summary = ols_results.summary()
    table = [ols_results.params[0], ols_results.bse[0]]
    table = np.asarray(table)
    fitted_values = ols_results.predict()
    coefficients = ols_results.params
    coefficients = np.asarray(coefficients)
    f_pvalue = ols_results.f_pvalue
    f_value = ols_results.fvalue
    model_aic = ols_results.aic
    if return_type==0:
        return table
    elif (return_type==1):
        return fitted_values
    elif (return_type==2):
        return coefficients
    elif (return_type==4):
        return summary
    elif (return_type==5):
        return f_pvalue
    elif (return_type==6):
        ###Only want the first coefficient, so this is good for univariate. ###
        t_stat = ols_results.params[0]/ols_results.bse[0]
        return t_stat
    elif (return_type ==7):
        return f_value
    elif (return_type ==8):
        return model_aic
    elif (return_type ==9):
        return ols_results.params[0]
    elif (return_type ==10):
        return ols_results.resid
    else:
        return table, fitted_values, coefficients
 
''
###Computes the revised 2SLS standard errors
def standard_errors_2SLS(treatment, dhat_vec, residuals, coef, exogeneous):
    ###Step 1: Compute sigma hat squared###
    ###Compute ui###
    ###coef is treatment coefficient
    dhat = pd.DataFrame(dhat_vec)
    dhat.columns = ['dhat']
    difference_vec = dhat_vec-treatment
    alter_residuals = [x * coef  for x in difference_vec]
    altered_residuals = residuals + alter_residuals 
    var_reg = sum(np.square(altered_residuals))/ len(altered_residuals)
    ###Step 2: Compute SSTj####
    sst_step1 = [x - np.mean(dhat_vec) for x in dhat_vec]
    sst_step2 = [x**2 for x in sst_step1]
    sst = sum(sst_step2)
    ###Step 3: Compute R^2
    ###Exogeneous has x's in dataframe format
    ###Add intercept column
    new_reg = pd.concat([dhat, exogeneous], axis=1)
    q = ols_formula(new_reg, 'dhat', intercept = ipYesINTERCEPT)
    ###Making sure formulas are correct####
    #print(q)
    model = sm.ols(formula = q, data= new_reg)
    results = model.fit(method= 'qr') ###Another fit method for SVD corner case###
    ###Obtain r-squared.###
    rsquared = results.rsquared
    denominator = sst * (1-rsquared)
    ##Now compute revised standard error.###
    revised_var = var_reg/denominator
    revised_se = np.sqrt(revised_var)
    
    return revised_se

###Runs the 2SLS process###
###Subset to include only the necessary instruments from the csv before using this function.
def two_stage_least_squares(outcome, treatment, exogeneous, instruments, dataframe=True):
    if (dataframe==False):
        temp = preprocess_arrays(outcome, treatment, exogeneous, instruments)
        outcome = temp[0]
        treatment = temp[1]
        exogeneous = temp[2]
        instruments = temp[3]
    ####Stage 1:
    ###Combine the dataframes
    stage1 = pd.concat([treatment, instruments, exogeneous], axis=1)
    q = ols_formula(stage1, 'd')
    ###Making sure formulas are correct####
    #print(q)
    
    model = sm.ols(formula = q, data= stage1)
    #results = model.fit(method = 'qr') ###this might help SVD issues

    results = model.fit() ###this might help SVD issues
    ###Print the parameter attribute. These work. Now export predictions.
    #print("Parameters")
    #print(results.params)
    print(results.summary())
    #print('Standard errors: ', results.bse)

    dhat_vec = results.predict()
    ##Convert to dataframe
    dhat = pd.DataFrame(dhat_vec)
    dhat.columns = ['dhat']
    ##Fitted Values
    #print('Predicted values: ', dhat)
    
    ##The fitted values are now stored a python dataframe; use for Step 2 of 2SLS
    stage2 = pd.concat([outcome, dhat, exogeneous], axis=1)
    qq = ols_formula (stage2, 'y')

    model_2 = sm.ols(formula =  qq, data = stage2)
    results2 = model_2.fit() ###this might help SVD issues
    #results2 = model_2.fit(method = 'qr') ###this might help SVD issues

    treatment_df = treatment
    treatment = treatment.as_matrix()
    treatment=treatment.ravel()
    residuals = results2.resid
    ivwrongtreatment_se = results2.bse[0]
    coef = results2.params[0]
    ###New Standard Error Calculation
    revised_se = standard_errors_2SLS(treatment, dhat_vec, residuals, coef, exogeneous)
    ###Run OLS for comparison.###
    ols_comp = pd.concat([outcome, treatment_df, exogeneous], axis=1)
    ols2 = ols_formula (ols_comp, 'y')
    ols_model = sm.ols(formula =  ols2, data = ols_comp)
    ols_results = ols_model.fit()
    ols_se = ols_results.bse[0]

    #table = [results2.params[0], ols_se, revised_se, ivwrongtreatment_se]
    table = [results2.params[0], revised_se]

    return table;

###Input the post-controls data in here (residual-modified)
def post_lasso_2SLS(outcome, treatment, instruments, dataframe=True):
    if (dataframe==False):
        temp = preprocess_arrays(outcome, treatment, [], instruments)
        outcome = temp[0]
        treatment = temp[1]
        instruments = temp[3]
    ####Stage 1:
    ###Combine the dataframes
    stage1 = pd.concat([treatment, instruments], axis=1)
    q = ols_formula(stage1, 'd')
    model = sm.ols(formula = q, data= stage1)
    results = model.fit(method = 'qr')
    dhat_vec = results.predict()
    ##Convert to dataframe
    dhat = pd.DataFrame(dhat_vec)
    dhat.columns = ['dhat']
    ##The fitted values are now stored a python dataframe; use for Step 2 of 2SLS
    stage2 = pd.concat([outcome, dhat], axis=1)
    qq = ols_formula (stage2, 'y')
    model_2 = sm.ols(formula =  qq, data = stage2)
    results2 = model_2.fit(method = 'qr')
    treatment = treatment.as_matrix()
    treatment=treatment.ravel()
    residuals = results2.resid
    old_se = results2.bse[0]  
    coef = results2.params[0]
    revised_se = standard_errors_2SLS (treatment, dhat_vec, residuals, coef, old_se)
    ###Now return the appropriate results
    table = [coef, old_se, revised_se]
    return table;

def forced_controls_2SLS(solver, x_array, y_array, d_array, z_array, num_folds = 10, num_iterations = 1000, perform_cv = True, ols_solution = True, instrument_list = True, chern_ols = False):
    ###This is just a double-check, since if this isn't true there will be major problems.####
    if (solver ==0):
        chern_ols = False
        ols_solution = True
    ###OLS coefficients for Stage 1. Must use solver [0], 2, 3, 5, or 6 (not original lp, or l1regls)####
    ###Data is transformed within the functions for the specific solvers called.####
    res = run_choice_method_once(solver, z_array, d_array, x_array, num_folds, num_iterations, perform_cv, ols_solution, instrument_list, chern_ols)
    ols_stage1_coef = np.transpose(np.matrix(res[0]))
    if(instrument_list == True):
        instruments = res[1]
    if(chern_ols ==False):
        A = np.matrix(np.concatenate((x_array, z_array), axis = 1))
        d_hat = np.asarray(A * ols_stage1_coef)
        A_2 = np.concatenate((d_hat, x_array), axis=1)
        all_coef= np.linalg.lstsq(A_2, y_array)
        stage2_coef = np.linalg.lstsq(A_2,y_array)[0] 
        treatment_coef = stage2_coef[0]
        ###Subset out all of the controls from the list. 
        if(instrument_list == True):
            num_controls = x_array.shape[1]
            instruments = [i for i in instruments if i>= num_controls]
            instruments= [q - num_controls for q in instruments]
    ##########################################################################################################
    ###This is the chern_ols = TRUE section####
    else:
        ###Only need the transoformed data if using this method.
         ###Transform data to reflect inclusion of the controls
        modified_variables = transform_data_with_controls(x_array, y_array, d_array, z_array) 
        d_norm = modified_variables[0]
        z_norm = np.matrix(modified_variables[1])
        d_mod = np.matrix(modified_variables[2])
        z_mod = np.matrix(modified_variables[3])
        y_mod = np.matrix(modified_variables[4]) 
       ###The Chernozukov approach is more complicated than originally implemented###
        ###Step 3 is the result from run_choice_method_once.        
        d_hat = np.matrix(z_mod * ols_stage1_coef)
        z_subset = z_mod[:,instruments]
        yz_coef = np.linalg.lstsq(z_subset,np.asarray(y_mod))[0] 
        ###System is Mz [subset] * coef = My.
        ### [312,2] * [2,1] --> [312,1], which are the correct dimensions.###
        y_hat = np.matrix(z_subset * yz_coef)
        md_transpose = np.transpose(d_mod)
        front = np.linalg.inv(md_transpose * d_hat)
        back = md_transpose * y_hat
        ##This is the final step, and mirrors the step taken in the Matlab Code
        treatment_coef = front * back
    #################################################################################################################
    if(instrument_list == True):
        return treatment_coef, instruments
    else:
        return treatment_coef


