###Imports
import pandas as pd
import statsmodels.formula.api as sm
import numpy as np
from data_preprocessing import preprocess_arrays, transform_data_with_controls

###############################################################################################################
###CONSTANTS
# Arguments for demean are "ALL," "FIRST", and "NONE."
ipNoINTERCEPT  = 0
ipYesINTERCEPT =  1
################################################################################################################


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
    ols_results = model.fit(method = 'qr') ###To work around SVD issues###
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
