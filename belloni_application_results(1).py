# Imports
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
from constants import *
from random import randint ##Imported for Random Numbers
from random import sample ##Select list of unique random integers
from data_preprocessing import transform_data_with_controls
###2SLS Infrastructure
from regression_formulas import forced_controls_2SLS
###Data Setup
from Eminent_Domain_Data_Setup import data_prep_GDP, data_prep_FHFA, data_prep_CS, data_prep_NonMetro
###############################################################################################################

###Step 1: IMPORT DATA
def GDP():
    data= data_prep_GDP()
    x, y, d, z = data[0], data[1], data[2], data[3]
    x_array = np.array(x)
    d_array = np.array(d)
    ###Convert to np.float64 so that the glmnet Lasso can run.
    d_array = d_array.astype(np.float64)
    y_array = np.array(y)
    z_array = np.array(z)


    ####Step 3: Run 2SLS with the controls

    ###Solvers: 2= Lasso, 4 = Glmnet, 5 = LassoLars, 6= OMP w/ Chernozukov controls method
    print('GDP')
    #res0 = forced_controls_2SLS(0, x_array, y_array, d_array, z_array, num_folds = 10, num_iterations = 1000)
    #print(res0)
    #res2 = forced_controls_2SLS(2, x_array, y_array, d_array, z_array, num_folds = 10, num_iterations = 1000, perform_cv = False)
    #print(res2)
    res4 = forced_controls_2SLS(7, x_array, y_array, d_array, z_array, num_folds = 10, num_iterations = 1000)
    print(res4)
    #res5 = forced_controls_2SLS(5, x_array, y_array, d_array, z_array, num_folds = 10, num_iterations = 1000, perform_cv = False)
    #print(res5)
    #res6 = forced_controls_2SLS(6, x_array, y_array, d_array, z_array, num_folds = 10, num_iterations = 1000, chern_ols = True)
    #print(res6)
    #print('GDP')
    #print(res0)
    #print(res2)
    #print(res4)
    #print(res5)
    #print(res6)
    return 0
def FHFA():
    data= data_prep_FHFA()
    x, y, d, z = data[0], data[1], data[2], data[3]
    x_array = np.array(x)
    d_array = np.array(d)
    ###Convert to np.float64 so that the glmnet Lasso can run.
    d_array = d_array.astype(np.float64)
    y_array = np.array(y)
    z_array = np.array(z)

    print('FHFA')
    #res10 = forced_controls_2SLS(0, x_array, y_array, d_array, z_array, num_folds = 10, num_iterations = 1000)
    #print(res10)
    #res12 = forced_controls_2SLS(2, x_array, y_array, d_array, z_array, num_folds = 10, num_iterations = 1000, perform_cv = False)
    #print(res12)
    res14 = forced_controls_2SLS(7, x_array, y_array, d_array, z_array, num_folds = 10, num_iterations = 1000)
    print(res14)
    #res15 = forced_controls_2SLS(5, x_array, y_array, d_array, z_array, num_folds = 10, num_iterations = 1000, perform_cv = False)
    #print(res15)
    #res16 = forced_controls_2SLS(6, x_array, y_array, d_array, z_array, num_folds = 10, num_iterations = 1000)
    #print(res16)  
    return 0
    
def NonMetro():
    data= data_prep_NonMetro()
    x, y, d, z = data[0], data[1], data[2], data[3]
    x_array = np.array(x)
    d_array = np.array(d)
    ###Convert to np.float64 so that the glmnet Lasso can run.
    d_array = d_array.astype(np.float64)
    y_array = np.array(y)
    z_array = np.array(z)

    print('NonMetro')
    #res20 = forced_controls_2SLS(0, x_array, y_array, d_array, z_array, num_folds = 10, num_iterations = 1000)
    #print(res20)
    #res22 = forced_controls_2SLS(2, x_array, y_array, d_array, z_array, num_folds = 10, num_iterations = 1000, perform_cv = False)
    #print(res22)
    res24 = forced_controls_2SLS(7, x_array, y_array, d_array, z_array, num_folds = 10, num_iterations = 1000)
    print(res24)
    #res25 = forced_controls_2SLS(5, x_array, y_array, d_array, z_array, num_folds = 10, num_iterations = 1000, perform_cv = False)
    #print(res25)
    #res26 = forced_controls_2SLS(6, x_array, y_array, d_array, z_array, num_folds = 10, num_iterations = 1000)
    #print(res26)    
    return 0
def CS():
    data= data_prep_CS()
    x, y, d, z = data[0], data[1], data[2], data[3]
    x_array = np.array(x)
    d_array = np.array(d)
    ###Convert to np.float64 so that the glmnet Lasso can run.
    d_array = d_array.astype(np.float64)
    y_array = np.array(y)
    z_array = np.array(z)

    print('CS')
    #res30 = forced_controls_2SLS(0, x_array, y_array, d_array, z_array, num_folds = 10, num_iterations = 1000)
    #print(res30)
    #res32 = forced_controls_2SLS(2, x_array, y_array, d_array, z_array, num_folds = 10, num_iterations = 1000, perform_cv = False)
    #print(res32)
    res34 = forced_controls_2SLS(7, x_array, y_array, d_array, z_array, num_folds = 10, num_iterations = 1000)
    print(res34)
    #res35 = forced_controls_2SLS(5, x_array, y_array, d_array, z_array, num_folds = 10, num_iterations = 1000, perform_cv = False)
    #print(res35)
    #res36 = forced_controls_2SLS(6, x_array, y_array, d_array, z_array, num_folds = 10, num_iterations = 1000)
    #print(res36)
    #print('CS')
    #print(res30)
    #print(res32)
    #print(res34)
    #print(res35)
    #print(res36)
    return 0


GDP()
FHFA()
NonMetro()
CS()








