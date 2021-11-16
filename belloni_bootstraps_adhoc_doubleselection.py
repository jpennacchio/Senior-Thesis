# Imports
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
from constants import *
from random import randint ##Imported for Random Numbers
from random import sample ##Select list of unique random integers
from runAnyMethod import *
from data_preprocessing import transform_data_with_controls
###2SLS Infrastructure
from regression_formulas import forced_controls_2SLS
###Data Setup
from Eminent_Domain_Data_Setup import data_prep_GDP, data_prep_FHFA, data_prep_CS, data_prep_NonMetro
##Multi-Step Bootstrap
from MultistepResidualBootstrap_AdHoc_DoubleSelection import union, Residual_Bootstrap_2SLS_DoubleSelection 

###############################################################################################################

###Step 1: IMPORT DATA
###Read in the preprocessed data from a CSV file exported from Matlab
###CSV file is of the form [x y d z].

#0-80, 80, 81, 82-222 for GDP and FHFA; 0-65, 65, 66, 67-212 for NonMetro; 0-72, 72, 73, 74-221 for CS


    
####Step 3: Run 2SLS with the controls

###Solvers: 0= Old OMP, 2= Lasso, 4 = Glmnet, 5 = LassoLars, 6= OMP w/ Chernozukov controls method
def GDP():
    data= data_prep_GDP()
    x, y, d, z = data[0], data[1], data[2], data[3]
    z = z.iloc[:,[0 ,1 ,3 ,7 ,8 ,9 ,15 ,18 ,24 ,31]]

    x_array = np.array(x)
    d_array = np.array(d)
    ###Convert to np.float64 so that the glmnet Lasso can run.
    d_array = d_array.astype(np.float64)
    y_array = np.array(y)
    z_array = np.array(z)
    print('GDP')
    res0 = Residual_Bootstrap_2SLS_DoubleSelection(x_array, y_array, d_array, z_array)
    print(res0)
    return res0

def FHFA():
    data= data_prep_FHFA()
    x, y, d, z = data[0], data[1], data[2], data[3]
    z = z.iloc[:,[0 ,1 ,3 ,7 ,8 ,9 ,15 ,18 ,24 ,31]]

    x_array = np.array(x)
    d_array = np.array(d)
    ###Convert to np.float64 so that the glmnet Lasso can run.
    d_array = d_array.astype(np.float64)
    y_array = np.array(y)
    z_array = np.array(z)

    print('FHFA')
    res10 = Residual_Bootstrap_2SLS_DoubleSelection(x_array, y_array, d_array, z_array)
    print(res10)
    return res10

def NonMetro():
    data= data_prep_NonMetro()
    x, y, d, z = data[0], data[1], data[2], data[3]
    z = z.iloc[:,[0 ,1 ,3 ,7 ,8 ,9 ,15 ,18 ,24 ,31]]

    x_array = np.array(x)
    d_array = np.array(d)
    ###Convert to np.float64 so that the glmnet Lasso can run.
    d_array = d_array.astype(np.float64)
    y_array = np.array(y)
    z_array = np.array(z)

    print('NonMetro')
    res20 = Residual_Bootstrap_2SLS_DoubleSelection(x_array, y_array, d_array, z_array)
    print(res20)
    return res20
   
def CS():
    data= data_prep_CS()
    x, y, d, z = data[0], data[1], data[2], data[3]
    z = z.iloc[:,[0 ,1 ,3 ,7 ,8 ,9 ,15 ,18 ,24 ,31]]

    x_array = np.array(x)
    d_array = np.array(d)
    ###Convert to np.float64 so that the glmnet Lasso can run.
    d_array = d_array.astype(np.float64)
    y_array = np.array(y)
    z_array = np.array(z)

    print('CS')
    res30 = Residual_Bootstrap_2SLS_DoubleSelection(x_array, y_array, d_array, z_array)
    print(res30)
    return res30
   

#
#GDP()
#FHFA()
#NonMetro()
CS()