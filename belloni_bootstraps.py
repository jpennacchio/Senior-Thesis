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
from MultistepResidualBootstrap import Residual_Bootstrap_2SLS

###############################################################################################################

###Step 1: IMPORT DATA
###Read in the preprocessed data from a CSV file exported from Matlab
###CSV file is of the form [x y d z].

def GDP():
    data= data_prep_GDP()
    x, y, d, z = data[0], data[1], data[2], data[3]
    x_array = np.array(x)
    d_array = np.array(d)
    ###Convert to np.float64 so that the glmnet Lasso can run.
    d_array = d_array.astype(np.float64)
    y_array = np.array(y)
    z_array = np.array(z)

    ###Solvers: 0= Old OMP, 2= Lasso, 4 = Glmnet, 5 = LassoLars, 6= OMP w/ Chernozukov controls method
    print('GDP')
    #res0 = Residual_Bootstrap_2SLS(0, x_array, y_array, d_array, z_array)
    #print(res0)
    #res2 = Residual_Bootstrap_2SLS(2, x_array, y_array, d_array, z_array)
    #print(res2)
    res4 = Residual_Bootstrap_2SLS(7, x_array, y_array, d_array, z_array, num_bootstrap_samples = 250)
    print(res4)
    #res5 = Residual_Bootstrap_2SLS(5, x_array, y_array, d_array, z_array)
    #print(res5)
    #res6 = Residual_Bootstrap_2SLS(6, x_array, y_array, d_array, z_array)
    #print(res6)

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
    #res10 = Residual_Bootstrap_2SLS(0, x_array, y_array, d_array, z_array)
    #print(res10)
    #res12 = Residual_Bootstrap_2SLS(2, x_array, y_array, d_array, z_array)
    #print(res12)
    res14 = Residual_Bootstrap_2SLS(7, x_array, y_array, d_array, z_array, num_bootstrap_samples = 250)
    print(res14)
    #res15 = Residual_Bootstrap_2SLS(5, x_array, y_array, d_array, z_array)
    #print(res15)
    #res16 = Residual_Bootstrap_2SLS(6, x_array, y_array, d_array, z_array)
    #print(res16)

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
    #res20 = Residual_Bootstrap_2SLS(0, x_array, y_array, d_array, z_array)
    #print(res20)
    #res22 = Residual_Bootstrap_2SLS(2, x_array, y_array, d_array, z_array)
    #print(res22)
    res24 = Residual_Bootstrap_2SLS(7, x_array, y_array, d_array, z_array, num_bootstrap_samples = 250)
    print(res24)
    #res25 = Residual_Bootstrap_2SLS(5, x_array, y_array, d_array, z_array)
    #print(res25)
    #res26 = Residual_Bootstrap_2SLS(6, x_array, y_array, d_array, z_array)
    #print(res26)
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
    #res30 = Residual_Bootstrap_2SLS(0, x_array, y_array, d_array, z_array)
    #print(res30)
    #res32 = Residual_Bootstrap_2SLS(2, x_array, y_array, d_array, z_array)
    #print(res32)
    res34 = Residual_Bootstrap_2SLS(7, x_array, y_array, d_array, z_array, num_bootstrap_samples = 250)
    print(res34)
    #res35 = Residual_Bootstrap_2SLS(5, x_array, y_array, d_array, z_array)
    #print(res35)
    #res36 = Residual_Bootstrap_2SLS(6, x_array, y_array, d_array, z_array)
    #print(res36)

#GDP()
#FHFA()
#NonMetro()
CS()






