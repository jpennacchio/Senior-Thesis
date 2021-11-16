import sys
import numpy as np
from constants import *
import pandas as pd

def preProcess(observations,demean,scale):
  """Preprocess observation matrix before starting the OMP iterations
  
  Args:
    observations: rectangular matrix with observations
    demean: Integer flag to demean columns in observation matrix. 
            It can assume the following values:
            - ipDemeanALL:   Subtract the associated mean from all columns
            - ipDemeanFIRST: Will do the same as ALL, but have call separate so easier to work with OMP
                             and switching back to the original observ. after 1st loop.
            - ipDemeanNONE:  Do not subtract the mean.

    scale: Integer flag to scale the columns of the observation matrix.
           It can assume the following values
           - ipScaleYES: Divide each column by its standard deviation
           - ipScaleNO:  Do not scale columns

  Returns:
    newObs: The modified observation matrix with demeaned/scaled columns

  """

  # Find Columns Mean/Std
  column_means = np.sum(observations,axis=0)/float(observations.shape[0])  
  column_sd = np.std(observations,axis=0)

  # If there is an intercept column, which by construction has 0 SD, set it to 1 to avoid any issues
  # This works since dividing by 1 won't scale the intercept at all. 
  if np.abs(column_sd[0]) < kMathZero:
    column_sd[0] = 1.0

  if(demean == ipDemeanALL or demean == ipDemeanFIRST):
    newObs = observations - column_means
  elif(demean == ipDemeanNONE):    
    newObs = observations
  else:
    print("ERROR: Demean Option not Found.")
    sys.exit(-1)

  # Arguments for scale are "YES" and "NO"
  if(scale == ipScaleYES):
    newObs = newObs/column_sd

  # Return the modified observations
  return newObs


########################################################################################
def preprocess_arrays(outcome, treatment, x_vars = [], instruments = []):
    ###Any missing parts will be shown as empty lists.###
    if(len(x_vars) !=0):
        x_vars = pd.DataFrame(x_vars)
        x_vars.columns = np.arange(len(x_vars.columns))
        x_vars = x_vars.add_prefix('x')
    else:
        x_vars = []  
    ###y and d will always be one column, so need to convert from series[default] to frame. 
    outcome = pd.DataFrame(outcome)
    outcome.columns = ['y']
    if(len(treatment) !=0):
        treatment = pd.DataFrame(treatment)
        treatment.columns = ['d']
    else:
        treatment = []
    
    if(len(instruments) !=0):
        instruments = pd.DataFrame(instruments)
        instruments.columns = np.arange(len(instruments.columns))
        instruments = instruments.add_prefix('z') 
    else:
        instruments = []
        
    return outcome, treatment, x_vars, instruments
##############################################################################################
###This is used to transform the data for the Lasso in particular (but can also use for OMP)####
###Don't use d or z empty sets, but would prefer not to change argument order.
def transform_data_with_controls(x,y = [],d = [],z = []):
    if(len(d)==0 or len(z) ==0):
        return -999
    ###Step 1: Perform OLS regression of d[treatment] on x[controls]
    d_coef= np.linalg.lstsq(x,d)[0]  
    d_resid = d-x.dot(d_coef)
    ###Step 2: Perform OLS regression of z[instruments] on x[controls]. 
    ###Need to do this instrument by instrument; so matrix multiplication might be easier###
    x = np.matrix(x)
    z = np.matrix(z)
    xt = x.transpose()
    xxinv = np.linalg.inv(xt*x);                           
                           
    z_resid = z - x*xxinv*(xt*z);
    ###Try this###
    z_resid = np.array(z_resid)
    ##Step 3: Perform OLS regression of y[outcome] on x[controls]
    if(len(y) !=0):
        y_coef = np.linalg.lstsq(x,y)[0]  
        y_resid = y-x.dot(y_coef)
    else:
        y_resid = -100
    ###Step 4: Normalize by dividing by the respective sd's.
    ###Need to normalize for Lasso selection, but not when actually computing the OLS coefficients down the line.###
    d_std = np.std(d_resid)
    d_normalized = np.divide(d_resid, d_std)    
    z_std = np.std(z_resid, axis=0)
    z_normalized = np.divide(z_resid, z_std)

    
    return d_normalized, z_normalized, d_resid, z_resid, y_resid


