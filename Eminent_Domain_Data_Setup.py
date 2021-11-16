###Imports
import pandas as pd
import statsmodels.formula.api as sm
import numpy as np

###IMPORT DATA
###Read in the preprocessed data from a CSV file exported from Matlab
###CSV file is of the form [x y d z].
###Import all instruments for simplicity; can subset them here. 

##Subdivide the csv file into x, y, d, and z. Then assign column names to use in the regressions. 
##This must be done before any of the functions


def data_prep_GDP():
    gdp_df = pd.read_csv('preprocessed_instruments_pre_lasso.csv', header=None)
    x = gdp_df.iloc[:,0:80]
    x.columns = np.arange(len(x.columns))
    x = x.add_prefix('x')

    ###y and d will always be one column, so need to convert from series[default] to frame. 
    y = gdp_df.iloc[:,80]
    y = y.to_frame()
    y.columns = ['y']

    d = gdp_df.iloc[:,81]
    d = d.to_frame()
    d.columns = ['d']

    z = gdp_df.iloc[:,82:222]
    z.columns = np.arange(len(z.columns))
    z = z.add_prefix('z')
    
    return x, y, d, z

def data_prep_FHFA():
    fhfa_df = pd.read_csv('fhfa_preprocessed_data.csv', header=None)
    x = fhfa_df.iloc[:,0:80]
    x.columns = np.arange(len(x.columns))
    x = x.add_prefix('x')

    ###y and d will always be one column, so need to convert from series[default] to frame. 
    y = fhfa_df.iloc[:,80]
    y = y.to_frame()
    y.columns = ['y']

    d = fhfa_df.iloc[:,81]
    d = d.to_frame()
    d.columns = ['d']

    z = fhfa_df.iloc[:,82:222]
    z.columns = np.arange(len(z.columns))
    z = z.add_prefix('z')
    
    return x, y, d, z


def data_prep_NonMetro():
    nonmetro_df = pd.read_csv('nonmetro_preprocessed_data.csv', header=None)
    x = nonmetro_df.iloc[:,0:65]
    x.columns = np.arange(len(x.columns))
    x = x.add_prefix('x')

    ###y and d will always be one column, so need to convert from series[default] to frame. 
    y = nonmetro_df.iloc[:,65]
    y = y.to_frame()
    y.columns = ['y']

    d = nonmetro_df.iloc[:,66]
    d = d.to_frame()
    d.columns = ['d']

    z = nonmetro_df.iloc[:,67:212]
    z.columns = np.arange(len(z.columns))
    z = z.add_prefix('z')
    
    return x, y, d, z
        
def data_prep_CS():
    
    cs_df = pd.read_csv('cs_preprocessed_data.csv', header=None)
    
    x = cs_df.iloc[:,0:72]
    x.columns = np.arange(len(x.columns))
    x = x.add_prefix('x')

    ###y and d will always be one column, so need to convert from series[default] to frame. 
    y = cs_df.iloc[:,72]
    y = y.to_frame()
    y.columns = ['y']

    d = cs_df.iloc[:,73]
    d = d.to_frame()
    d.columns = ['d']

    z = cs_df.iloc[:,74:223]
    z.columns = np.arange(len(z.columns))
    z = z.add_prefix('z')
    
    return x, y, d, z
