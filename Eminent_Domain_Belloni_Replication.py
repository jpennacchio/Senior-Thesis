
# coding: utf-8

# In[3]:

import pandas as pd
import statsmodels.formula.api as sm
import numpy as np

from regression_formulas import ols_formula, ols, two_stage_least_squares

from Eminent_Domain_Data_Setup import data_prep_GDP, data_prep_FHFA, data_prep_CS, data_prep_NonMetro


###############################################################################################################
###CONSTANTS
# Arguments for demean are "ALL," "FIRST", and "NONE."
ipNoINTERCEPT  = 0
ipYesINTERCEPT =  1

####################################################################################################################
####Code to Replicate the Belloni Table####
    
def test_GDP():
    ###Set up GDP Data for use
    data= data_prep_GDP()
    x, y, d, z = data[0], data[1], data[2], data[3]
    ###GDP as Outcome Variable
    ###OLS
    gdp_ols = ols(y, d, x, return_type = 4)
    print(gdp_ols)

    ###1 instrument test
    z_one_instrument = z.iloc[:,[0]]
    gdp_one_instrument = two_stage_least_squares(y, d, x, z_one_instrument)
    print(gdp_one_instrument)

    ###2SLS [Chen and Yeh]
    z_subset = z.iloc[:,[0,1]]
    gdp_2SLS = two_stage_least_squares(y, d, x, z_subset)
    print(gdp_2SLS)
    ###Lasso
    z_subset = z.iloc[:,[23]]
    gdp_lasso = two_stage_least_squares(y, d, x, z_subset)
    print(gdp_lasso)
    ###Lasso +
    z_subset = z.iloc[:,[0,1,23]]
    gdp_lasso_plus = two_stage_least_squares(y, d, x, z_subset)
    print(gdp_lasso_plus)
 
def test_FHFA():
    ###Set up FHFA Data for use
    data= data_prep_FHFA()
    x, y, d, z = data[0], data[1], data[2], data[3]
    ###FHFA as Outcome Variable
    ###OLS
    fhfa_ols = ols(y, d, x, return_type = 4)
    print(fhfa_ols)

    ###2SLS [Chen and Yeh]
    z_subset = z.iloc[:,[0,1]]
    fhfa_2SLS = two_stage_least_squares(y, d, x, z_subset)
    print(fhfa_2SLS)
    ###Lasso
    z_subset = z.iloc[:,[23]]
    fhfa_lasso = two_stage_least_squares(y, d, x, z_subset)
    print(fhfa_lasso)
    ###Lasso +
    z_subset = z.iloc[:,[0,1,23]]
    fhfa_lasso_plus = two_stage_least_squares(y, d, x, z_subset)
    print(fhfa_lasso_plus)


def test_NonMetro():
    ###Set up NonMetro Data for use
    data= data_prep_NonMetro()
    x, y, d, z = data[0], data[1], data[2], data[3]

    ###NonMetro as Outcome Variable
    ###OLS
    nonmetro_ols = ols(y, d, x, return_type = 4)
    print(nonmetro_ols)

    ###2SLS [Chen and Yeh]
    z_subset = z.iloc[:,[0,1]]
    nonmetro_2SLS = two_stage_least_squares(y, d, x, z_subset)
    print(nonmetro_2SLS)
    ###Lasso
    z_subset = z.iloc[:,[41, 66, 69, 119]]
    nonmetro_lasso = two_stage_least_squares(y, d, x, z_subset)
    print(nonmetro_lasso)
    ###Lasso +
    z_subset = z.iloc[:,[0,1,41, 66, 69, 119]]
    nonmetro_lasso_plus = two_stage_least_squares(y, d, x, z_subset)
    print(nonmetro_lasso_plus)


def test_CS():
    ###Set up CS Data for use
    data= data_prep_CS()
    x, y, d, z = data[0], data[1], data[2], data[3]

    ###Case Shiller as Outcome Variable
    ###OLS
    cs_ols = ols(y, d, x, return_type = 4)
    print(cs_ols)

    ###2SLS [Chen and Yeh]
    z_subset = z.iloc[:,[0,1]]
    cs_2SLS = two_stage_least_squares(y, d, x, z_subset)
    print(cs_2SLS)
    ###Lasso
    z_subset = z.iloc[:,[1,22]]
    cs_lasso = two_stage_least_squares(y, d, x, z_subset)
    print(cs_lasso)
    ###Lasso +
    z_subset = z.iloc[:,[0,1,22]]
    cs_lasso_plus = two_stage_least_squares(y, d, x, z_subset)
    print(cs_lasso_plus)


#######################################################################################
# MAIN FUNCTION
if __name__ == "__main__":
  # Perform Tests
  print("")
  print("=== Test GDP ===")
  print("")
  test_GDP()
  print("")
  print("=== Test FHFA ===")
  print("")
  test_FHFA()
  print("")
  print("=== Test NonMetro ===")
  print("")
  test_NonMetro()
  print("")
  print("=== Test CShiller ===")
  print("")
  test_CS()
  
  #print("Replicating Belloni Table Results...")
  #print("")
  #main()




