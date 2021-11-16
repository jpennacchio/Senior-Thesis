
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

    x_subset = x.iloc[:,[10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 47, 49, 52, 64, 70, 71]]
    z_subset = z.iloc[:,[1, 38]]
    gdp_omp = two_stage_least_squares(y, d, x_subset, z_subset)
    print(gdp_omp)
    
    x_subset = x.iloc[:,[38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]]
    z_subset = z.iloc[:,[67, 69, 6, 28]]
    gdp_lasso = two_stage_least_squares(y, d, x_subset, z_subset)
    print(gdp_lasso)
    
    x_subset = x.iloc[:,[50]]
    z_subset = z.iloc[:,[17, 22, 1, 2, 5]]
    gdp_glm = two_stage_least_squares(y, d, x_subset, z_subset)
    print(gdp_glm)
    
    x_subset = x.iloc[:,[0, 1, 13, 14, 15, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 53, 56, 57, 74]]
    z_subset = z.iloc[:,[1, 5, 6, 28, 36, 37, 38, 43, 46, 47, 63, 66, 67, 71, 73, 76, 80, 81, 103, 108, 120, 121, 132]]
    gdp_lars = two_stage_least_squares(y, d, x_subset, z_subset)
    print(gdp_lars)
    
def test_FHFA():
    ###Set up FHFA Data for use
    data= data_prep_FHFA()
    x, y, d, z = data[0], data[1], data[2], data[3]
    ###FHFA as Outcome Variable
   
    x_subset = x.iloc[:,[10, 11, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 45, 48, 49, 56, 70]]
    z_subset = z.iloc[:,[1]]
    gdp_omp = two_stage_least_squares(y, d, x_subset, z_subset)
    print(gdp_omp)
    
    x_subset = x.iloc[:,[38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]]
    z_subset = z.iloc[:,[67, 69, 6, 28]]
    gdp_lasso = two_stage_least_squares(y, d, x_subset, z_subset)
    print(gdp_lasso)
    
    x_subset = x.iloc[:,[50]]
    z_subset = z.iloc[:,[17, 22, 1, 2, 5]]
    gdp_glm = two_stage_least_squares(y, d, x_subset, z_subset)
    print(gdp_glm)
    
    x_subset = x.iloc[:,[0, 1, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 53, 55, 56, 58, 60, 63, 64, 65, 66, 67, 68, 69, 70, 71, 73, 74, 76]]
    z_subset = z.iloc[:,[0, 1, 2, 4, 5, 6, 7, 10, 11, 12, 13, 15, 16, 17, 18, 19, 21, 25, 26, 28, 30, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 56, 57, 59, 61, 62, 64, 66, 67, 68, 70, 71, 73, 75, 76, 77, 78, 80, 81, 82, 83, 86, 88, 89, 90, 95, 96, 97, 99, 102, 103, 105, 106, 107, 109, 110, 111, 112, 113, 114, 115, 116, 118, 119, 120, 121, 122, 123, 125, 126, 129, 130, 133, 134, 135, 136, 138, 139]]
    gdp_lars = two_stage_least_squares(y, d, x_subset, z_subset)
    print(gdp_lars)

def test_NonMetro():
    ###Set up NonMetro Data for use
    data= data_prep_NonMetro()
    x, y, d, z = data[0], data[1], data[2], data[3]

    x_subset = x.iloc[:,[13, 14, 15, 16, 17, 18, 19, 20, 31]]
    z_subset = z.iloc[:,[66]]
    gdp_omp = two_stage_least_squares(y, d, x_subset, z_subset)
    print(gdp_omp)
    
    x_subset = x.iloc[:,[21, 22, 23, 24, 25, 26, 27, 28, 29, 30]]
    z_subset = z.iloc[:,[]]
    gdp_lasso = two_stage_least_squares(y, d, x_subset, z_subset)
    print(gdp_lasso)
    
    x_subset = x.iloc[:,[0, 3, 14, 15, 17, 20, 24, 36, 43, 47]]
    z_subset = z.iloc[:,[66, 91, 92, 102, 124, 127, 1, 4, 10, 11, 141, 14, 143, 33, 42, 46, 50, 51]]
    gdp_glm = two_stage_least_squares(y, d, x_subset, z_subset)
    print(gdp_glm)
    
    x_subset = x.iloc[:,[0, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 46]]
    z_subset = z.iloc[:,[0, 2, 5, 6, 7, 11, 14, 16, 18, 26, 28, 33, 40, 42, 44, 45, 46, 48, 50, 51, 53, 57, 59, 62, 65, 66, 69, 74, 75, 80, 82, 85, 86, 88, 108, 110, 114, 117, 123, 124, 126, 128, 132, 139, 140]]
    gdp_lars = two_stage_least_squares(y, d, x_subset, z_subset)
    print(gdp_lars)


def test_CS():
    ###Set up CS Data for use
    data= data_prep_CS()
    x, y, d, z = data[0], data[1], data[2], data[3]

    x_subset = x.iloc[:,[2, 4, 10, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 38, 39]]
    z_subset = z.iloc[:,[1, 104]]
    gdp_omp = two_stage_least_squares(y, d, x_subset, z_subset)
    print(gdp_omp)
    
    x_subset = x.iloc[:,[29, 30, 31, 32, 33, 34, 35, 36, 37, 38]]
    z_subset = z.iloc[:,[69, 6, 71, 73, 29]]
    gdp_lasso = two_stage_least_squares(y, d, x_subset, z_subset)
    print(gdp_lasso)
    
    x_subset = x.iloc[:,[40]]
    z_subset = z.iloc[:,[1, 47, 48, 23]]
    gdp_glm = two_stage_least_squares(y, d, x_subset, z_subset)
    print(gdp_glm)
    
    x_subset = x.iloc[:,[0, 12, 15, 16, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 43, 48, 50, 54, 66, 70]]
    z_subset = z.iloc[:,[1, 3, 5, 6, 10, 11, 13, 15, 16, 17, 25, 26, 27, 29, 30, 31, 34, 35, 41, 42, 43, 45, 47, 48, 51, 59, 62, 63, 67, 68, 69, 70, 72, 81, 82, 83, 84, 88, 91, 93, 97, 101, 112, 115, 117, 120, 124, 126, 127, 128, 129, 130, 131, 133, 134, 135, 141, 144, 145, 146]]
    gdp_lars = two_stage_least_squares(y, d, x_subset, z_subset)
    print(gdp_lars)
    


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




