
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

    z_subset = z.iloc[:,[23]]
    #gdp_omp = two_stage_least_squares(y, d, x, z_subset)
    #print(gdp_omp)
    
    z_subset = z.iloc[:,[22, 67]]
    #gdp_lasso = two_stage_least_squares(y, d, x, z_subset)
    #print(gdp_lasso)
    
    z_subset = z.iloc[:,[23]]
    gdp_glm = two_stage_least_squares(y, d, x, z_subset)
    print(gdp_glm)
    
    z_subset = z.iloc[:,[  1,   2,   4,   5,   7,   9,  10,  14,  16,  17,  18,  19,  20,
        21,  23,  24,  25,  26,  28,  29,  30,  31,  32,  33,  34,  38,
        43,  47,  49,  52,  55,  60,  61,  62,  63,  64,  66,  78,  87,
        89,  90,  92,  93,  94,  95, 100, 103, 110, 111, 114, 119, 121,
       126, 131, 132, 137]]

    #gdp_lars = two_stage_least_squares(y, d, x, z_subset)
    #print(gdp_lars)
    
    z_subset = z.iloc[:,[22]]
    #gdp_omp = two_stage_least_squares(y, d, x, z_subset)
    #print(gdp_omp)
    
def test_FHFA():
    ###Set up FHFA Data for use
    data= data_prep_FHFA()
    x, y, d, z = data[0], data[1], data[2], data[3]
    ###FHFA as Outcome Variable
   
    z_subset = z.iloc[:,[23]]
    #gdp_omp = two_stage_least_squares(y, d, x, z_subset)
    #print(gdp_omp)
    
    z_subset = z.iloc[:,[22, 67]]
    #gdp_lasso = two_stage_least_squares(y, d, x, z_subset)
    #print(gdp_lasso)
    
    z_subset = z.iloc[:,[23]]

    gdp_glm = two_stage_least_squares(y, d, x, z_subset)
    print(gdp_glm)
    
    z_subset = z.iloc[:,[1, 2, 4, 5, 7, 9, 10, 16, 17, 18, 19, 21, 23, 25, 27, 28, 29, 30, 31, 32, 33, 36, 37, 38, 43, 47, 61, 62, 63, 64, 78, 84, 89, 90, 93, 95, 103, 110, 111, 114, 119, 120, 126, 132, 137]]

    #gdp_lars = two_stage_least_squares(y, d, x, z_subset)
    #print(gdp_lars)
    
    z_subset = z.iloc[:,[22]]
    #gdp_omp = two_stage_least_squares(y, d, x, z_subset)
    #print(gdp_omp)

def test_NonMetro():
    ###Set up NonMetro Data for use
    data= data_prep_NonMetro()
    x, y, d, z = data[0], data[1], data[2], data[3]

    z_subset = z.iloc[:,[70, 71, 102, 124]]
    gdp_omp = two_stage_least_squares(y, d, x, z_subset)
    print(gdp_omp)
    
    z_subset = z.iloc[:,[4, 9, 10, 11, 17, 18, 20, 21, 26, 31, 32, 40, 46, 50, 51, 56, 59, 62, 69, 71, 81, 90, 91, 94, 108, 124, 127, 128, 137, 139, 140, 143]]

    #gdp_lasso = two_stage_least_squares(y, d, x, z_subset)
    #print(gdp_lasso)
    
    z_subset = z.iloc[:,[4, 10, 51, 66, 71, 124, 127]]

    gdp_glm = two_stage_least_squares(y, d, x, z_subset)
    print(gdp_glm)
    
    z_subset = z.iloc[:,[7, 9, 11, 14, 15, 16, 17, 18, 19, 20, 25, 26, 30, 31, 32, 33, 34, 35, 36, 39, 41, 45, 46, 48, 51, 52, 59, 66, 69, 71, 74, 81, 82, 84, 91, 97, 100, 107, 109, 115, 118, 124, 128, 137, 140, 141, 143]]

    #gdp_lars = two_stage_least_squares(y, d, x, z_subset)
    #print(gdp_lars)

    z_subset = z.iloc[:,[1, 11, 12, 13, 14, 15, 16, 18, 21, 25, 31, 33, 35, 37, 46, 49, 65, 97, 102, 107, 124, 129, 143]]
    
    

    #gdp_omp = two_stage_least_squares(y, d, x, z_subset)
    #print(gdp_omp)                 
                      

def test_CS():
    ###Set up CS Data for use
    data= data_prep_CS()
    x, y, d, z = data[0], data[1], data[2], data[3]

    z_subset = z.iloc[:,[24, 104]]
    #gdp_omp = two_stage_least_squares(y, d, x, z_subset)
    #print(gdp_omp)
    
    
    z_subset = z.iloc[:,[ 1, 3, 4, 5, 7, 10, 11, 12, 14, 15, 16, 17, 20, 21, 22, 27, 30, 31, 32, 34, 37, 40, 43, 48, 49, 55, 56, 60, 62, 66, 84, 93, 101, 104, 107, 111, 112, 115, 119, 123, 127, 128, 131, 141, 144, 146]]

    #gdp_lasso = two_stage_least_squares(y, d, x, z_subset)
    #print(gdp_lasso)
    
    z_subset = z.iloc[:,[0, 4, 20, 22, 23, 27, 64, 69, 97, 104]]

    gdp_glm = two_stage_least_squares(y, d, x, z_subset)
    print(gdp_glm)
    
    z_subset = z.iloc[:,[ 0, 1, 2, 3, 6, 7, 9, 10, 12, 13, 14, 15, 16, 17, 19, 20, 24, 25, 26, 27, 29, 30, 31, 34, 35, 37, 38, 39, 40, 41, 42, 44, 46, 47, 48, 49, 51, 52, 53, 55, 56, 57, 58, 64, 65, 66, 72, 74, 75, 76, 79, 80, 84, 85, 86, 87, 88, 91, 93, 94, 97, 100, 101, 102, 104, 109, 110, 111, 112, 113, 116, 117, 118, 119, 122, 124, 128, 129, 133, 135, 137, 142, 143, 144, 147, 148]]

    #gdp_lars = two_stage_least_squares(y, d, x, z_subset)
    #print(gdp_lars)
    


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




