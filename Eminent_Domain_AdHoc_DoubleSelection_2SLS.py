import pandas as pd
import statsmodels.formula.api as sm
import numpy as np

from regression_formulas import ols_formula, ols, two_stage_least_squares

from Eminent_Domain_Data_Setup import data_prep_GDP, data_prep_FHFA, data_prep_CS, data_prep_NonMetro

####################################################################################################################
###Variables selected using stepAIC in Python. Here are the 2SLS results. ###
def run_two_stage_models():
    data3= data_prep_GDP()
    data = data_prep_FHFA()
    data1 = data_prep_NonMetro()
    data2 = data_prep_CS()
    
    ###GDP
    x, y, d, z = data3[0], data3[1], np.array(data3[2]), data3[3]
    ####1st stage regressions
    
    instruments = z.iloc[:,[0, 1, 3, 7, 8, 9, 15, 18, 24, 31]]
    instruments = z.iloc[:,[0, 1, 6, 7, 9]]
    instruments = np.array(instruments)
    x = np.array(x)
    xz = np.concatenate((x, instruments), axis=1)
    #cols = [49]
    #x = x.drop(x.columns[cols], axis=1)
    gdp_res = ols(d, [], xz, return_type = 4, dataframe = False)
    print("GDP RES")
    print(gdp_res)
    
    ###FHFA
    x, y, d, z = data[0], data[1], np.array(data[2]), data[3]
    ####1st stage regressions
    cols = [49]
    x = x.drop(x.columns[cols], axis=1)
    instruments = z.iloc[:,[0, 1, 3, 7, 8, 9, 15, 18, 24, 31]]
    instruments = np.array(instruments)
    x = np.array(x)
    xz = np.concatenate((x, instruments), axis=1)
    fhfa_res = ols(d, [], xz, return_type = 4, dataframe = False)
    print(fhfa_res)
    
    ###NonMetro
    x, y, d, z = data1[0], data1[1], np.array(data1[2]), data1[3]
    ####1st stage regressions
    cols = [0, 1, 31, 36, 37, 40, 41, 47, 48, 49, 50, 55, 57, 59, 62, 64]
    x = x.drop(x.columns[cols], axis=1)
    instruments = z.iloc[:,[0, 1, 3, 7, 8, 9, 15, 18, 24, 31]]
    instruments = instruments.iloc[:,[1, 2, 6]]
    instruments = np.array(instruments)
    x = np.array(x)
    xz = np.concatenate((x, instruments), axis=1)
    nm_res = ols(d, [], xz, return_type = 4, dataframe = False)
    print(nm_res)
    
    ##Case Shiller    
    x, y, d, z = data2[0], data2[1], np.array(data2[2]), data2[3]
    ####1st stage regressions
    cols = [0, 1, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 29, 30, 32, 33, 39, 44, 49, 50, 52, 53, 55, 59, 60, 61, 62, 66, 71]
    x = x.drop(x.columns[cols], axis=1)
    instruments = z.iloc[:,[0, 1, 3, 7, 8, 9, 15, 18, 24, 31]]
    instruments = instruments.iloc[:,[0, 1, 8, 9]]
    instruments = np.array(instruments)
    x = np.array(x)
    xz = np.concatenate((x, instruments), axis=1)
    cs_res = ols(d, [], xz, return_type = 4, dataframe = False)
    print(cs_res)


#######################################################################################
# MAIN FUNCTION
if __name__ == "__main__":
  # Perform Tests
  #print("")
  #print("=== Test GDP ===")
  #print("")
  #test_GDP()
  #print("")
  #print("=== Test FHFA ===")
  #print("")
  #test_FHFA()
  #print("")
  #print("=== Test NonMetro ===")
  #print("")
  #test_NonMetro()
  #print("")
  #print("=== Test CShiller ===")
  #print("")
  #test_CS()
  #print("=== Test Stage 2 Models ===")
  print("")
  run_two_stage_models()
  




