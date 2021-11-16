###The primary goal of this is to examine the percent cutoff in the OMP Algorithm

###Key Imports
import sys
import numpy as np
import pandas as pd
from constants import *
from sklearn.model_selection import KFold
from sklearn import linear_model
from data_preprocessing import transform_data_with_controls
###Import OMP since that is what the CV will be tested on 
from omp import omp, omp_controls

from Eminent_Domain_Data_Setup import data_prep_GDP, data_prep_FHFA, data_prep_CS, data_prep_NonMetro
from joblib import Parallel, delayed

def outer_omp_cv(Z, y, x, train_index, test_index, cutoff_list, split_counter):
    ###Will bind all of the ResNorms together at the end.###
    temp_ResNorm = np.zeros(len(cutoff_list))
    Z_train, Z_test = Z[train_index], Z[test_index]
    y_train, y_test = y[train_index], y[test_index]
    if(len(x) !=0):
        x_train, x_test = x[train_index], x[test_index]
        xz_test = np.matrix(np.concatenate ((x_test,Z_test), axis=1))
    else:
        x_train = []
        x_test = []
        xz_test = Z_test
    if(y_test.ndim ==2):
        y_test = np.concatenate(y_test)  
        ###Parallelize this section###
    for cutoff_index in range(len(cutoff_list)):
        temp_ResNorm[cutoff_index] = inner_omp_cross_validation(Z_train, y_train, x_train, xz_test, y_test, cutoff_list[cutoff_index])    
    return temp_ResNorm
    
    
####Parallelize this section so it doesn't take forever.###
def inner_omp_cross_validation(Z_train, y_train, x_train, xz_test, y_test, temp_cutoff): 
    ####run OMP here on the training data and obtain the coefficients (argument [0])
    res = omp(Z_train, y_train, x_train, cutoff = temp_cutoff)
    train_coefficients = res[0]
    train_coefficients = np.transpose(np.matrix(train_coefficients))
    ###Now test data is A, and train_coefficients is x--> compute b and compare to true b.
    ##Include the controls first when combining with instruments. 
    test_predicted_outcomes = np.matrix(xz_test) * np.matrix(train_coefficients)       
    ###convert to array, then to vector
    test_predicted_outcomes = np.concatenate(np.asarray(test_predicted_outcomes))
    ###now measure error on the test data
    test_residuals = y_test - test_predicted_outcomes
    test_norm = np.linalg.norm(y_test,2)
    if(np.linalg.norm(y_test,2) ==0):
           ##Set equal to 0.01 to avoid divide by zero issue. It is possible to have all zeros in test set. 
            test_norm = 0.01
    resNorm= np.linalg.norm(test_residuals,2)/test_norm
    return resNorm
                           
###z=instruments, y= outcomes, x=controls[force]
###Default is only checking the 1% cutoff.
def n_fold_cross_validation(Z, y, x=[], n_splits = 10, cutoff_list = [1.00], shuffle=False, omp_control_method = False):
    ###Transform the data at the beginning###
    if(omp_control_method ==True):
        modified_variables = transform_data_with_controls(x, [], y, Z)
        Z = modified_variables [1]
        y = modified_variables [0]
        x = []
    ###Divide into 10 folds; run OMP on each, and then on the test set apply res(the coefficients) from the training data
    ###and compute the error on B.
    ####Also decide how to work on this with forcing the controls to be included. Maybe merge and then force the corresponding
    ###columns to be included (that's how OMP would work)
    kf = KFold(n_splits, shuffle)
    kf.get_n_splits(Z)
    resNormVec = np.zeros((n_splits, len(cutoff_list)))
    split_counter = 0
    #resNormVec= Parallel(n_jobs = nproc, verbose = 150)(delayed(outer_omp_cv)(Z, y, x, train_index, test_index, cutoff_list, split_counter, omp_control_method) for train_index, test_index in kf.split(Z))                
    #resNormVec = np.asarray(resNormVec)
    for train_index, test_index in kf.split(Z):
        resNormVec[split_counter,] = outer_omp_cv(Z, y, [], train_index, test_index, cutoff_list, split_counter)
        split_counter += 1    
    ###Compute the means of each column(possible cutoff). Rows now correspond to each split
    test_means = np.mean(resNormVec, axis=0)
    #print(test_means)
    min_index = np.argmin(test_means)
    min_cutoff = cutoff_list[min_index]
    return min_cutoff


def test1():
    ''' This is a simple test of the CV split that
    replicates an online example. '''
    
    X2 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y2 = np.array([9, 10, 11, 12])
    q = n_fold_cross_validation(X2, y2, n_splits=2)
    return q

def test_belloni():
    data1= data_prep_GDP()
    data2= data_prep_FHFA()
    data3 = data_prep_NonMetro()
    data4= data_prep_CS()
    s = np.zeros((1,4))
    for num_reps_cv in range(1):
        for i in range(4):
            if i==0:
                data = data1
            elif i==1:
                data = data2
            elif i==2:
                data = data3
            else:
                data = data4
            x, y, d, z = data[0], data[1], data[2], data[3]
            x_array = np.array(x)
            d_array = np.array(d)
            d_array = d_array.astype(np.float64)
            z_array = np.array(z)
            #y_array = np.array(y)
            ###In step 1, you are working with d~x+z. So exclude y right now.
            ###Right now, only 1 fold is being tested
            ##Order of the command is z, y, x. 
            #s[num_reps_cv, i]= n_fold_cross_validation(z_array, d_array, x_array, shuffle=True, cutoff_list = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5,0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 12.5, 15.0], omp_control_method = False)
            s[num_reps_cv, i]= n_fold_cross_validation(z_array, d_array, x_array, shuffle=True, cutoff_list = [1, 1.5, 2.0, 2.5, 3.5, 5, 7.5], omp_control_method = False)  
            #s[i]= n_fold_cross_validation(z_array, d_array, x_array, shuffle=True, cutoff_list = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.5, 8.0, 10.0, 12.5, 15.0])
        rep_cv_means = np.mean(s, axis=0)

    return rep_cv_means

def test2():
    data= data_prep_GDP()
    x, y, d, z = data[0], data[1], data[2], data[3]
    x_array = np.array(x)
    d_array = np.array(d)
    d_array = d_array.astype(np.float64)
    z_array = np.array(z)
    q= n_fold_cross_validation(z_array, d_array, x_array, shuffle=True, cutoff_list = [1, 1.5, 2.0, 2.5, 3.5, 5, 7.5], omp_control_method = True, nproc=4) 
    return q

#t= test_belloni()
#t = test1()
#print(t)
#print("STARTED")
#qq = test2()
#print(qq)
