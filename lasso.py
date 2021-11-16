# Imports
from sklearn import linear_model
import numpy as np
#from runAnyMethod import run_choice_method_once
from data_preprocessing import transform_data_with_controls


def lassoSolve(A,b, num_folds = 10, num_iterations = 1000, perform_cv = True):
  if (perform_cv==True):
      clf = linear_model.LassoCV(tol=1.0e-16, fit_intercept= False, cv = num_folds, max_iter = num_iterations)
      clf.fit(A,b)
      alpha = clf.alpha_
      #print(clf.alphas_)
  else:
      clf = linear_model.Lasso(alpha=0.2,fit_intercept=False)  
      clf.fit(A,b)
      alpha = 0.2
  ###This is the alpha selected
  #print("Selected alpha value")
  #print(clf.alpha_)
  ###These are the alphas that were considered. 
  #print("Alpha Path")
  coefficients = clf.coef_.ravel()
  return coefficients, alpha


####Default to perform 10-fold Cross Validation.####
def lasso_force_controls(observations,outcomes, forced_variables, num_folds = 10, num_iterations = 1000, perform_cv = True):
    ###Should be converted from a column vector to a 1D array
    if(len(forced_variables) !=0):     
        modified_variables = transform_data_with_controls(forced_variables, [], outcomes, observations)
        d_lasso = modified_variables [0]
        d_lasso = d_lasso.ravel()
        z_lasso = modified_variables [1]
        ###Use Cross Validation here#####
        alpha_list = np.linspace(0.01, 1.00, 500)
        if (perform_cv==True):
            clf = linear_model.LassoCV(tol=1.0e-16, fit_intercept= False, cv = num_folds, max_iter = num_iterations, alphas = alpha_list)
            clf.fit(z_lasso, d_lasso)
            alpha = clf.alpha_
            print(alpha)
            #print(clf.alphas_)
        else:
            clf = linear_model.Lasso(alpha=0.20,fit_intercept=False,max_iter=500,tol=1.0e-16)
            clf.fit(z_lasso, d_lasso)
            alpha = 0.20
        coefficients = clf.coef_.ravel()
    else:
        res = lassoSolve(observations,outcomes, num_folds, num_iterations, perform_cv)
        coefficients = res[0].ravel()
        alpha = res[1]
        
    return coefficients, alpha
    

def test1():
  """
  Test on slightly larger matrix (5 observations * 10 predictors)
  """
  ###Note: Lasso did not converge with this test case.
  # Create test observations:
  A = np.array([[1, 0.9, 0.8, 0.2, -0.7, 1.3, 2.3, -0.5, 1.4, -0.2],[1, -0.1, 0.7, 0.3, -0.2, 0.9, 0.7, 1.3, 1.8, -2.5], [1,0.5,0.4,0.3,0.9,1.9, 2.3, -3.1, -2.9, -0.7], [1,0.2,0.9,0.1,0.7,2.3, 0.5, 2.5, -1.5, -2.9], [1,0.5,1,1.2,1.5,-0.1, 0.3, 0.7, 1.9, -0.6]])
  X = np.array( [[0.5], [0], [0], [0], [-0.2], [0], [0], [3.1], [0], [0]])
  B = A.dot(X)
  B = B.ravel()
  controls2 = np.array([[0.3], [0.7], [-0.9], [0.4], [2.3]])
  A_controls = np.concatenate((controls2, A), axis=1)
  X_controls = np.array( [[0.5], [0.5], [0], [0], [0], [-0.2], [0], [0], [3.1], [0], [0]])
  B_controls = A_controls.dot(X_controls)
  B_controls = B_controls.ravel()

  ####NOTE: USE A 1D ARRAY INSTEAD OF A COLUMN VECTOR USING RAVEL. 
  #x_lasso=lassoSolve(A,B, num_iterations = 2000, num_folds = 2)
  #x_lasso2=lasso_force_controls(A,B, [], num_iterations = 2000, num_folds = 2)
  ###work with CV, now try no CV
  x_lasso = lasso_force_controls(A, B_controls, controls2, num_folds=2, perform_cv = False)
  #x_lasso2 = run_choice_method_once(2, A, B_controls, controls=controls2, perform_cv= False, num_folds=10)
  x_lasso3 = lasso_force_controls(A_controls, B_controls, [], num_folds = 2, perform_cv = False)
  print(x_lasso)
  #print(x_lasso2)
  print(x_lasso3)
  
  x_lasso = lasso_force_controls(A, B_controls, controls2, num_folds=2, perform_cv = True)
  #x_lasso2 = run_choice_method_once(2, A, B_controls, controls=controls2, perform_cv= False, num_folds=10)
  x_lasso3 = lasso_force_controls(A_controls, B_controls, [], num_folds = 2, perform_cv = True)
  print(x_lasso)
  #print(x_lasso2)
  print(x_lasso3)
  
    
# =============
# MAIN FUNCTION
# =============
if __name__ == "__main__":
  
  # Perform Tests
  #print("")
  #print("=== Test 1 ===")
  #print("")
  test1()
  #print("")
  #print("=== Test 2 ===")
  #print("")
  #test2() 