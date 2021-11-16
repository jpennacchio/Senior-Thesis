import numpy as np
from cvxopt import matrix, solvers

def lpSolve(A,b):

  # Get Problem Size
  numRows = A.shape[0]
  numCols = A.shape[1]

  # Set Linear Cost Function
  lpc = matrix(np.ones(2*numCols))
  # Set G Matrix to minus the identity matrix
  lpG = matrix(-1.0*np.eye(2*numCols))
  # Set h matrix to all zeros
  lph = matrix(np.zeros(2*numCols))
  # Set the matrix A
  lpA = matrix(np.column_stack((A,-A)))
  # Set the rhs b
  lpb = matrix(b)

  # Solve Linear Program with CVX
  solvers.options['show_progress'] = False
  sol = np.array(solvers.lp(lpc, lpG, lph, lpA, lpb)['x']).ravel()

  # Assemble the sparse Solution
  res = sol[:numCols] - sol[numCols:] 
  return res



def test1():
  """
  Test on slightly larger matrix (5 observations * 10 predictors)
  """
  # Create test observations:
  A = np.array([[1, 0.9, 0.8, 0.2, -0.7, 1.3, 2.3, -0.5, 1.4, -0.2],[1, -0.1, 0.7, 0.3, -0.2, 0.9, 0.7, 1.3, 1.8, -2.5], [1,0.5,0.4,0.3,0.9,1.9, 2.3, -3.1, -2.9, -0.7], [1,0.2,0.9,0.1,0.7,2.3, 0.5, 2.5, -1.5, -2.9], [1,0.5,1,1.2,1.5,-0.1, 0.3, 0.7, 1.9, -0.6]])
  X = np.array( [[0.5], [0], [0], [0], [-0.2], [0], [0], [3.1], [0], [0]])
  B = A.dot(X)
  x_lp=lpSolve(A,B)
  
    
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

