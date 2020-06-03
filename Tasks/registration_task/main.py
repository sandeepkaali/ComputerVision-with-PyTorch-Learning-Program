"""
Implement the linear regression model using python and numpy in the following class.
The method fit() should take inputs like,
x = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
"""

import numpy

class LinearRegression:
    def fit(self,X,Y):
      X=np.array(X).reshape(-1,1)
      Y=np.array(Y).reshape(-1,1)
      
      x_shape = X.shape
      
      num_var = x_shape[1]
      weight_matrix = np.random.normal(0,1,(num_var,1))
      intercept = np.random.rand(1)
      for i in range(50):
          dcostdm = np.sum(np.multiply(((np.matmul(X,weight_matrix)+intercept)-Y),X))*2/x_shape[0]
          dcostdc = np.sum(((np.matmul(X,weight_matrix)+intercept)-Y))*2/x_shape[0]
          weight_matrix -= 0.1*dcostdm
          intercept -= 0.1*dcostdc
      return weight_matrix,intercept
  
    def predict(self, X):
      product = np.matmul(np.array(X).reshape(-1,1),self.weight_matrix)+self.intercept
      return product
    
