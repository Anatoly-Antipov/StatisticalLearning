import numpy as np
from tqdm._tqdm_notebook import tqdm_notebook
import warnings

from Splines import Splines

# Natural cubic splines are used
# To be added: stopping rule for Newton-Raphson method. At the moment, fixed number of iterations (iters=1200) is set for optimization
class NonparametricLogisticRegression:

    def __init__(self, alpha=0.01, l=1, iters=1200):
        
        self.alpha = alpha
        self.l = l
        self.iters = iters
    
    def Check_type(self, X):
        
        # type of X: np.array or np.ndarray
        if type(X) is np.ndarray:
            try:
                # if matrix - ok
                X.shape[1]
            except:
                # if np.array vector - reshape
                X = X.reshape(-1,1)
        else:
            raise NameError('X should be of type np.array or np.ndarray')
        return X
        
    def Dimensionality_check_X(self, X, Y):
        
        if X.shape[0] != Y.shape[0]:
            X = X.T
        return X
    
    def append_unit(self, X):
        
        u = np.ones((X.shape[0],1))
        try:
            # if matrix
            X.shape[1]
            X = np.append(u, X, axis=1)
        except:
            # if np.array
            X = np.append(u, X.reshape(-1,1), axis=1)
        return X
    
    def sigmoid(self, t):
        
        return  1.0 / (1.0 + np.exp(-1.0 * t))

    def CostFunc(self):
        
        self.cost = np.sum(-1*self.Y*np.log(self.Prediction)-(1-self.Y)*np.log(1-self.Prediction),axis=0)
        self.cost += 0.5 * self.l * ( (self.theta.T).dot(self.splines.Omega()) ).dot(self.theta)
        return self.cost
    
    def predict_optimized(self):
        
        self.Prediction = np.dot(self.splines.N(self.X), self.theta)
        self.Prediction = self.sigmoid(self.Prediction)
        return self.Prediction
    
    def predict(self, X=None):
        
        if X is not None:
            self.X = X
        
        self.X = self.Check_type(self.X)
        if self.X.shape[0] <= self.X.shape[1]:
            warnings.warn('X.shape[0] <= X.shape[1]')
        
        self.Prediction = np.dot(self.splines.N(self.X), self.theta)
        self.Prediction = self.sigmoid(self.Prediction)
        return self.Prediction
    
    def UpdateWeight(self):
        
        self.theta = np.linalg.inv( (self.N.T).dot(self.W).dot(self.N) + self.l*self.Omega ).dot( (self.N.T).dot(self.W) ).dot( self.N.dot(self.theta) + np.linalg.inv(self.W).dot(self.Y - self.Prediction) )
        self.theta = self.theta[0]
        
        return True  
    
    def fit(self, X, Y):
        
        self.X = self.Check_type(X)
        self.Y = self.Check_type(Y)
        self.X = self.Dimensionality_check_X(self.X, self.Y)
        
        # calculate matrices for Newton-Raphson method - ESLII p.162 (5.33)
        self.splines = Splines(knots = self.X)
        self.N = self.splines.N(self.X)
        self.Omega = self.splines.integral_of_N2N2(knots = self.X)
        
        # initialize weights
        self.theta = np.zeros(self.N.shape[1]).reshape(-1,1)
        
        # update weights
        for It in tqdm_notebook(range(0,self.iters)):
            
            # calculate matrices for Newton-Raphson method - ESLII p.162 (5.33)
            self.Prediction = self.predict_optimized()
            self.W = np.diag( (self.Prediction*(1-self.Prediction)).reshape(1,-1)[0] )
            # optimization step
            self.UpdateWeight()
