import numpy as np
import pandas as pd
import scipy.integrate as integrate

class Splines:
    
    def __init__(self, knots, linear_tails_share=0.05):
        
        self.all_knots = self.Check_type(knots)
        self.all_knots = np.hstack([self.knots,self.knots-1])
        # linear_tails_share - share of x's to be used for linear splines in tails. Used in automatical definition of knots only - good for 1-dimensional self.knots
#         self.linear_tails_share = linear_tails_share
        # remove linear tails from knots
#         self.knots = self.knots[int(self.linear_tails_share*self.knots.shape[0]):self.knots.shape[0]-int(self.linear_tails_share*self.knots.shape[0])]
        
        
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
            raise NameError('variable should be of type np.array or np.ndarray')
        
        # to avoid np.power(80,5) = -1018167296
        X = X.astype('float')
        
        return X
    
    
    def d(self, k, knots):
        
        self.knots = knots
        self.k = k
        
        return ( ((self.x >= self.knots[self.k-1])*np.power(self.x-self.knots[self.k-1],3)).astype('float') - ((self.x >= self.knots[len(self.knots)-1])*np.power(self.x-self.knots[len(self.knots)-1],3)).astype('float') )/\
        ( self.knots[self.k-1] - self.knots[len(self.knots)-1] )
    
    def N(self, x, knots):
        
        self.knots = knots
        self.x = self.Check_type(x)
        self.X = np.hstack((np.ones(shape=self.x.shape),self.x))
        
        for i in range(len(self.knots)-2):
            self.X_ = self.d(i+1, self.knots) - self.d(len(self.knots)-1, self.knots)
            self.X = np.hstack((self.X,self.X_))
        
        return self.X

    # function for Omega2() and Omega3()
    def integral_of_1(self, l, u):
        
        # function calculates (in np.array form):
        # integral_(lower_limit)^(upper_limit)[ 1 ] dX = X |_(lower_limit)^(upper_limit)
        
        return u - l
    
    # function for Omega3()
    def integral_of_X(self, l, u):
        
        # function calculates (in np.array form):
        # integral_(lower_limit)^(upper_limit)[ X ] dX = X^2/2 |_(lower_limit)^(upper_limit)
        
        return (u)**2/2 - (l)**2/2
    
    # function for Omega3()
    def integral_of_X2(self, l, u):
        
        # function calculates (in np.array form):
        # integral_(lower_limit)^(upper_limit)[ X^2 ] dX = X^3/3 |_(lower_limit)^(upper_limit)
        
        return (u)**3/3 - (l)**3/3
    
    # function for Omega3()
    def integral_of_N(self, l, u, xi_j2, xi_m, xi_m_):
        
        # function calculates (in np.array form):
        # integral_(lower_limit)^(upper_limit)[ N(X) ] dX = ... |_(lower_limit)^(upper_limit)
        
        return (1/4) * ( ( (u-xi_j2)**4 - (np.maximum(xi_j2,l)-xi_j2)**4 - (u-xi_m)**4 + (np.maximum(xi_m,l)-xi_m)**4 )/(xi_m-xi_j2) - ( (u-xi_m_)**4 - (np.maximum(xi_m_,l)-xi_m_)**4 - (u-xi_m)**4 + (np.maximum(xi_m,l)-xi_m)**4 )/(xi_m-xi_m_) )
    
    # function for Omega3()
    def integral_of_XN(self, l, u, xi_j2, xi_m, xi_m_):
        
        # function calculates (in np.array form):
        # integral_(lower_limit)^(upper_limit)[ N(X) ] dX = ... |_(lower_limit)^(upper_limit)
        
        return ( ((u**5/5-(3/4)*(u**4)*xi_j2 + (u**3)*(xi_j2**2) - (1/2)*(u**2)*(xi_j2**3)) - (np.maximum(xi_j2,l)**5/5-(3/4)*(np.maximum(xi_j2,l)**4)*xi_j2 + (np.maximum(xi_j2,l)**3)*(xi_j2**2) - (1/2)*(np.maximum(xi_j2,l)**2)*(xi_j2**3))) - \
                ((u**5/5-(3/4)*(u**4)*xi_m + (u**3)*(xi_m**2) - (1/2)*(u**2)*(xi_m**3)) - (np.maximum(xi_m,l)**5/5-(3/4)*(np.maximum(xi_m,l)**4)*xi_m + (np.maximum(xi_m,l)**3)*(xi_m**2) - (1/2)*(np.maximum(xi_m,l)**2)*(xi_m**3))) ) / \
                (xi_m-xi_j2) - \
                ( ((u**5/5-(3/4)*(u**4)*xi_m_ + (u**3)*(xi_m_**2) - (1/2)*(u**2)*(xi_m_**3)) - (np.maximum(xi_m_,l)**5/5-(3/4)*(np.maximum(xi_m_,l)**4)*xi_m_ + (np.maximum(xi_m_,l)**3)*(xi_m_**2) - (1/2)*(np.maximum(xi_m_,l)**2)*(xi_m_**3))) - \
                ((u**5/5-(3/4)*(u**4)*xi_m + (u**3)*(xi_m**2) - (1/2)*(u**2)*(xi_m**3)) - (np.maximum(xi_m,l)**5/5-(3/4)*(np.maximum(xi_m,l)**4)*xi_m + (np.maximum(xi_m,l)**3)*(xi_m**2) - (1/2)*(np.maximum(xi_m,l)**2)*(xi_m**3))) ) / \
                (xi_m-xi_m_)
    
    # function for Omega2()
    def integral_of_N1(self, l, u, xi_j2, xi_m, xi_m_):
        
        # N(X) - see ESLII p.145 (5.4-5.5)
        # function calculates (in np.array form):
        # integral_(lower_limit)^(upper_limit)[ N'(X) ] dX = ( (X-xi_j2)^3 - (X-xi_m)^3 )/(xi_m-xi_j2) - ( (X-xi_m_)^3 - (X-xi_m_)^3) )/(xi_m-xi_m_) |_(lower_limit)^(upper_limit)
        
        return ( (u-xi_j2)**3 - (np.maximum(xi_j2,l)-xi_j2)**3 - (u-xi_m)**3 + (np.maximum(xi_m,l)-xi_m)**3 )/(xi_m-xi_j2) - ( (u-xi_m_)**3 - (np.maximum(xi_m_,l)-xi_m_)**3 - (u-xi_m)**3 + (np.maximum(xi_m,l)-xi_m)**3 )/(xi_m-xi_m_)
    
    # function for Omega()
    def integral_of_product(self, lower_limit, upper_limit, a, b):
        
        # function calculates (in matrix form):
        # integral_(lower_limit)^(upper_limit)[ (X-a)_+ * (X-b)_+ ] dX =
        # = [X^3/3 - (a+b) * X^2/2 + a*b*X] |_(max(a,b,lower_limit))^(upper_limit)
        
        return (upper_limit)**3/3 - (np.maximum(np.maximum(a,b),lower_limit))**3 / 3 - (a+b)*(upper_limit)**2/2 + (a+b)*(np.maximum(np.maximum(a,b),lower_limit))**2/2 + a*b*(upper_limit) - a*b*np.maximum(np.maximum(a,b),lower_limit)
    
    # function for Omega2()
    def integral_of_product2(self, lower_limit, upper_limit, a, b):
        
        # function calculates (in matrix form):
        # integral_(lower_limit)^(upper_limit)[ (X-a)^2_+ * (X-b)^2_+ ] dX =
        # = [X^5/5 - ((a+b)/2)*X^4 + ((a^2+4ab+b^2)/3)*X^3 - ab(a+b)*X^2 + a^2b^2*X] |_(max(a,b,lower_limit))^(upper_limit)
        
        return (upper_limit)**5/5 - (np.maximum(np.maximum(a,b),lower_limit))**5/5 - ((a+b)/2)*(upper_limit)**4 + ((a+b)/2)*(np.maximum(np.maximum(a,b),lower_limit))**4 + ((a**2+4*a*b+b**2)/3)*(upper_limit)**3 - ((a**2+4*a*b+b**2)/3)*(np.maximum(np.maximum(a,b),lower_limit))**3 - a*b*(a+b)*(upper_limit)**2 + a*b*(a+b)*(np.maximum(np.maximum(a,b),lower_limit))**2 + (a**2)*(b**2)*(upper_limit) - (a**2)*(b**2)*(np.maximum(np.maximum(a,b),lower_limit))
    
    # function for Omega3()
    def integral_of_product3(self, lower_limit, upper_limit, a, b):
        
        # function calculates (in matrix form):
        # integral_(lower_limit)^(upper_limit)[ (X-a)^3_+ * (X-b)^3_+ ] dX =
        # = [X^7/7 - (1/2)(a+b)*X^6+(3/5)(a^2+3ab+b^2)*X^5 - (1/4)(a^3+9a^2b+9ab^2+b^3)*X^4 + ab(a^2+3ab+b^2)*X^3 - (3/2)a^2b^2(a+b)*X^2 + a^3b^3*X] |_(max(a,b,lower_limit))^(upper_limit)
        
        return (upper_limit)**7/7 - (np.maximum(np.maximum(a,b),lower_limit))**7/7 - (1/2)*(a+b)*(upper_limit)**6 + (1/2)*(a+b)*(np.maximum(np.maximum(a,b),lower_limit))**6 + (3/5)*(a**2+3*a*b+b**2)*(upper_limit)**5 - (3/5)*(a**2+3*a*b+b**2)*(np.maximum(np.maximum(a,b),lower_limit))**5 - (1/4)*(a**3+9*(a**2)*b+9*a*b**2+b**3)*(upper_limit)**4 + (1/4)*(a**3+9*(a**2)*b+9*a*b**2+b**3)*(np.maximum(np.maximum(a,b),lower_limit))**4 + a*b*(a**2+3*a*b+b**2)*(upper_limit)**3 - a*b*(a**2+3*a*b+b**2)*(np.maximum(np.maximum(a,b),lower_limit))**3 - (3/2)*(a**2)*(b**2)*(a+b)*(upper_limit)**2 + (3/2)*(a**2)*(b**2)*(a+b)*(np.maximum(np.maximum(a,b),lower_limit))**2 + (a**3)*(b**3)*(upper_limit) - (a**3)*(b**3)*(np.maximum(np.maximum(a,b),lower_limit))
    
    
    # Calculates matrix Omega for natural cubic splines - ESLII p.152, (5.11-5.12)
    def integral_of_N2N2(self,knots):
        
        self.knots = knots
        
        # transform initial parameter to matrix form for fast computations
        # a ~ xi_(j-2), b ~ xi_(k-2), m ~ xi_M, m_ ~ xi_(M-1) - ESLII p.145, (5.4-5.5)
        # " - 2", because of ESLII p.145, formula (5.4). Additionally, first two rows and first two columns of Omega are zeros (product of second derivatives of linear functions)
        self.lower_limit = self.knots.min(axis=0)
        self.lower_limit = np.array([[[self.lower_limit[k] for i in range(self.knots.shape[0] - 2)] for j in range(self.knots.shape[0] - 2)] for k in range(self.knots.shape[1])])
        self.upper_limit = self.knots.max(axis=0)
        self.upper_limit = np.array([[[self.upper_limit[k] for i in range(self.knots.shape[0] - 2)] for j in range(self.knots.shape[0] - 2)] for k in range(self.knots.shape[1])])
        self.m = np.array([[[self.knots[-1][k] for i in range(self.knots.shape[0] - 2)] for j in range(self.knots.shape[0] - 2)] for k in range(self.knots.shape[1])])
        self.m_ = np.array([[[self.knots[-2][k] for i in range(self.knots.shape[0] - 2)] for j in range(self.knots.shape[0] - 2)] for k in range(self.knots.shape[1])])
        self.a = np.array([np.hstack([self.knots[:-2][:,k].reshape(-1,1) for i in range(self.knots.shape[0] - 2)]) for k in range(self.knots.shape[1])])
        self.b = np.array([np.vstack([self.knots[:-2][:,k] for i in range(self.knots.shape[0] - 2)]) for k in range(self.knots.shape[1])])
        
        # submatrix ([2:,2:]) of integral of N''.T x N''
        self.integral_of_N2N2_matrix = 36 * (\
        ( self.integral_of_product(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.a,b=self.b) - \
         self.integral_of_product(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.a,b=self.m) - \
         self.integral_of_product(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.m,b=self.b) + \
         self.integral_of_product(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.m,b=self.m) ) /\
        ( (self.m-self.a)*(self.m-self.b) ) - \
        ( self.integral_of_product(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.m_,b=self.b) - \
         self.integral_of_product(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.m_,b=self.m) - \
         self.integral_of_product(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.m,b=self.b) + \
         self.integral_of_product(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.m,b=self.m) ) /\
        ( (self.m-self.m_)*(self.m-self.b) ) - \
        ( self.integral_of_product(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.a,b=self.m_) - \
         self.integral_of_product(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.a,b=self.m) - \
         self.integral_of_product(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.m,b=self.m_) + \
         self.integral_of_product(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.m,b=self.m) ) /\
        ( (self.m-self.a)*(self.m-self.m_) ) + \
        ( self.integral_of_product(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.m_,b=self.m_) - \
         self.integral_of_product(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.m_,b=self.m) - \
         self.integral_of_product(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.m,b=self.m_) + \
         self.integral_of_product(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.m,b=self.m) ) /\
        ( (self.m-self.m_)*(self.m-self.m_) ) \
        )
        
        # add 2 zero-rows 
        self.integral_of_N2N2_matrix = np.array([np.vstack([np.zeros(self.integral_of_N2N2_matrix.shape[1]),np.zeros(self.integral_of_N2N2_matrix.shape[1]),self.integral_of_N2N2_matrix[k]]) for k in range(self.integral_of_N2N2_matrix.shape[0])])
        # add 2 zero-columns
        self.integral_of_N2N2_matrix = np.array([np.hstack([np.zeros(self.integral_of_N2N2_matrix.shape[1]).reshape(-1,1),np.zeros(self.integral_of_N2N2_matrix.shape[1]).reshape(-1,1),self.integral_of_N2N2_matrix[k]]) for k in range(self.integral_of_N2N2_matrix.shape[0])])
        
        return self.integral_of_N2N2_matrix
    
    
    # Calculates Omega-like matrix for natural cubic splines - ESLII p.152, (5.11-5.12)
    def integral_of_N1N1(self, knots):
        
        self.knots = knots
        
        # transform initial parameter to np.array form to find first two rows and first two columns of Omega2
        self.l = self.knots.min(axis=0)
        self.l = np.array([[self.l[k] for i in range(self.knots.shape[0] - 2)] for k in range(self.knots.shape[1])])
        self.u = self.knots.max(axis=0)
        self.u = np.array([[self.u[k] for i in range(self.knots.shape[0] - 2)] for k in range(self.knots.shape[1])])
        self.xi_j2 = self.knots[:-2].T
        self.xi_m = np.array([[self.knots[-1][k] for i in range(self.knots.shape[0] - 2)] for k in range(self.knots.shape[1])])
        self.xi_m_ = np.array([[self.knots[-2][k] for i in range(self.knots.shape[0] - 2)] for k in range(self.knots.shape[1])])
        
        
        # transform initial parameter to matrix form for fast computations
        # a ~ xi_(j-2), b ~ xi_(k-2), m ~ xi_M, m_ ~ xi_(M-1) - ESLII p.145, (5.4-5.5)
        # " - 2", because of ESLII p.145, formula (5.4)
        self.lower_limit = self.knots.min(axis=0)
        self.lower_limit = np.array([[[self.lower_limit[k] for i in range(self.knots.shape[0] - 2)] for j in range(self.knots.shape[0] - 2)] for k in range(self.knots.shape[1])])
        self.upper_limit = self.knots.max(axis=0)
        self.upper_limit = np.array([[[self.upper_limit[k] for i in range(self.knots.shape[0] - 2)] for j in range(self.knots.shape[0] - 2)] for k in range(self.knots.shape[1])])
        self.m = np.array([[[self.knots[-1][k] for i in range(self.knots.shape[0] - 2)] for j in range(self.knots.shape[0] - 2)] for k in range(self.knots.shape[1])])
        self.m_ = np.array([[[self.knots[-2][k] for i in range(self.knots.shape[0] - 2)] for j in range(self.knots.shape[0] - 2)] for k in range(self.knots.shape[1])])
        self.a = np.array([np.hstack([self.knots[:-2][:,k].reshape(-1,1) for i in range(self.knots.shape[0] - 2)]) for k in range(self.knots.shape[1])])
        self.b = np.array([np.vstack([self.knots[:-2][:,k] for i in range(self.knots.shape[0] - 2)]) for k in range(self.knots.shape[1])])
        
        # submatrix ([2:,2:]) of integral of N'.T x N'
        self.integral_of_N1N1_matrix = 9 * (\
        ( self.integral_of_product2(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.a,b=self.b) - \
         self.integral_of_product2(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.a,b=self.m) - \
         self.integral_of_product2(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.m,b=self.b) + \
         self.integral_of_product2(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.m,b=self.m) ) /\
        ( (self.m-self.a)*(self.m-self.b) ) - \
        ( self.integral_of_product2(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.m_,b=self.b) - \
         self.integral_of_product2(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.m_,b=self.m) - \
         self.integral_of_product2(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.m,b=self.b) + \
         self.integral_of_product2(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.m,b=self.m) ) /\
        ( (self.m-self.m_)*(self.m-self.b) ) - \
        ( self.integral_of_product2(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.a,b=self.m_) - \
         self.integral_of_product2(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.a,b=self.m) - \
         self.integral_of_product2(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.m,b=self.m_) + \
         self.integral_of_product2(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.m,b=self.m) ) /\
        ( (self.m-self.a)*(self.m-self.m_) ) + \
        ( self.integral_of_product2(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.m_,b=self.m_) - \
         self.integral_of_product2(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.m_,b=self.m) - \
         self.integral_of_product2(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.m,b=self.m_) + \
         self.integral_of_product2(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.m,b=self.m) ) /\
        ( (self.m-self.m_)*(self.m-self.m_) ) \
        )
        
        # add 1 row with integral of the first derivative of N
        self.integral_of_N1N1_matrix = np.array([np.vstack([self.integral_of_N1(self.l, self.u, self.xi_j2, self.xi_m, self.xi_m_)[k],self.integral_of_N1N1_matrix[k]]) for k in range(self.integral_of_N1N1_matrix.shape[0])])
        # diagonal element from the second column is an integral of 1
        self.second_column = np.array([np.vstack([self.integral_of_1(self.l, self.u)[k][0], self.integral_of_N1(self.l, self.u, self.xi_j2, self.xi_m, self.xi_m_).T[:,k].reshape(-1,1)]) for k in range(self.integral_of_N1N1_matrix.shape[0])])
        # add 1 column integral of the first derivative of N
        self.integral_of_N1N1_matrix = np.array([np.hstack([self.second_column[k],self.integral_of_N1N1_matrix[k]]) for k in range(self.integral_of_N1N1_matrix.shape[0])])
        
        # add 1 zero-row
        self.integral_of_N1N1_matrix = np.array([np.vstack([np.zeros(self.integral_of_N1N1_matrix.shape[1]),self.integral_of_N1N1_matrix[k]]) for k in range(self.integral_of_N1N1_matrix.shape[0])])
        # add 1 zero-column
        self.integral_of_N1N1_matrix = np.array([np.hstack([np.zeros(self.integral_of_N1N1_matrix.shape[1]).reshape(-1,1),self.integral_of_N1N1_matrix[k]]) for k in range(self.integral_of_N1N1_matrix.shape[0])])
        
        return self.integral_of_N1N1_matrix
    
    
    # Calculates Omega-like matrix for natural cubic splines - ESLII p.152, (5.11-5.12)
    def integral_of_NN(self, knots):
        
        self.knots = knots
        
        # transform initial parameter to np.array form to find first two rows and first two columns of Omega3
        self.l = self.knots.min(axis=0)
        self.l = np.array([[self.l[k] for i in range(self.knots.shape[0] - 2)] for k in range(self.knots.shape[1])])
        self.u = self.knots.max(axis=0)
        self.u = np.array([[self.u[k] for i in range(self.knots.shape[0] - 2)] for k in range(self.knots.shape[1])])
        self.xi_j2 = self.knots[:-2].T
        self.xi_m = np.array([[self.knots[-1][k] for i in range(self.knots.shape[0] - 2)] for k in range(self.knots.shape[1])])
        self.xi_m_ = np.array([[self.knots[-2][k] for i in range(self.knots.shape[0] - 2)] for k in range(self.knots.shape[1])])
        
        # transform initial parameter to matrix form for fast computations
        # a ~ xi_(j-2), b ~ xi_(k-2), m ~ xi_M, m_ ~ xi_(M-1) - ESLII p.145, (5.4-5.5)
        # " - 2", because of ESLII p.145, formula (5.4)
        self.lower_limit = self.knots.min(axis=0)
        self.lower_limit = np.array([[[self.lower_limit[k] for i in range(self.knots.shape[0] - 2)] for j in range(self.knots.shape[0] - 2)] for k in range(self.knots.shape[1])])
        self.upper_limit = self.knots.max(axis=0)
        self.upper_limit = np.array([[[self.upper_limit[k] for i in range(self.knots.shape[0] - 2)] for j in range(self.knots.shape[0] - 2)] for k in range(self.knots.shape[1])])
        self.m = np.array([[[self.knots[-1][k] for i in range(self.knots.shape[0] - 2)] for j in range(self.knots.shape[0] - 2)] for k in range(self.knots.shape[1])])
        self.m_ = np.array([[[self.knots[-2][k] for i in range(self.knots.shape[0] - 2)] for j in range(self.knots.shape[0] - 2)] for k in range(self.knots.shape[1])])
        self.a = np.array([np.hstack([self.knots[:-2][:,k].reshape(-1,1) for i in range(self.knots.shape[0] - 2)]) for k in range(self.knots.shape[1])])
        self.b = np.array([np.vstack([self.knots[:-2][:,k] for i in range(self.knots.shape[0] - 2)]) for k in range(self.knots.shape[1])])
        
        # submatrix ([2:,2:]) of integral of N.T x N
        self.integral_of_NN_matrix = \
        ( self.integral_of_product3(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.a,b=self.b) - \
         self.integral_of_product3(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.a,b=self.m) - \
         self.integral_of_product3(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.m,b=self.b) + \
         self.integral_of_product3(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.m,b=self.m) ) /\
        ( (self.m-self.a)*(self.m-self.b) ) - \
        ( self.integral_of_product3(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.m_,b=self.b) - \
         self.integral_of_product3(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.m_,b=self.m) - \
         self.integral_of_product3(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.m,b=self.b) + \
         self.integral_of_product3(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.m,b=self.m) ) /\
        ( (self.m-self.m_)*(self.m-self.b) ) - \
        ( self.integral_of_product3(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.a,b=self.m_) - \
         self.integral_of_product3(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.a,b=self.m) - \
         self.integral_of_product3(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.m,b=self.m_) + \
         self.integral_of_product3(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.m,b=self.m) ) /\
        ( (self.m-self.a)*(self.m-self.m_) ) + \
        ( self.integral_of_product3(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.m_,b=self.m_) - \
         self.integral_of_product3(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.m_,b=self.m) - \
         self.integral_of_product3(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.m_,b=self.m) + \
         self.integral_of_product3(lower_limit=self.lower_limit,upper_limit=self.upper_limit,a=self.m,b=self.m) ) /\
        ( (self.m-self.m_)*(self.m-self.m_) )
        
        # add 2 rows to Omega[:2,2:] with integral of N and integral of X*N 
        self.integral_of_NN_matrix = np.array([np.vstack([self.integral_of_N(self.l, self.u, self.xi_j2, self.xi_m, self.xi_m_)[k],self.integral_of_XN(self.l, self.u, self.xi_j2, self.xi_m, self.xi_m_)[k],self.integral_of_NN_matrix[k]]) for k in range(self.integral_of_NN_matrix.shape[0])])
        # elements from Omega[:2,:2] are [[int_1,int_X],[int_X,intX^2]]. Make Omega[:,:2] and stack it:
        self.first_column = np.array([np.vstack([self.integral_of_1(self.l, self.u)[k][0], self.integral_of_X(self.l, self.u)[k][0], self.integral_of_N(self.l, self.u, self.xi_j2, self.xi_m, self.xi_m_).T[:,k].reshape(-1,1)]) for k in range(self.integral_of_NN_matrix.shape[0])])
        self.second_column = np.array([np.vstack([self.integral_of_X(self.l, self.u)[k][0], self.integral_of_X2(self.l, self.u)[k][0], self.integral_of_XN(self.l, self.u, self.xi_j2, self.xi_m, self.xi_m_).T[:,k].reshape(-1,1)]) for k in range(self.integral_of_NN_matrix.shape[0])])
        # add 2 columns
        self.integral_of_NN_matrix = np.array([np.hstack([self.first_column[k], self.second_column[k], self.integral_of_NN_matrix[k]]) for k in range(self.integral_of_NN_matrix.shape[0])])
        
        return self.integral_of_NN_matrix
    
#     def O(p1,p2,a1,a2,knots)