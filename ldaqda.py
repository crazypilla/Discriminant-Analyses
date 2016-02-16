
# coding: utf-8

# In[14]:

import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import scipy.io
import matplotlib.pyplot as plt
import pickle

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD
    
    return means,covmat


# In[15]:

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    
    return means,covmats


# In[16]:

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    
    # IMPLEMENT THIS METHOD
    return acc


# In[17]:

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    
    # IMPLEMENT THIS METHOD
    return acc


# In[18]:

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1                                                                
    # IMPLEMENT THIS METHOD     
    B=np.dot(X.transpose(),X)     
    C=np.linalg.inv(B) 
    D = np.dot(C,X.transpose())
    w= np.dot(D,y)                                        
    return w


# In[19]:

def learnRidgeRegression(X,y,lambd):

    



    # add intercept
    #print y.shape
   

    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD 
    samples = X.shape[0] 
       
    I = np.identity(X.shape[1])

    X_trans = np.transpose(X)                            
    X_sqr = np.dot(X_trans,X)                   
    right_inter = np.dot(X_trans,y)             
   
    left_inter1 = samples*lambd*I
    left_inter = np.add(left_inter1,X_sqr)
    w = np.dot(np.linalg.inv(left_inter),right_inter)
    
    return w



# In[20]:

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse
    
    # IMPLEMENT THIS METHOD
   # print len(ytest)
   # print np.shape(ytest)
   # print np.shape(Xtest)
    actual_op = np.dot(Xtest,w)
    err = ytest - actual_op
    
    rmse = np.divide(np.sqrt(np.sum(np.square(err))),len(ytest))
    return rmse


# In[21]:

def regressionObjVal(w, X, y, lambd):
    # Main script
       # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD 

    samples = X.shape[0]


    w1 = np.reshape(w,(w.shape[0],1))




    w_transpose = np.transpose(w1)
    x_transpose =  np.transpose(X)



    oneBytwoN = 1.0/(2*samples)
    oneByN = 1.0/samples


    xW = np.dot(X,w)
    xWshaped = np.reshape(xW,(xW.shape[0],1))

    errorleft = np.subtract(y, xWshaped)

    errorlefttranspose = np.transpose(errorleft)


    errordot = np.dot(errorlefttranspose,errorleft)

    errorbeforeflatten = oneBytwoN * errordot

    errorleft = np.ndarray.flatten(errorbeforeflatten)
    errorright = 0.5 * lambd * np.ndarray.flatten(np.dot(w_transpose,w))
    error = errorleft + errorright


    y_transpose = np.transpose(y)

    yTx = np.dot(y_transpose, X)

    xTx = np.dot(x_transpose,X)

    wTXX = np.dot(w_transpose, xTx)


    righterrgrad = oneByN * np.subtract(wTXX,yTx)
    righttrans = np.transpose(righterrgrad)

    errGradright = lambd * w_transpose

    error_grad1 = righterrgrad + errGradright
    error_grad = error_grad1.flatten()





    return error, error_grad


# In[22]:

def mapNonLinear(x,p):
# Inputs:
# x - a single column vector (N x 1)
# p - integer (>= 0)
# Outputs:
# Xd - (N x (d+1))
# IMPLEMENT THIS METHOD

    a = range(p+1)
    b=np.zeros((np.shape(x)[0],p+1))
    for i in range(np.shape(x)[0]):
        b[i:] =np.power(x[i],a)

    Xd=b
    return Xd


# In[23]:

# Main script

# Problem 1
# load the sample data                                                                 
X,y,Xtest,ytest = pickle.load(open('C:\Users\Amair\Desktop\MachineLearning\hw2\sample.pickle','rb'))            



       

# LDA
#means,covmat = ldaLearn(X,y)
#ldaacc = ldaTest(means,covmat,Xtest,ytest)
#print('LDA Accuracy = '+str(ldaacc))
# QDA
#means,covmats = qdaLearn(X,y)
#qdaacc = qdaTest(means,covmats,Xtest,ytest)
#print('QDA Accuracy = '+str(qdaacc))

# Problem 2

X,y,Xtest,ytest = pickle.load(open('C:\Users\Amair\Desktop\MachineLearning\hw2\diabetes.pickle','rb'))    
# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses3 = np.zeros((k,1))
rmses31 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    rmses31[i] = testOLERegression(w_l,X_i,y)
    i = i + 1
plt.plot(lambdas,rmses3)

#plt.plot(lambdas,rmses31)

# Problem 4
k = 101
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.zeros((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l_1 = np.zeros((X_i.shape[1],1))
    for j in range(len(w_l.x)):
        w_l_1[j] = w_l.x[j]
    rmses4[i] = testOLERegression(w_l_1,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses4)
plt.show()

# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
plt.plot(range(pmax),rmses5)
plt.legend(('No Regularization','Regularization'))
plt.show()


# In[ ]:



