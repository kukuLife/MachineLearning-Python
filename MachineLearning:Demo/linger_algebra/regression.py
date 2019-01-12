#regression.py

import numpy as np

def loadDataSet(fileName):
    
    numFeats = len(open(fileName).readline().split('\t')) - 1 #get X num
    print(numFeats)
    X = []
    y = []
    for n in open(fileName).readlines():
        oneLine = n.strip().split('\t')
        lineArr = []
        for i in range(numFeats):
            lineArr.append(float(oneLine[i]))
        X.append(lineArr)
        y.append(float(oneLine[-1]))
    
    return np.mat(X), np.mat(y).T
    
    
def bgd(rate, maxLoop, epsilon, X, y):
    converged = False
    count = 0
    m, n = X.shape
    theta = np.zeros((n,1))
    thetas = {}
    error = float('inf')
    errors = []
    for j in range(n):
        thetas[j] = [theta[j,0]]
    while count < maxLoop:
        count = count + 1
        if converged == True:
            break;
        for j in range(n):
            deriv = ((y - X * theta).T * X[:, j])/m
            theta[j,0] = theta[j,0] + rate * deriv
            thetas[j].append(theta[j, 0])
        error = J(X, y, theta)
        print(error)
        errors.append(error)
        if error < epsilon:
            converged = True
    return thetas, theta, errors
            
def J(X, y, theta):
    m = len(X)
    a = 2 * (y-X*theta).T*(y-X*theta)/m
    return a
    
def sgd(maxLoop, rate, X, y, epsilon):
    m , n = X.shape
    count = 0
    theta = np.zeros((n, 1))
    thetas = {}
    errors = []
    error = float('inf')
    converged = False
    
    for j in range(n):
        thetas[j] = [theta[j, 0]]
        
    while count < maxLoop:
        if converged:
            break
        
        for i in range(m):
            deriv = h(y[i, 0], X[i , :], theta)

            for j in range(n):
                theta[j, 0] = theta[j, 0] + rate*deriv*X[i, j]
                thetas[j].append(theta[j, 0])
                print(deriv)
            error = J(X, y, theta)
            errors.append(error)
            if error < epsilon:
                converged = True
                break;
    return theta,thetas,errors

def h(y, X, theta):
    
    return y - X*theta



def standardize(X):
    m,n = X.shape
    
    for j in range(n):
        features = X[: , j]
        meanVal = features.mean(axis=0)
        std = features.std(axis=0);
        if std == 0:
            X[: , j] = 0
        else :
            X[: , j] = (features - meanVal)/std
    
    return X



def normalize(X):
    m,n = X.shape
    
    for j in range(n):
        features = X[:, j]
        minVal = features.min(axis=0)
        maxVal = features.max(axis=0)
        diff = maxVal - minVal
        if diff == 0:
            X[:,j] = 0
        else:
            X[:,j] = (features - minVal)/diff
    
    return X


def JLwr(theta, X, y, x, c):
    m, n = X.shape
    summerize = 0
    for i in range(m):
        diff = (X[i] - x)*(X[i] - x).T
        w = np.exp(-diff/(2*c*c))
        predictDiff = np.power(y[i] - X[i]*theta,2)
        summerize = summerize + w*predictDiff
    return summerize

def lwr(maxLoop, rate, X, x, y, c):
    m, n = X.shape
    error = float('inf')
    errors = []
    thetas = {}
    theta = np.zeros((n ,1))
    for j in range(n):
        thetas[j] = [theta[j,0]]
    cout = 0
    
    while count <= maxLoop:
        count = count + 1
        if(converged):
            break
        
        for j in range(n):
            deriv = (y - X * theta).T * X[:, j]/m
            theta[j, 0] = theta[j,0] + rate*deriv
            thetas[j] = thetas[j].append(theta[j,0])
        error = JLwr(theta, X, y, x, c)
        erros.append(error[0,0])
        if(error < epsilon):
            converged = True
    return theta, errors, thetas
        
        
        
        
    
    
    