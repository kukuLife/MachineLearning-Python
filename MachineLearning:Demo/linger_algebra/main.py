#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 06:19:52 2018

@author: kukuLife
"""
import numpy as np
import regression

X,y = regression.loadDataSet('ex1.txt')
m, n = X.shape
X = np.concatenate((np.ones((m,1)), X), axis=1)

maxLoop = 1500
epsilon = 0.01
rate = 0.02

theta, thetas, errors = regression.sgd(maxLoop, rate, X, y, epsilon)
