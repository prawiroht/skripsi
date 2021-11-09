# input data
#input data
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
dataset = pd.read_csv('data/cardio.csv',sep=";")
X = dataset.values[:,1:12] #Fitur
y = dataset.values[:,12]   #Target
INF = 1e8

# fuzzy membership functions

#Triangular
def trimf(x,a,b,c):
    if (x <= a or x >= c): return 0
    elif (x > a and x <= b): return (x-a)/(b-a)
    else: return (c-x)/(c-b)

# Trapezoidal
def trapmf(x,a,b,c,d):
    return max(min((x-a)/(b-a),1,(d-x)/(d-c)),0)

# Pi-Shaped
def pimf(x,a,b,c,d):
    if (x <= a): return 0
    elif(x > a and x <= (a+b)/2): return 2*((x-a)/(b-a))**2
    elif(x > (a+b)/2 and x <= b): return 1-2*((x-b)/(b-a))**2
    elif(x > b and x <= c): return 1
    elif(x > c and x <= (c+d)/2): return 1-2*((x-c)/(d-c))**2
    elif(x > (c+d)/2 and x <= d): return 2*((x-d)/(d-c))**2
    else: return 0

# Gaussian
def findSigma(x,m):
    sigma = math.sqrt(((x-m)**2)/(6.91*2))
    return sigma
def gaussmf(x,m,sigma):
    return (math.e)**((-((x-m)**2))/(2*findSigma(x,m)**2))
