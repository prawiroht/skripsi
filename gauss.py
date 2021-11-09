import numpy as np
import pandas as pd
import math

def findSigma(x,c):
    sigma = math.sqrt(((x-c)**2)/(6.91*2))
    return sigma

def mf(x,c,sigma):
    return (math.e)**((-((x-c)**2))/(2*sigma**2))


# Age
def LowAge(x):
    if x <= 25: return 1
    else: return mf(x,25,findSigma(35,25))
def MidAge(x):
    return mf(x,40,findSigma(55,40))
def HighAge(x):
    if x >= 60: return 1
    else: return mf(x,60,findSigma(80,60))

# Height
def LowHei(x):
    if x <= 140: return 1
    else: return mf(x,140,findSigma(160,140))
def MidHei(x):
    return mf(x,160,findSigma(170,160))
def HighHei(x):
    if x >= 180: return 1
    else: return mf(x,180,findSigma(195,180))

# Weight
def LowWei(x):
    if x <= 40: return 1
    else: return mf(x,40,findSigma(55,40))
def MidWei(x):
    return mf(x,65,findSigma(80,65))
def HighWei(x):
    if x >= 90: return 1
    else: return mf(x,90,findSigma(105,90))

# Systole
def LowSys(x):
    if x <= 90: return 1
    else: return mf(x,90,findSigma(100,90))
def MidSys(x):
    return mf(x,105,findSigma(125,105))
def HighSys(x):
    if x >= 135: return 1
    else: return mf(x,135,findSigma(160,135))

# Diastole
def LowDia(x):
    if x <= 55: return 1
    else: return mf(x,55,findSigma(65,55))
def MidDia(x):
    return mf(x,70,findSigma(85,70))
def HighDia(x):
    if x >= 85: return 1
    else: return mf(x,85,findSigma(95,85))

# Cholesterol
def LowCho(x):
    if x <= 1: return 1
    else: return mf(x,1,findSigma(2,1))
def MidCho(x):
    return mf(x,2,findSigma(3,2))
def HighCho(x):
    if x >= 3: return 1
    else: return mf(x,3,findSigma(4.5,3))

# Glucose
def LowGlu(x):
    if x <= 1: return 1
    else: return mf(x,1,findSigma(2,1))
def MidGlu(x):
    return mf(x,2,findSigma(3,2))
def HighGlu(x):
    if x >= 3: return 1
    else: return mf(x,3,findSigma(4.5,3))

def transform(x):
    x = np.concatenate(
        (LowAge(x[0]),MidAge(x[0]),HighAge(x[0])
        ,x[1]
        ,LowHei(x[2]),MidHei(x[2]),HighHei(x[2])
        ,LowWei(x[3]),MidWei(x[3]),HighWei(x[3])
        ,LowSys(x[4]),MidSys(x[4]),HighSys(x[4])
        ,LowDia(x[5]),MidDia(x[5]),HighDia(x[5])
        ,LowCho(x[6]),MidCho(x[6]),HighCho(x[6])
        ,LowGlu(x[7]),MidGlu(x[7]),HighGlu(x[7])
        ,x[8]
        ,x[9]
        ,x[10]
        ),axis=None)
    return x