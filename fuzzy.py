INF = 1e8
import math


def mf(x,a,b,c,d):
    if (x <= a): return 0
    elif(x > a and x <= (a+b)/2): return 2*((x-a)/(b-a))**2
    elif(x > (a+b)/2 and x <= b): return 1-2*((x-b)/(b-a))**2
    elif(x > b and x <= c): return 1
    elif(x > c and x <= (c+d)/2): return 1-2*((x-c)/(d-c))**2
    elif(x > (c+d)/2 and x <= d): return 2*((x-d)/(d-c))**2
    else: return 0
    return 0


# Age
def LowAge(x):
    if x <= 25: return 1
    else: return mf(x,-INF,22.5,27.5,35)
def MidAge(x):
    return mf(x,25,37.5,42.5,55)
def HighAge(x):
    if x >= 60: return 1
    else: return mf(x,40,57.5,62.5,INF)

# Height
def LowHei(x):
    if x <= 140: return 1
    else: return mf(x,-INF,135,145,160)
def MidHei(x):
    return mf(x,150,157,163,170)
def HighHei(x):
    if x >= 180: return 1
    else: return mf(x,165,175,185,INF)

# Weight
def LowWei(x):
    if x <= 40: return 1
    else: return mf(x,-INF,38,42,55)
def MidWei(x):
    return mf(x,50,62,68,80)
def HighWei(x):
    if x >= 90: return 1
    else: return mf(x,75,87,93,INF)

# Systole
def LowSys(x):
    if x <= 90: return 1
    else: return mf(x,-INF,87,93,100)
def MidSys(x):
    return mf(x,85,102,108,125)
def HighSys(x):
    if x >= 135: return 1
    else: return mf(x,110,132,138,INF)

# Diastole
def LowDia(x):
    if x <= 55: return 1
    else: return mf(x,-INF,52,58,65)
def MidDia(x):
    return mf(x,55,67,73,85)
def HighDia(x):
    if x >= 85: return 1
    else: return mf(x,75,83,87,INF)

# Cholesterol
def LowCho(x):
    if x <= 1: return 1
    else: return mf(x,-INF,0.8,1.2,2)
def MidCho(x):
    return mf(x,1,1.75,2.25,3)
def HighCho(x):
    if x >= 3: return 1
    else: return mf(x,1.5,2.75,3.25,INF)

# Glucose
def LowGlu(x):
    if x <= 1: return 1
    else: return mf(x,0,0.8,1.2,2)
def MidGlu(x):
    return mf(x,1,1.75,2.25,3)
def HighGlu(x):
    if x >= 3: return 1
    else: return mf(x,1.5,2.75,3.25,INF)

from matplotlib import pyplot
def plotting(low,mid,high,txt):
    pyplot.plot(low, label='Low')
    pyplot.plot(mid, label='Mid')
    pyplot.plot(high, label='High')
    pyplot.xlabel(txt)
    pyplot.ylabel("Value")
    pyplot.legend(loc="upper right")
    pyplot.title(txt)
    pyplot.show()

lowage = []
midage = []
highage = []
for i in range(160):
    lowage.append(LowAge(i/2))
    midage.append(MidAge(i/2))
    highage.append(HighAge(i/2))
plotting(lowage,midage,highage,"Age")

lowhei = []
midhei = []
highhei = []
for i in range(160):
    lowhei.append(LowHei(i/2+120))
    midhei.append(MidHei(i/2+120))
    highhei.append(HighHei(i/2+120))
plotting(lowhei,midhei,highhei,"Height")

lowwei = []
midwei = []
highwei = []
for i in range(140):
    lowwei.append(LowWei(i/2+30))
    midwei.append(MidWei(i/2+30))
    highwei.append(HighWei(i/2+30))
plotting(lowwei,midwei,highwei,"Weight")

lowsys = []
midsys = []
highsys = []
for i in range(225):
    lowsys.append(LowSys(i/3+75))
    midsys.append(MidSys(i/3+75))
    highsys.append(HighSys(i/3+75))
plotting(lowsys,midsys,highsys,"Systole")

lowdia = []
middia = []
highdia = []
for i in range(120):
    lowdia.append(LowDia(i/3+50))
    middia.append(MidDia(i/3+50))
    highdia.append(HighDia(i/3+50))
plotting(lowdia,middia,highdia,"Diastole")

lowcho = []
midcho = []
highcho = []
for i in range(200):
    lowcho.append(LowCho(i/50))
    midcho.append(MidCho(i/50))
    highcho.append(HighCho(i/50))
plotting(lowcho,midcho,highcho,"Cholesterol")

lowglu = []
midglu = []
highglu = []
for i in range(200):
    lowglu.append(LowGlu(i/50))
    midglu.append(MidGlu(i/50))
    highglu.append(HighGlu(i/50))
plotting(lowglu,midglu,highglu,"Glucose")