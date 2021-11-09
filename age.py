INF = 1e8
import math
# def mf(x,a,b,c):
#     if (x <= a or x >= c): return 0
#     elif (x > a and x <= b): return (x-a)/(b-a)
#     else: return (c-x)/(c-b)
def findSigma(x,c):
    sigma = math.sqrt(((x-c)**2)/(6.91*2))
    return sigma
def mf(x,a,b,c):
    if (x <= a or x >= c): return 0
    elif (x > a and x <= b): return (x-a)/(b-a)
    else: return (c-x)/(c-b)

# Age
def LowAge(x):
    if x <= 25: return 1
    else: return mf(x,-INF,25,35)
def MidAge(x):
    return mf(x,25,40,55)
def HighAge(x):
    if x >= 60: return 1
    else: return mf(x,40,60,INF)

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
for i in range(400):
    lowage.append(LowAge(i/5))
    midage.append(MidAge(i/5))
    highage.append(HighAge(i/5))

# pyplot.plot(lowage, label='Low')
pyplot.plot(midage, label='Mid')
# pyplot.plot(highage, label='High')
pyplot.plot(25*5, MidAge(25),"go")
# pyplot.plot(37.5*5, MidAge(37.5),"go")
pyplot.plot(40*5, MidAge(40),"go")
# pyplot.plot(42.5*5, MidAge(42.5),"go")
pyplot.plot(55*5, MidAge(55),"go")
pyplot.xlabel("Age")
pyplot.ylabel("Value")
# pyplot.legend(loc="upper right")
pyplot.title("Age")
pyplot.show()