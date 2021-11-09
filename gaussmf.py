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
i = 0
X_fuzz = np.empty((70000,25)) 
for x in X:
    x[0] /= 365
    x[1] -= 1
    x = transform(x)
    X_fuzz[i] = x
    i += 1    
print(X_fuzz.shape)
print(X_fuzz[0])

# Split Data
Xtrain,Xtest,ytrain,ytest = train_test_split(X_fuzz,y,test_size=0.25)

# PCA
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler

sclr = StandardScaler()
Xtrain_std = sclr.fit_transform(Xtrain)
Xtest_std = sclr.fit_transform(Xtest)

pca = decomposition.PCA(n_components=11)
Xtrain = pca.fit_transform(Xtrain_std)
Xtest = pca.fit_transform(Xtest_std)

# ANN
#create & initialization model
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam_v2,gradient_descent_v2
from keras import backend as K
def RMSE(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

inp = 11
out = 1
hid = math.floor(math.sqrt(inp*out))
model = Sequential()
model.add(Dense(hid,input_dim=inp,activation="sigmoid"))
model.add(Dense(out,activation="sigmoid"))
opt = gradient_descent_v2.SGD(learning_rate=0.001)
model.compile(
    optimizer=opt, 
    loss=RMSE, 
    metrics=['accuracy',f1_m,precision_m,recall_m])

#train model
history = model.fit(Xtrain, ytrain, validation_data=(Xtest, ytest), epochs=100, batch_size=10)

model.save_weights("gaussmf_model.h5")

from matplotlib import pyplot
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.xlabel("epoch")
pyplot.ylabel("Accuracy")
pyplot.legend(loc="upper right")
pyplot.title("Accuracy")
pyplot.show()

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.xlabel("epoch")
pyplot.ylabel("Loss")
pyplot.legend(loc="upper right")
pyplot.title("Loss")
pyplot.show()

#scores
loss,acc,fscore,precision,recall = model.evaluate(Xtest,ytest,verbose=0)
print("Accuracy =",acc)
print("Precision =",precision)
print("Recall =",recall)
print("F-score =",fscore)