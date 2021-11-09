import gauss
import pi
import tri
import trap
#input data

import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
dataset = pd.read_csv('data/cardio.csv',sep=";")
X = dataset.values[:,1:12] #Fitur
y = dataset.values[:,12]   #Target
contoh_y = y[0:20]
y = y[20:]
INF = 1e8
print(contoh_y)

i = 0
contoh_mlp_x = np.empty((20,11)) 
contoh_gauss_x = np.empty((20,25)) 
contoh_pi_x = np.empty((20,25))
contoh_tri_x = np.empty((20,25))
contoh_trap_x = np.empty((20,25))
X_MLP = np.empty((69980,11))
X_gauss = np.empty((69980,25)) 
X_pi = np.empty((69980,25)) 
X_tri = np.empty((69980,25)) 
X_trap = np.empty((69980,25)) 
for x in X:
    x[0] /= 365
    x[1] -= 1
    if (i < 20):
        contoh_mlp_x[i] = x
        contoh_gauss_x[i] = gauss.transform(x)
        contoh_pi_x[i] = pi.transform(x)
        contoh_tri_x[i] = tri.transform(x)
        contoh_trap_x[i] = trap.transform(x)
    else:    
        X_MLP[i-20] = x
        X_gauss[i-20] = gauss.transform(x)
        X_pi[i-20] = pi.transform(x)
        X_tri[i-20] = tri.transform(x)
        X_trap[i-20] = trap.transform(x)
    i += 1    
# Split Data
Xtrain0,Xtest0,ytrain0,ytest0 = train_test_split(X_MLP,y,test_size=0.25)
Xtrain1,Xtest1,ytrain1,ytest1 = train_test_split(X_gauss,y,test_size=0.25)
Xtrain2,Xtest2,ytrain2,ytest2 = train_test_split(X_pi,y,test_size=0.25)
Xtrain3,Xtest3,ytrain3,ytest3 = train_test_split(X_tri,y,test_size=0.25)
Xtrain4,Xtest4,ytrain4,ytest4 = train_test_split(X_trap,y,test_size=0.25)

# PCA
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
def PCA(Xtrain,Xtest):
    sclr = StandardScaler()
    Xtrain_std = sclr.fit_transform(Xtrain)
    Xtest_std = sclr.fit_transform(Xtest)

    pca = decomposition.PCA(n_components=11)
    Xtrain = pca.fit_transform(Xtrain_std)
    Xtest = pca.fit_transform(Xtest_std)
    return Xtrain,Xtest
Xtrain0,Xtest0 = PCA(Xtrain0,Xtest0)
Xtrain1,Xtest1 = PCA(Xtrain1,Xtest1)
Xtrain2,Xtest2 = PCA(Xtrain2,Xtest2)
Xtrain3,Xtest3 = PCA(Xtrain3,Xtest3)
Xtrain4,Xtest4 = PCA(Xtrain4,Xtest4)
contoh_mlp_x = StandardScaler().fit_transform(contoh_mlp_x)
contoh_gauss_x,temp = PCA(contoh_gauss_x,Xtest1)
contoh_pi_x,temp = PCA(contoh_pi_x,Xtest1)
contoh_trap_x,temp = PCA(contoh_trap_x,Xtest1)
contoh_tri_x,temp = PCA(contoh_tri_x,Xtest1)

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
# mlp model
MlpModel = Sequential()
MlpModel.add(Dense(hid,input_dim=inp,activation="sigmoid"))
MlpModel.add(Dense(out,activation="sigmoid"))
opt = gradient_descent_v2.SGD(learning_rate=0.001)
MlpModel.compile(
    optimizer=opt, 
    loss=RMSE, 
    metrics=['accuracy',f1_m,precision_m,recall_m])
MlpModel.load_weights("mlp_model.h5")

loss,acc,fscore,precision,recall = MlpModel.evaluate(Xtest0,ytest0,verbose=0)
print()
print("MLP model")
print("Accuracy =",acc)
print("Precision =",precision)
print("Recall =",recall)
print("F-score =",fscore)

# gauss model
GaussModel = Sequential()
GaussModel.add(Dense(hid,input_dim=inp,activation="sigmoid"))
GaussModel.add(Dense(out,activation="sigmoid"))
opt = gradient_descent_v2.SGD(learning_rate=0.001)
GaussModel.compile(
    optimizer=opt, 
    loss=RMSE, 
    metrics=['accuracy',f1_m,precision_m,recall_m])
GaussModel.load_weights("gaussmf_model.h5")

loss,acc,fscore,precision,recall = GaussModel.evaluate(Xtest1,ytest1,verbose=0)
print()
print("Gauss model")
print("Accuracy =",acc)
print("Precision =",precision)
print("Recall =",recall)
print("F-score =",fscore)

# pi model
PiModel = Sequential()
PiModel.add(Dense(hid,input_dim=inp,activation="sigmoid"))
PiModel.add(Dense(out,activation="sigmoid"))
opt = gradient_descent_v2.SGD(learning_rate=0.001)
PiModel.compile(
    optimizer=opt, 
    loss=RMSE, 
    metrics=['accuracy',f1_m,precision_m,recall_m])
PiModel.load_weights("pimf_model.h5")

loss,acc,fscore,precision,recall = PiModel.evaluate(Xtest2,ytest2,verbose=0)
print()
print("Pi model")
print("Accuracy =",acc)
print("Precision =",precision)
print("Recall =",recall)
print("F-score =",fscore)

# tri model
TriModel = Sequential()
TriModel.add(Dense(hid,input_dim=inp,activation="sigmoid"))
TriModel.add(Dense(out,activation="sigmoid"))
opt = gradient_descent_v2.SGD(learning_rate=0.001)
TriModel.compile(
    optimizer=opt, 
    loss=RMSE, 
    metrics=['accuracy',f1_m,precision_m,recall_m])
TriModel.load_weights("trimf_model.h5")

loss,acc,fscore,precision,recall = TriModel.evaluate(Xtest3,ytest3,verbose=0)
print()
print("Triangular model")
print("Accuracy =",acc)
print("Precision =",precision)
print("Recall =",recall)
print("F-score =",fscore)

# trap model
TrapModel = Sequential()
TrapModel.add(Dense(hid,input_dim=inp,activation="sigmoid"))
TrapModel.add(Dense(out,activation="sigmoid"))
opt = gradient_descent_v2.SGD(learning_rate=0.001)
TrapModel.compile(
    optimizer=opt, 
    loss=RMSE, 
    metrics=['accuracy',f1_m,precision_m,recall_m])
TrapModel.load_weights("trapmf_model.h5")

loss,acc,fscore,precision,recall = TrapModel.evaluate(Xtest4,ytest4,verbose=0)
print()
print("Trapezoidal model")
print("Accuracy =",acc)
print("Precision =",precision)
print("Recall =",recall)
print("F-score =",fscore)
print("\n===================================\n")
hasilMLP = (MlpModel.predict(contoh_mlp_x) > 0.5).astype("int32")
TP = 0
FN = 0
FP = 0
TN = 0
for i in range(20):
    if hasilMLP[i] == contoh_y[i]:
        if contoh_y[i] == 1 : TP += 1
        else : TN += 1
    else :
        if contoh_y[i] == 1 : FN += 1
        else : FP += 1
print("MLP =",np.reshape(hasilMLP, 20))
print("MLP =",TP, FN, FP, TN)
print()

hasilGauss = (GaussModel.predict(contoh_gauss_x) > 0.5).astype("int32")
TP = 0
FN = 0
FP = 0
TN = 0
for i in range(20):
    if hasilGauss[i] == contoh_y[i]:
        if contoh_y[i] == 1 : TP += 1
        else : TN += 1
    else :
        if contoh_y[i] == 1 : FN += 1
        else : FP += 1
print("Gauss =",np.reshape(hasilGauss, 20))
print("Gauss =",TP, FN, FP, TN)
print()

hasilPi = (PiModel.predict(contoh_pi_x) > 0.5).astype("int32")
TP = 0
FN = 0
FP = 0
TN = 0
for i in range(20):
    if hasilPi[i] == contoh_y[i]:
        if contoh_y[i] == 1 : TP += 1
        else : TN += 1
    else :
        if contoh_y[i] == 1 : FN += 1
        else : FP += 1
print("Pi =",np.reshape(hasilPi, 20))
print("Pi =",TP, FN, FP, TN)
print()

hasilTrap = (TrapModel.predict(contoh_trap_x) > 0.5).astype("int32")
TP = 0
FN = 0
FP = 0
TN = 0
for i in range(20):
    if hasilTrap[i] == contoh_y[i]:
        if contoh_y[i] == 1 : TP += 1
        else : TN += 1
    else :
        if contoh_y[i] == 1 : FN += 1
        else : FP += 1
print("Trap =",np.reshape(hasilTrap, 20))
print("Trap =",TP, FN, FP, TN)
print()

hasilTri = (TriModel.predict(contoh_tri_x) > 0.5).astype("int32")
TP = 0
FN = 0
FP = 0
TN = 0
for i in range(20):
    if hasilTri[i] == contoh_y[i]:
        if contoh_y[i] == 1 : TP += 1
        else : TN += 1
    else :
        if contoh_y[i] == 1 : FN += 1
        else : FP += 1
print("Tri =",np.reshape(hasilTri, 20))
print("Tri =",TP, FN, FP, TN)
print()