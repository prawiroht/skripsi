import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Input data
dataset = pd.read_csv('data/cardio.csv',sep=";")
X = dataset.values[:,1:12] #Fitur
y = dataset.values[:,12]   #Target
# Normalisasi
for x in X:
    x[0] /= 365
    x[1] -= 1
# Split data
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.25)
# Standardisasi
from sklearn.preprocessing import StandardScaler
sclr = StandardScaler()
Xtrain = sclr.fit_transform(Xtrain)
Xtest = sclr.fit_transform(Xtest)

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
model.save_weights("mlp_model.h5")

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