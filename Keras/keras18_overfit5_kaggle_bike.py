import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

path = './_data/bike/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
samplesubmission = pd.read_csv(path +'samplesubmission.csv', index_col=0)
print(train_csv)
print(train_csv.shape)
print(samplesubmission.shape)
print(train_csv.columns)
print(train_csv.info())
print(test_csv.info())
print(train_csv.describe())
#결측지제거
print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.shape)
x = train_csv.drop(['casual','registered','count'], axis=1)
print(x)
y = train_csv['count']
print(y)
print(y.shape)
x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.2, shuffle=True, random_state=123)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
model = Sequential()
model.add(Dense(1, input_dim=8)) # 행과 열
model.add(Dense(2, input_shape=(8, ))) #(13, )
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=10, batch_size=5,
          validation_split=0.2, verbose=1)

loss = model.evaluate(x_test, y_test)
print('loss:', loss)

print("================================")
print(hist)#<keras.callbacks.History object at 0x0000021C42043DF0>
print("================================")
print(hist.history)
print("================================")
print(hist.history['loss'])
print("================================")
print(hist.history['val_loss'])


y_predict = model.predict(x_test)
print(y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE:", rmse)
y_submit = model.predict(test_csv)
samplesubmission['count'] = y_submit
samplesubmission.to_csv(path + 'submission_val1.csv')

import matplotlib.pyplot as plt

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c = 'red'
         , marker = '.', label = 'loss')
plt.plot(hist.history['val_loss'], c = 'blue'
         , marker = '.', label = 'val_loss')
plt.grid() 
plt.xlabel('epoch')   
plt.ylabel('loss')
plt.title('boston loss')
plt.legend()
plt.show()