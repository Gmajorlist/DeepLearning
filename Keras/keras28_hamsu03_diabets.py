import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#모델구성
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) 

x_train, x_test, y_train, y_test = train_test_split(x,y
    , test_size=0.2, random_state=29 )
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

scaler = MinMaxScaler() 
# scaler = StandardScaler()
scaler.fit(x_train)
x = scaler.transform(x_train)
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#모델구성
model = Sequential()
model.add(Dense(1, activation='relu',input_dim=10)) 
model.add(Dense(2, activation='relu',input_shape=(10, ))) 
model.add(Dense(7, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='relu'))

input1 = Input(shape=(13, ))
dense1 = Dense(1, activation='relu')(input1)
dense2 = Dense(2, activation='relu')(dense1)
dense3 = Dense(7, activation='relu' )(dense2)
dense4 = Dense(4, activation='relu')(dense3)
dense5 = Dense(2, activation='relu')(dense4)
output1 = Dense(1, activation='relu')(dense5)






model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping #대문자 - 파이썬에 클래스으로 구성
earlyStopping = EarlyStopping(monitor='val_loss',
                              mode='min',
                              patience=5, # 인내심 - 몇번참는다 
                              restore_best_weights=True, #최소의 loss일 때 데이터를 가져옴
                              verbose=1,
                               )
hist = model.fit(x_train, y_train, epochs=300, batch_size=1,
          validation_split=0.2, verbose=1, callbacks=[earlyStopping],
          )

#4.평가 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)
