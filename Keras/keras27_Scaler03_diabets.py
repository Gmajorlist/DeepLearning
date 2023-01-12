import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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
model.add(Dense(1, input_dim=10)) 
model.add(Dense(2, input_shape=(10, ))) 
model.add(Dense(7, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='relu'))


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


import matplotlib.pyplot as plt

# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c = 'red'
#          , marker = '.', label = 'loss')
# plt.plot(hist.history['val_loss'], c = 'blue'
#          , marker = '.', label = 'val_loss')
# plt.grid()
# plt.xlabel('epoch')  
# plt.ylabel('loss')
# plt.title('boston loss')
# plt.legend() 

# plt.show()