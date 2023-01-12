import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#데이터
path = './_data/ddarung/'
train_csv = pd.read_csv(path +'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'submission.csv', index_col=0)

print(train_csv)
print(train_csv.shape)
print(submission.shape)
print(train_csv.columns)
print(train_csv.info())
print(test_csv.info())
print(train_csv.describe())

print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.shape)
x = train_csv.drop(['count'], axis=1)
print(x)
y = train_csv['count']
print(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.3, shuffle=True, random_state=124)
print(x_train.shape, x_test.shape) #(1021, 9) (438, 9)
print(y_train.shape, y_test.shape)

# scaler = MinMaxScaler() 
scaler = StandardScaler()
scaler.fit(x_train)
x = scaler.transform(x_train)
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#2.모델구성
model = Sequential()
# model.add(Dense(100,activation='relu', input_dim=9)) # 행과 열
# model.add(Dense(10,activation='relu', input_shape=(9, ))) #(13, )
# model.add(Dense(1))

a1 = Input(shape=(9,))
b1 = Dense(3)(a1)
b2 = Dense(50)(b1)
b3 = Dense(1)(b2)
model = Model(inputs=a1, outputs=b3)

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss',
                              mode='min',
                              patience=10,  
                              restore_best_weights=True,  
                              verbose=1,
                               )

hist = model.fit(x_train, y_train, epochs=99999, batch_size=24,
          validation_split=0.2, verbose=1, callbacks=[earlyStopping],
          )

#4.평가 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)


y_predict = model.predict(x_test)
print(y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE:", rmse)

#파일만들기
y_submit = model.predict(test_csv)
print(y_submit.shape)
submission['count'] = y_submit
submission.to_csv(path + 'submission_00.csv')
