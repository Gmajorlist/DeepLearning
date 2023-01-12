import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
model.add(Dense(100, input_dim=8)) # 행과 열
model.add(Dense(90, input_shape=(8, ))) #(13, )
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

input1 = Input(shape=(8, ))
dense1 = Dense(100)(input1)
dense2 = Dense(90)(dense1)
dense3 = Dense(50, activation='relu' )(dense2)
output1 = Dense(1)(dense3)

model = Model(inputs=input1, outputs=output1)
model.summary()

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping #대문자 - 파이썬에 클래스으로 구성
earlyStopping = EarlyStopping(monitor='val_loss',
                              mode='min',
                              patience=5, # 인내심 - 몇번참는다 
                              restore_best_weights=True, #최소의 loss일 때 데이터를 가져옴
                              verbose=1,
                               )

hist = model.fit(x_train, y_train, epochs=39999, batch_size=1,
          validation_split=0.2, verbose=1, callbacks=[earlyStopping],
          )

#4.평가 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)


# import matplotlib.pyplot as plt

# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c = 'red'
#          , marker = '.', label = 'loss')
# plt.plot(hist.history['val_loss'], c = 'blue'
#          , marker = '.', label = 'val_loss')
# plt.grid()#격자들어감
# plt.xlabel('epoch')   #양쪽에 라벨이 생김
# plt.ylabel('loss')
# plt.title('boston loss')
# plt.legend() # 레전드만하면 알아서 빈자리에 위치에 생김
# # plt.legend(loc='upper left')     # 위치 지정할 수 있음
# plt.show()


y_predict = model.predict(x_test)
print(y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE:", rmse)
y_submit = model.predict(test_csv)
samplesubmission['count'] = y_submit
samplesubmission.to_csv(path + 'submission_val1.csv')

