import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
#데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x)
print(y)
print(x.shape, y.shape)#(20640, 8)
x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
#모델
model = Sequential()
model.add(Dense(10, input_dim=8))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))
#컴파일 훈련
model.compile(loss='mse', optimizer='adam'
             , metrics=['mae', 'acc'])
model.fit(x_train, y_train, epochs=10, batch_size=1
          ,validation_split=0.25)
# 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)
y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE :", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 :", r2)