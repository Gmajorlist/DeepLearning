#실습
#  R2 0.55~0.6 이상
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
#1.데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x)
print(x.shape)  #(20640, 8)  # input_dim 8
print(y)
print(y.shape)  #(20640,)  # out =1

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True
    )

model = Sequential()
model.add(Dense(122, input_dim=8))
model.add(Dense(245))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam',
              metrics=['mae', 'acc'])
model.fit(x_train, y_train, epochs=11112, batch_size=251)

loss = model.evaluate(x_test, y_test)
print('loss :', loss)
y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE :", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 :", r2)

#R2 : 0.6039808435422079

