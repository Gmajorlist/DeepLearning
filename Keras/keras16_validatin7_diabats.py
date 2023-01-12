import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x)
print(y)
print(x.shape, y.shape) #(442, 10),(442,)
x_train, x_test, y_train, y_test =train_test_split(x,y,
    test_size=0.2, random_state=11)
#모델
model = Sequential()
model.add(Dense(5, input_dim=10))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))
#컴파일 훈련
model.compile(loss='mse', optimizer='adam',
              metrics=['mae','acc'])
model.fit(x_train, y_train, epochs=10, batch_size=1)
#평가 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE:",RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2:", r2)
