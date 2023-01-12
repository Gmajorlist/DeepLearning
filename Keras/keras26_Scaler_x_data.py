#확인
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

# scaler = MinMaxScaler() 
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
print(x)
print(type(x)) #<class 'numpy.ndarray'>

# print("최소값:", np.min(x))
# print("최대값:", np.max(x))



x_train, x_test, y_train, y_test = train_test_split(x,y
    , test_size=0.2, random_state=29 )
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

#모델
model = Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))

#컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=1
          ,validation_split=0.25)
#평가 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)
y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE :", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 :", r2)
