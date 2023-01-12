# 과제,실습
# R2 0.62 이상


from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
#1.데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(x.shape) #442,10
print(y)
print(y.shape) #442, 

x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.7, shuffle=True
    )

model = Sequential()
model.add(Dense(199, input_dim=10))
model.add(Dense(200))
model.add(Dense(44))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam',
              metrics=['mae', 'acc'])
model.fit(x_train, y_train, epochs=1234567, batch_size=2)

loss = model.evaluate(x_test, y_test) 
print('loss : ', loss)
y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE :", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 :", r2)

#R2 : 0.5314419760798559