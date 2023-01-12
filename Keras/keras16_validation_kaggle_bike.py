import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
#데이터
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
#모델구성
model = Sequential()
model.add(Dense(100,input_dim=8))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))
#컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=1
         , validation_split=0.25)
#평가 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)
print(y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE:", rmse)
y_submit = model.predict(test_csv)
# print(y_submit)
# print(y_submit.shape) 

# print(samplesubmission)
samplesubmission['count'] = y_submit
# print(samplesubmission)
samplesubmission.to_csv(path + 'submission_val1.csv')
