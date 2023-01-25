from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten
from sklearn.preprocessing import MinMaxScaler as MMS, StandardScaler as SDS
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd

#데이터
path ='C:/study/_data/ddarung/' 
train_data = pd.read_csv(path + 'train.csv', index_col = 0)         # index_col = 0 → date_t 열 데이터로 취급 X
test_data = pd.read_csv(path + 'test.csv', index_col = 0)
submission = pd.read_csv(path + 'sampleSubmission.csv', index_col = 0)

# print(train_data.shape)          # (10886, 11)  
# print(test_data.shape)           # (6493, 8)
# print(train_data.columns)   
# print(train_data.info())         # Missing Attribute Values: 결측치 - 데이터에 값이 없는 것
# print(train_data.describe())     # 평균, 표준편차, 최대값 등

# ---------------------- shape 맞추기 (열 제거) ------------------------ #
train_data = train_data.drop(['casual', 'registered'], axis = 1)

# ---------------------- x,y 분리 ------------------------ #
x = train_data.drop(['count'], axis=1)                              # y 값(count 열) 분리, axis = 1 → 열에 대해 동작
y = train_data['count']                                             # y 값(count 열)만 추출

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=3333)

# scaler = MMS()
scaler = SDS()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_data = scaler.transform(test_data)

print("x_train: ", x_train.shape, "x_test:", x_test.shape)

# ---------- CNN 모델에 적용해보기 위해 4차원으로 변환 ----------- #
x_train = x_train.reshape(-1, 8, 1, 1)
x_test = x_test.reshape(-1, 8, 1, 1)
test_data = test_data.reshape(-1, 8, 1, 1)

print("x_train: ", x_train.shape, "x_test:", x_test.shape)

#모델
model = Sequential()
model.add(Conv2D(32, (2,1), input_shape = (8, 1, 1), activation='relu'))
model.add(Dropout(0.5)) # 과적합 방지
model.add(Conv2D(16, (2,1), activation='relu'))
model.add(Dropout(0.3)) # 과적합 방지
model.add(Flatten())    # DNN모델에 적용하기 위해 2차원으로 변환
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2)) # 과적합 방지
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

#컴파일 훈련
model.compile(loss='mse', optimizer='adam') # one-hot encoding 하지 않아도 되는 데이터이므로 loss= sparse_categorical_crossentropy

ES = EarlyStopping(monitor='val_loss', mode='auto', patience=4, restore_best_weights=True) 
model.fit(x_train, y_train, epochs=64, batch_size=5, validation_split=0.2, callbacks = [ES], verbose=2) 

#평가 예측
loss = model.evaluate(x_test, y_test, verbose=2)
print('loss(mse): ', loss)

y_predict = model.predict(x_test)
print('x_test:\n', x_test[5])
print('y_test:\n', y_test[5])

print('y_predict:\n', y_predict[5])

RMSE = np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE: ", RMSE)

r2 = r2_score(y_test, y_predict)
print("R2: ", r2)

y_submit = model.predict(test_data)
submission['count'] = y_submit
submission.to_csv(path + 'submission_0125.csv')
