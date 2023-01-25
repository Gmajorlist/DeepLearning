from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Flatten, Conv2D, Dense
from sklearn.preprocessing import MinMaxScaler as MMS, StandardScaler as SDS
from tensorflow.keras.callbacks import EarlyStopping 
import numpy as np

# 1. 데이터
dataset = load_boston()        
x = dataset.data                
y = dataset.target              

print("원본 데이터")
# print("x: ", x, "\ny: ", y)
print(x.shape, y.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=3333)

# scaler = MMS()
scaler = SDS()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print("split + scailing 데이터")
# print("x_test: ", x_test, "\nx_trian: ", x_train)
# print("y_test: ", y_test, "\ny_trian: ", y_train)
print(x_train.shape, x_test.shape)

# ---------- CNN 모델에 적용해보기 위해 4차원으로 변환 ----------- #
x_train = x_train.reshape(354, 13, 1, 1)
x_test = x_test.reshape(152, 13, 1, 1)


# 2. 모델
model = Sequential()
model.add(Conv2D(32, (2,1), input_shape = (13, 1, 1), activation='relu'))
model.add(Dropout(0.5)) # 과적합 방지
model.add(Conv2D(16, (2,1), activation='relu'))
model.add(Dropout(0.3)) # 과적합 방지
model.add(Flatten())    # DNN모델에 적용하기 위해 2차원으로 변환
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2)) # 과적합 방지
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam') # one-hot encoding 하지 않아도 되는 데이터이므로 loss= sparse_categorical_crossentropy

ES = EarlyStopping(monitor='val_loss', mode='auto', patience=4, restore_best_weights=True) 
model.fit(x_train, y_train, epochs=64, batch_size=5, validation_split=0.2, callbacks = [ES], verbose=2) # verbose: 함수 수행시 발생하는 상세한 정보들을 표준 출력으로 자세히 내보낼 것인지

        #   4. 평가 및 예측
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