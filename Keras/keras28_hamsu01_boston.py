#확인
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping 
from sklearn.metrics import mean_squared_error, r2_score
#데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y
    , test_size=0.2, random_state=29 )

scaler = MinMaxScaler() 
# scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)




# #모델 (순차형)
# model = Sequential()
# model.add(Dense(50, activation='relu', input_dim=13))
# model.add(Dense(40, activation='sigmoid'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(20, activation='linear'))
# model.add(Dense(1, activation='linear'))
# model.summary()
#Total params: 4,611

#모델 (함수형)
input1 = Input(shape=(13, ))
dense1 = Dense(50, activation='relu')(input1)
dense2 = Dense(40, activation='sigmoid')(dense1)
dense3 = Dense(30, activation='relu' )(dense2)
dense4 = Dense(20, activation='linear')(dense3)
output1 = Dense(1, activation='linear')(dense4)

model = Model(inputs=input1, outputs=output1)
# model.summary()

#컴파일 훈련
model.compile(loss='mse', optimizer='adam')
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=16,
                              restore_best_weights=True, verbose=3) 
hist = model.fit(x_train, y_train, epochs=256, batch_size=16,
                 validation_split=0.2, callbacks = [earlyStopping], 
                 verbose=3)
#평가 예측
loss = model.evaluate(x_test, y_test, verbose=3)
print('loss:', loss)
y_predict = model.predict(x_test)


# print(hist) # <keras.callbacks.History object at 0x000001ECB4986D00>
# print(hist.history) # 딕셔너리(key, value) → loss의 변화값을 list로(value는 list로 저장된다.) 
print('val_loss: ', hist.history['val_loss']) # key = loss인 것만 출력


RMSE = np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE: ", RMSE)

r2 = r2_score(y_test, y_predict)
print("R2 :", r2)
