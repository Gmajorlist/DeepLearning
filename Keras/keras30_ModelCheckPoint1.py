from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import MinMaxScaler as MMS, StandardScaler as SS
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

#1. 데이터
path = 'C:/study/keras_save/MCP/'

dataset = load_boston()       
x = dataset.data               
y = dataset.target              

x_train, x_test, y_train, y_test = train_test_split(x, y,
                            train_size=0.7, random_state=123)

# scaler = MMS()
scaler = SS()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성 
input = Input(shape=(13,))
dense1 = Dense(32)(input)
dense2 = Dense(64, activation= 'relu')(dense1)
dense3 = Dense(256, activation= 'relu')(dense2)
dense4 = Dense(128, activation= 'relu')(dense3)
output = Dense(1)(dense4)
model = Model(inputs=input, outputs=output)
# model.summary()

#3. 컴파일 및 훈련
model.compile(loss = 'mse', optimizer='adam')

ES = EarlyStopping(monitor='val_loss', mode='min', patience=16,
                   verbose = 1, restore_best_weights=True) 
MCP = ModelCheckpoint(monitor='val_loss', mode = 'auto',
                      save_best_only=True, verbose = 1,
                      filepath = path + 'keras30_ModelCheckPoint1.hdf5')
# ModelCheckpoint: 모델과 가중치 저장, 
# save_best_only=True: 가장 좋은 가중치 저장
hist = model.fit(x_train, y_train, epochs=1024, batch_size=16,
                 validation_split=0.2, callbacks=[ES, MCP]) 

#4. 평가 및 예측
loss = model.evaluate(x_test, y_test, verbose=3)
print('loss: ', loss)

y_predict = model.predict(x_test)

RMSE = np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE: ", RMSE)

r2 = r2_score(y_test, y_predict)
print("R2: ", r2)


# R2 : 0.6361258284319979