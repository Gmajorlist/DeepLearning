import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional # 방향성을 정해주는 것 양방향으로 쓰임

dataset = np.array(range(1, 101)) 
pre_data = np.array(range(96, 106))  
timesteps = 5

def split_x(dataset, timesteps):
    tmp = []
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : (i + timesteps)]
        tmp.append(subset)
    return np.array(tmp)

dataset = split_x(dataset, timesteps) 
pre_data = split_x(pre_data, timesteps)

print("dataset.shape :", dataset.shape)
print("pre_data.shape: ", pre_data.shape)

x = dataset[:, :-1] 
y = dataset[:, -1]

pre_data = pre_data[:, :-1] 
print("pre_data : ", pre_data)
print("pre_data.shape : ", pre_data.shape)

print("x.shape: ", x.shape, "y.shape: ", y.shape)

x = x.reshape(-1, 4, 1)

model = Sequential() 
# model.add(LSTM(100, input_shape=(4,1)))                       #40800                                              
model.add(Bidirectional(LSTM(100,  return_sequences= True) ,input_shape=(4, 1),))                #81600
model.add(LSTM(64))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(16))
model.add(Dense(1))
model.summary() 


model.compile(loss='mse', optimizer= 'adam')
model.fit(x, y, epochs=5)

loss = model.evaluate(x, y)
print(loss)

y_predict = np.array(pre_data)

result = model.predict(y_predict)

print('prediction result:\n', result) 

