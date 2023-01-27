import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

a = np.array(range(1,101))

timesteps = 5  # x 는 4개       y 는 1개

def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps + 1 ):
        subset = dataset[i : (i + timesteps)] 
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, timesteps)

print(bbb)
print(bbb.shape)

x = bbb[:, :-1]
y = bbb[:, -1]
print(x, y)
print(x.shape , y.shape) # (96, 4) (96,)

x_predict = np.array(range(96, 106)) # 예상 y = 100, 107  

model = Sequential()              
model.add(LSTM(units = 64, input_shape=(4, 1), activation='relu' ,
               return_sequences=True))  
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32))
model.add(Dense(1))
model.summary() 

model.compile(loss='mse', optimizer= 'adam')
model.fit(x, y, epochs=200)

loss = model.evaluate(x, y)
print(loss)

y_predict = np.array(range(96, 106))

result = model.predict(y_predict)

print('prediction result: ', result) 

