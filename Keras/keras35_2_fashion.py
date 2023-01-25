from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 데이터 확인
print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,) 흑백데이터  # 처음에는 이렇게 다운받는다 
print(x_test.shape, y_test.shape)  #(10000, 28, 28) (10000,)

# 몇번째 그림이 무엇인지 확인
print(x_train[1000])       #빛의 삼원색
print(y_train[1000])
# # 그림 
# import matplotlib.pyplot as plt
# plt.imshow(x_train[1000], 'gray')
# plt.show()

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
print(x_train.shape, y_train.shape) 
print(x_test.shape, y_test.shape)


x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test,
                              train_size=0.8, random_state=123)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D ,Dense, Flatten , MaxPooling2D

#모델
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2,2), input_shape=(28,28,1),
                 padding='same',
                 strides=1,  #<<< 연산을 함 연산량이 많음
                 activation='relu'))            
model.add(MaxPooling2D())
model.add(Conv2D(filters=64, kernel_size=(2,2))) 
model.add(MaxPooling2D())
model.add(Conv2D(filters=64, kernel_size=(2,2)))                         
model.add(Flatten()) # ---->40000
model.add(Dense(32, activation='relu')) #input_shape = (40000,) 
                                        #(60000,40000)이 인풋이야 (batch_size, input_dim)
model.add(Dense(10, activation= 'softmax'))

#컴파일 훈련

import datetime
date = datetime.datetime.now() #현재 시간이 나옴
print(date)
print(type(date)) #<class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M") 
print(date)  #0112_1503
print(type(date)) 

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='min',patience=20,
                   verbose=3, restore_best_weights=True)
filepath = './_save/MCP/'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'  
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto',
                      save_best_only=True, verbose = 1,
                     filepath = filepath + 'k35_2' + date + '_' + filename)
model.fit(x_train, y_train, epochs=10,validation_data=(x_valid, y_valid) ,verbose=3,
        callbacks=[es,mcp] , batch_size=32)

# 평가 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0] )
print('acc:', results[1])



# acc: 0.8887500166893005