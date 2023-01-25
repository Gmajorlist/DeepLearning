import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
from sklearn.model_selection import train_test_split
path = 'C:/study/keras_save/MCP/'


# 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 데이터 확인
print(x_train.shape, y_train.shape)  
print(x_test.shape, y_test.shape)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
print(x_train.shape, y_train.shape) 
print(x_test.shape, y_test.shape)


x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test,
                              train_size=0.8, random_state=123)
# 선생님을 믿을 수 없을 때 행동
print(np.unique(y_train, return_counts=True))


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D ,Dense, Flatten, MaxPooling2D

#모델
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(3,3), input_shape=(28,28,1),
                 padding='same',   # padding이 valid 라면
                 activation='relu'))    #28,28, 128   에서 Maxpool쓰면 아래                                 
model.add(MaxPooling2D())               #14,14, 128
model.add(Conv2D(filters=64, kernel_size=(2,2),
                 padding='same')) 
model.add(MaxPooling2D())                            
model.add(Conv2D(filters=64, kernel_size=(2,2)))                         
model.add(Flatten())
model.add(Dense(32, activation='relu')) 
model.add(Dense(10, activation= 'softmax'))
model.summary()


#컴파일 훈련

import datetime
date = datetime.datetime.now() 
print(date)
print(type(date)) 
date = date.strftime("%m%d_%H%M") 
print(date) 
print(type(date)) 

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='min',patience=20,
                   verbose=3, restore_best_weights=True)
filepath = './_save/MCP/'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'  
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto',
                      save_best_only=True, verbose = 1,
                     filepath = filepath + 'k34_1' + date + '_' + filename)
model.fit(x_train, y_train, epochs=10,validation_data=(x_valid, y_valid) ,verbose=3,
        callbacks=[es,mcp] , batch_size=32)

# 평가 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0] )
print('acc:', results[1])


# es. mcp 적용 val 적용
#loss :  0.1349068135023117
#acc: 0.9700000286102295


# 기존 성능
# padding  적용시
# acc: 0.9645000100135803

#Maxpooling2D 적용시

# acc: 0.9793750047683716