from tensorflow.keras.datasets import cifar10, cifar100 # 칼라다
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D ,Dense, Flatten,MaxPooling2D, Dropout


path = 'C:/study/keras_save/MCP/'
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# 데이터 확인
print(x_train.shape, y_train.shape)# (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train, return_counts=True)) 
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],
     # dtype=int64))
x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test,
                              train_size=0.8, random_state=123)

model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2,2), input_shape=(32,32,3),  
                 activation='relu'))                                          
model.add(Conv2D(filters=64, kernel_size=(2,2))) 
model.add(MaxPooling2D(pool_size=(2,2))) 
model.add(Dropout(0.25))                      
model.add(Conv2D(filters=64, kernel_size=(2,2)))
model.add(MaxPooling2D(pool_size=(2,2))) 
model.add(Dropout(0.25))                              
model.add(Dense(64, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2))) 
model.add(Dropout(0.25))            
model.add(Dense(units=32, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2))) 
model.add(Dropout(0.25))   
model.add(Flatten())                      
model.add(Dense(10, activation= 'softmax'))


#컴파일 훈련

# import datetime
# date = datetime.datetime.now() #현재 시간이 나옴
# print(date)
# print(type(date)) #<class 'datetime.datetime'>
# date = date.strftime("%m%d_%H%M") 
# print(date)  #0112_1503
# print(type(date)) 

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='min',patience=20,
                   verbose=3, restore_best_weights=True)
filepath = './_save/MCP/'
filename ='{epoch:04d}-{val_loss:.4f}.hdf5'  
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto',
                      save_best_only=True, verbose = 1,
                     filepath = filepath + 'k34_2' + '_' + filename)
model.fit(x_train, y_train, epochs=10,validation_data=(x_valid, y_valid) ,verbose=3,
        callbacks=[es,mcp] , batch_size=32)

# 평가 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0] )
print('acc:', results[1])