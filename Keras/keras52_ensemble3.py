import numpy as np

# 데이터 

x1 = np.array([range(100), range(301, 401)]).T                          # ex) 삼전 시가, 고가
x2 = np.array([range(101, 201), range(411, 511), range(150, 250)]).T    # ex) 아모레 시가, 고가, 종가
x3 = np.array([range(100, 200), range(1301, 1401)]).T
y1 = np.array(range(2001, 2101))                                        # 삼전 하루 뒤 종가
y2 = np.array(range(201, 301))                                          # 아모레 하루 뒤 종가

print(x1.shape)
print(x2.shape)
print(x3.shape)
print(y1.shape)
print(y2.shape)

from sklearn.model_selection import train_test_split

x1_train, x1_test, \
    x2_train, x2_test, \
        x3_train, x3_test, \
            y1_train, y1_test, \
                y2_train, y2_test = train_test_split(x1, x2, x3, y1, y2, train_size=0.7, random_state=3333)

print(x1_train.shape, x1_test.shape,
      x2_train.shape, x2_test.shape,
      x3_train.shape, x3_test.shape,
      y1_train.shape, y1_test.shape,
      y2_train.shape, y2_test.shape)
 #(30, 2) (30, 3) (30, 2) (70,)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense , Input ,concatenate, Concatenate


#2-1 모델 1 
input1 = Input(shape = (2, ))
dense1 = Dense(11, activation='relu', name='ds11')(input1)
dense2 = Dense(12, activation='relu', name='ds12')(dense1)
dense3 = Dense(13, activation='relu', name='ds13')(dense2)
output1 = Dense(11, activation='relu', name='ds14')(dense3)

#2-2 모델 2
input2 = Input(shape = (3, ))
dense21 = Dense(21, activation='linear', name='ds21')(input2)
dense22 = Dense(22, activation='linear', name='ds22')(dense21)
output2 = Dense(23, activation='linear', name='ds23')(dense22)

#2-3 모델 3 
input3 = Input(shape = (2, ))
dense33 = Dense(31, activation='linear', name='ds33')(input3)
dense44 = Dense(32, activation='linear', name='ds44')(dense33)
output3 = Dense(33, activation='linear', name='ds55')(dense44)

# 2-4모델병합
merge1 = concatenate([output1,output2,output3], name='mg1')
merge2 = Dense(12, activation='relu', name='mg2')(merge1)
merge3 = Dense(13, activation='relu', name='mg3')(merge2)
last_output = Dense(1, name = 'last')(merge3)  # 여기서 1은 y 이다 컬럼이 하나임

#2-3 모델 3 

dense5 = Dense(31, activation='linear', name='ds657')(last_output)
dense5 = Dense(32, activation='linear', name='ds234')(dense5)
output5 = Dense(33, activation='linear', name='ds123')(dense5)


dense6 = Dense(31, activation='linear', name='ds8')(last_output)
dense6 = Dense(32, activation='linear', name='ds9')(dense6)
output6 = Dense(33, activation='linear', name='ds99')(dense6)

model = Model(inputs=[input1, input2, input3], outputs=[output5,output6])

model.summary()

#3 컴파일 훈련
model.compile(loss='mse', optimizer= 'adam')
model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train],
          epochs=16, batch_size=3)


#4 평가 예측
loss = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test])
print("loss(mse): :", loss)





