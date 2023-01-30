import numpy as np
from sklearn.model_selection import train_test_split

x1 = np.array([range(100), range(301, 401)]).T                          # ex) 삼전 시가, 고가

y1 = np.array(range(2001, 2101)) # 삼전 하루 뒤 종가
y2 = np.array(range(201, 301)) # 아모레 하루 뒤 종가
y3 = np.array(range(201, 301)) # 아모레 하루 뒤 종가


print(x1.shape)

x1_train, x1_test, \
    y1_train, y1_test, \
            y2_train, y2_test, \
                y3_train, y3_test = train_test_split(x1, y1, y2, y3, train_size=0.7, random_state=3333)

print(x1_train.shape, x1_test.shape,
      y1_train.shape, y1_test.shape,
      y2_train.shape, y2_test.shape)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate

# 모델 1
input1 = Input(shape=(2, ))
dense1 = Dense(16, activation='relu', name='dense1')(input1) # summary를 봤을 때 보기쉽게 이름 지정
dense2 = Dense(16, activation='relu', name='dense2')(dense1)
dense3 = Dense(16, activation='relu', name='dense3')(dense2)
output1 = Dense(16, activation='relu', name='output1')(dense3)

# 분기 모델 2
dense2_1 = Dense(16, activation='linear', name='dense2_1')(output1)
dense2_2 = Dense(16, activation='linear', name='dense2_2')(dense2_1)
output2 = Dense(16, activation='linear', name='output2')(dense2_2)

# 분기 모델 3
dense3_1 = Dense(16, activation='linear', name='dense3_1')(output1)
dense3_2 = Dense(16, activation='linear', name='dense3_2')(dense3_1)
output3 = Dense(16, activation='linear', name='output3')(dense3_2)

# 분기 모델 4
dense4_1 = Dense(16, activation='linear', name='dense4_1')(output1)
dense4_2 = Dense(16, activation='linear', name='dense4_2')(dense4_1)
output4 = Dense(16, activation='linear', name='output4')(dense4_2)

model = Model(inputs=[input1], outputs=[output2, output3, output4])

model.summary()

model.compile(loss='mse', optimizer= 'adam', metrics='mae')
model.fit([x1_train], [y1_train, y2_train, y3_train], epochs=16, batch_size=3)


loss = model.evaluate([x1_test], [y1_test, y2_test, y3_test])
print("loss(mse): ", loss)
print("순서대로: loss, output2_loss, output3_loss, output4_loss, output2_mae, output3_mae, output4_mae")
