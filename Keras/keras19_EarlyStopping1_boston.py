from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1.데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

# #(506, 13) (506,) 행무시 열 ! 열이중요! 

x_train, x_test, y_train, y_test = train_test_split(x, y ,
    shuffle= True, random_state=333, test_size=0.2)

#2.모델구성
model = Sequential()
model.add(Dense(100, input_dim=13)) # 행과 열
model.add(Dense(80, input_shape=(13, ))) #(13, )
model.add(Dense(60))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping #대문자 - 파이썬에 클래스으로 구성
earlyStopping = EarlyStopping(monitor='val_loss',
                              mode='min',
                              patience=5, # 인내심 - 몇번참는다 
                              restore_best_weights=True, #최소의 loss일 때 데이터를 가져옴
                              verbose=1,
                               )

hist = model.fit(x_train, y_train, epochs=300, batch_size=1,
          validation_split=0.2, verbose=3, callbacks=[earlyStopping],
          )

#4.평가 예측
loss = model.evaluate(x_test, y_test, verbose =3)
y_predict = model.predict(x_test)
print('loss:', loss)

########t시각화###########
import matplotlib.pyplot as plt

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c = 'red'
         , marker = '.', label = 'loss')
plt.plot(hist.history['val_loss'], c = 'blue'
         , marker = '.', label = 'val_loss')
plt.grid()#격자들어감
plt.xlabel('epoch')   #양쪽에 라벨이 생김
plt.ylabel('loss')
plt.title('boston loss')
plt.legend() # 레전드만하면 알아서 빈자리에 위치에 생김
# plt.legend(loc='upper left')     # 위치 지정할 수 있음
plt.show()



