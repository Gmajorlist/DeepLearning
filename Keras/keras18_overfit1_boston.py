from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1.데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(506, 13) (506,) 행무시 열 ! 열이중요! 

x_train, x_test, y_train, y_test = train_test_split(x, y ,
    shuffle= True, random_state=333, test_size=0.2)

#2.모델구성
model = Sequential()
model.add(Dense(5, input_dim=13)) # 행과 열
model.add(Dense(5, input_shape=(13, ))) #(13, )
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=5, batch_size=1,
          validation_split=0.2, verbose=1)

#4.평가 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)


###################### 시각화 ###############################
#한글 만들어짐 윈도우 프론트 폰트에서 속성누르면 이름나옴
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.ttc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

import matplotlib.pyplot as plt

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c = 'red'
         , marker = '.', label = 'loss')
plt.plot(hist.history['val_loss'], c = 'blue'
         , marker = '.', label = 'val_loss')
plt.grid()#격자들어감
plt.xlabel('epoch')   #양쪽에 라벨이 생김
plt.ylabel('loss')
plt.title('보스톤 손실함수')
plt.legend() # 레전드만하면 알아서 빈자리에 위치에 생김
# plt.legend(loc='upper left')     # 위치 지정할 수 있음
plt.show()




#matplotlib 한글 깨짐


