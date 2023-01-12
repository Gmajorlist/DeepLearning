import numpy as np #연산
import pandas as pd #별거다들어가있음
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1.데이터
path = './_data/ddarung/'                            # [.<<현재폴더 /<<하단을 가르킴 ]- 데이터가 잇는 표시를 한거임
train_csv = pd.read_csv(path + 'train.csv', index_col=0)            # 문자열은 + 하면 이어진다    #  인덱스는 0번째라고 인식해주는 것 
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'submission.csv', index_col=0)

print(train_csv)
print(train_csv.shape) #(1459, 10)
print(submission.shape)

print(train_csv.columns)
print(train_csv.info()) # 결측치 - 빠진 데이터 
print(test_csv.info())
print(train_csv.describe())

# ###################### 결측치 처리 1. 제거#################
print(train_csv.isnull().sum()) 
train_csv = train_csv.dropna() # drop 삭제
print(train_csv.shape) #(1328, 10)

x = train_csv.drop(['count'], axis=1)
print(x) #[1459 rows x 9 columns]
y = train_csv['count']
print(y) 
print(y.shape) #(1459, )

x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.7 , shuffle=True, random_state=14)
print(x_train.shape, x_test.shape) #(1021, 9) (438, 9)
print(y_train.shape, y_test.shape) #(1021,) (438,)

#2. 모델구성
model = Sequential()
model.add(Dense(11, input_dim=9))
model.add(Dense(12))
model.add(Dense(13))
model.add(Dense(16))
model.add(Dense(1))

#3. 컴파일, 훈련
import time
model.compile(loss='mse', optimizer='adam')
start = time.time()
model.fit(x_train, y_train, epochs=12, batch_size=3)
end = time.time()

#4.평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)
print(y_predict)

# 결측치 나쁜놈!!!

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)  
print('걸린시간 :', end - start)

# 제출할 파일만들기

y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape) #(715, 1)

# .to_csv()를 사용해서 
# submission_0105.csv를 완성하시오!!
print(submission)
submission['count'] = y_submit
print(submission)

submission.to_csv(path + 'submission_4.csv')
