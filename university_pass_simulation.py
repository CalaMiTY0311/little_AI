import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('gpascore.csv')
#만약만약에 중단에 있어야하는 값이 공백으로 있어서 찾고싶다
#print(data.isnull().sum())             #값이 빠져있는 부분 체크
data = data.dropna()                    #NAN/빈값있는 행 제거
#data.fillna(100)                       #빈값을 100으로 채운다

# x데이터 예시 인풋 [[380,3.21,3], [660,3.67,3], [], [], []]
x_data=[]
for i,rows in data.iterrows():
    x_data.append([rows['gre'],rows['gpa'],rows['rank']])

# y데이터 인풋 [[0], [1], [0], [0], [1]....]
y_data = data['admit'].values

#keras를 안쓰면 weight값을 tf.Variable()로 노가다를 요구하기떄문에 
#텐서플로우 안의 keras의 도움을 받아야함

#딥러닝은 여러개의 노드의 여러개의 히든레이어들을 연결해서 총 결과물을 나중에 생성을해야함
#Sequential 을 쓰면 신경망 레이어들을 쉽게 만들어줌

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation = 'tanh' ),    #강의에서 배워왔던 히든레이어를 표현한것
    tf.keras.layers.Dense(128, activation = 'tanh'),    #Dense안의 값은 히든레이어 노드의 갯수임
    tf.keras.layers.Dense(1, activation = 'sigmoid'),   #이거 갯수는 내맴임 결과 잘 나올떄까지 실험으로 파악해야함
                                                        #Dense(1)은 마지막 출력레이어기 때문에 1로 표현
])

model.compile(optimizer= tf.keras.optimizers.Adam(lr=0.01),loss='binary_crossentropy', metrics=['accuracy'] )
#loss 확률 예측에 자주 사용하는 함수 binary_crossentropy
#즉 이번 결과 0과 1사이의 분류/확률 문제에서 사용하는 함수임

history = model.fit(np.array(x_data) , np.array(y_data), epochs=3000)         
#fit 모델 학습시키기
#epochs 학습 전체 데이터셋을 열번 반복하면서 학습시키는것임
#fit 첫번쨰 인자에는 학습데이터(정답예측에 필요한 인풋) ,두번째는 실제 정답


# x데이터 예시 인풋 [[380,3.21,3], [660,3.67,3], [], [], []]
# y데이터 인풋 [[0], [1], [0], [0], [1]....]
# x데이터와 y데이터 위에 참고

#예측
will = model.predict([[750, 3.70, 3],[400,2.2,1]]) 
print(will)


plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss'])
plt.show()

plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_accuracy'])
plt.show()
