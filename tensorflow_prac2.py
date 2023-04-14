import tensorflow as tf

"""
키와 신발 사이즈의 관련에 관한 문제
"""

height = 170
shoes_size = 260
#shoes_size = height * a + b 그저 예측 자신이 만들어본 식

#딥러닝 구현
a = tf.Variable(0.1)
b = tf.Variable(0.2)

#이 함수는 그저 자신이 예측 맘대로 예측값을 만들어줬을 뿐임
def loss_function():
    #실제값 - 예측값 인 오차값 리턴
    #오차를 그냥 마이너스로 극한으로 가는것을 방지하기위해 제곱으로한다
    #tf.suqare는 제곱을 해주는 함수
    #return tf.square(real_num - will_num)
    return tf.square(shoes_size - height * a + b)

#경사하강법 Adam은 선택할수있는 optimizers의 한 종류
opt = tf.keras.optimizers.Adam(learning_rate = 0.1)

#경사하강 실행하는법 minimize에 첫번째 인자로 손실함수 두번쨰는 var_list=[]
#var_list안에는 경사하강법으로 업데이트할 weight 변수 목록
#손실함수는 자신이 만들어본다

for i in range(1000):
    opt.minimize(loss_function,var_list=[a,b])
    #print(a,b) 뭔가 numpy값만 보고싶다
    print(a.numpy(),b.numpy())

will_shoes_size = 171 * a + b
print(will_shoes_size)