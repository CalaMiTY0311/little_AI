import tensorflow as tf

#tensor = tf.constant(3) -> tf.Tensor(3, shape=(), dtype=int32)
#tensor = tf.constant([3,4,5]) ->  tf.Tensor([3 4 5], shape=(3,), dtype=int32)

"""
tensor = tf.constant([3,4,5])
tensor2 = tf.constant([6,7,8])
print(tensor+tensor2)
print(tf.add(tensor1,tensor2))
----->      tf.Tensor([ 9 11 13], shape=(3,), dtype=int32)

#tensor의 행렬
tensor3 = tf.constant([[1,2],
                       [3,4]])
print(tensor3)
---->
tf.Tensor(
[[1 2]
 [3 4]], shape=(2, 2), dtype=int32)
 
shape(2,2)의 의미 ->리스트안에 값이 2개있는 리스트가 2개 있다는것


tensor4 = tf.zeros( 10 )
print(tensor4)
------------>
tf.Tensor([0. 0. 0. 0. 0. 0. 0. 0. 0. 0.], shape=(10,), dtype=float32)
이거 왜 0.으로 되어있냐면 뒤에 타입이 float이라 그럼

tensor4 = tf.zeros([2,2]) tensor의 shape
------------>
tf.Tensor(
[[0. 0.]
 [0. 0.]], shape=(2, 2), dtype=float32)

print(tensor4.shape)
하면 (2,2) 출력됨


"""
w =tf.Variable(1.0)   #Variable 딥러닝에서의 weight 우리가 평소 알고있는 변수
print(w)
#<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=1.0>
#만약에 numpy만 불러오고싶으면 w.numpy()
