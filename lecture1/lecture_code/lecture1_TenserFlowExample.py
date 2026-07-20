import tensorflow as tf
from tensorflow.keras import layers

class MyDenseLayer(layers.Layer):
    def __init__(self, input_dim, output_dim):
        super(MyDenseLayer, self).__init__()
        self.w = self.add_weight(shape=(input_dim, output_dim),
                                 initializer="random_normal",
                                 trainable=True)
        self.b = self.add_weight(shape=(output_dim,),
                                 initializer="zeros",
                                 trainable=True)

    def call(self, inputs):
        z = tf.matmul(inputs, self.w) + self.b
        return tf.nn.sigmoid(z)

# --- 여기서부터 실행부 ---
# 1. 레이어 만들기 (입력 3개 -> 출력 1개)
layer = MyDenseLayer(3, 1)

# 2. 데이터 넣기 (데이터 2개를 한꺼번에 넣기: shape=(2, 3))
data = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# 3. 결과 뽑기
result = layer(data)

print(result)