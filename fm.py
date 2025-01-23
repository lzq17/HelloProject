import tensorflow as tf
import tensorflow.keras.backend as K

class FM_layer(tf.keras.layers.Layer):
    def __init__(self, k, w_reg, v_reg):
        super(FM_layer, self).__init__()
        self.k = k   # 隐向量vi的维度
        self.w_reg = w_reg  # 权重w的正则项系数
        self.v_reg = v_reg  # 权重v的正则项系数

    def build(self, input_shape): # 需要根据input来定义shape的变量，可在build里定义)
        self.w0 = self.add_weight(name='w0', shape=(1,), # shape:(1,)
                                 initializer=tf.zeros_initializer(),
                                 trainable=True,)
        self.w = self.add_weight(name='w', shape=(input_shape[-1], 1), # shape:(n, 1)
                                 initializer=tf.random_normal_initializer(), # 初始化方法
                                 trainable=True, # 参数可训练
                                 regularizer=tf.keras.regularizers.l2(self.w_reg)) # 正则化方法
        self.v = self.add_weight(name='v', shape=(input_shape[-1], self.k), # shape:(n, k)
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=tf.keras.regularizers.l2(self.v_reg))

    def call(self, inputs, **kwargs):
        # inputs维度判断，不符合则抛出异常
        if K.ndim(inputs) != 2:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(inputs)))

        # 线性部分，相当于逻辑回归
        linear_part = tf.matmul(inputs, self.w) + self.w0   #shape:(batchsize, 1)
        # 交叉部分——第一项
        inter_part1 = tf.pow(tf.matmul(inputs, self.v), 2)  #shape:(batchsize, self.k)
        # 交叉部分——第二项
        inter_part2 = tf.matmul(tf.pow(inputs, 2), tf.pow(self.v, 2)) #shape:(batchsize, k)
        # 交叉结果
        inter_part = 0.5*tf.reduce_sum(inter_part1 - inter_part2, axis=-1, keepdims=True) #shape:(batchsize, 1)
        # 最终结果
        output = linear_part + inter_part
        return output
    
    
    
class FM(tf.keras.Model):
    def __init__(self, k, w_reg=1e-4, v_reg=1e-4):
        super(FM, self).__init__()
        self.fm = FM_layer(k, w_reg, v_reg) # 调用写好的FM_layer

    def call(self, inputs, training=None, mask=None):
        output = self.fm(inputs)  # 输入FM_layer得到输出
        return output