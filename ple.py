import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.python.keras.layers import Layer
from tensorflow.keras import regularizers



class PleLayer(tf.keras.layers.Layer):
    '''
    n_experts:list,每个任务使用几个expert。[2,3]第一个任务使用2个expert，第二个任务使用3个expert。
    n_expert_share:int,共享的部分设置的expert个数。
    expert_dim:int,每个专家网络输出的向量维度。
    n_task:int,任务个数。（2个任务：点击、下单）
    '''
    def __init__(self,n_task,n_experts,expert_dim,n_expert_share,dnn_reg_l2 = 1e-5):
        super(PleLayer, self).__init__()
        self.n_task = n_task
        
        # 生成多个任务特定网络和1个共享网络。
        self.E_layer = []
        for i in range(n_task):
            sub_exp = [Dense(expert_dim,activation = 'relu') for j in range(n_experts[i])]
            self.E_layer.append(sub_exp)
            
        self.share_layer = [Dense(expert_dim,activation = 'relu') for j in range(n_expert_share)]
        #定义门控网络
        self.gate_layers = [Dense(n_expert_share+n_experts[i],kernel_regularizer=regularizers.l2(dnn_reg_l2),
                                  activation = 'softmax') for i in range(n_task)]

    def call(self,x):
        #特定网络和共享网络
        E_net = [[expert(x) for expert in sub_expert] for sub_expert in self.E_layer]
        share_net = [expert(x) for expert in self.share_layer]
        
        #门的权重乘上，指定任务和共享任务的输出。
        towers = []
        for i in range(self.n_task):
            g = self.gate_layers[i](x)
            g = tf.expand_dims(g,axis = -1) #(bs,n_expert_share+n_experts[i],1)
            _e = share_net+E_net[i]  
            _e = Concatenate(axis = 1)([expert[:,tf.newaxis,:] for expert in _e]) #(bs,n_expert_share+n_experts[i],expert_dim)
            _tower = tf.matmul(_e, g,transpose_a=True)
            # print("ccccc",_tower.shape) # (?, 32, 1)   Flatten()(_tower).shape  (32, 1)
            towers.append(Flatten()(_tower)) #(bs,expert_dim)
            
        return towers