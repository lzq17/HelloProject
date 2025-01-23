import tensorflow as tf
tf.enable_eager_execution()
tf.executing_eagerly()
import warnings
warnings.filterwarnings('ignore') 

print("*"*100)

import logging
import os
import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import *
from tensorflow.python.keras.layers import Layer
from tensorflow.keras import regularizers

from aibrain_common.conf import tf_context
from aibrain_common.data.dataset_builder import DatasetBuilder
from aibrain_common.utils import archive_utils, env_utils, oss_utils
from aibrain_job.utils import param_utils
from aibrain_common.utils.date_convert_utils import DateConvertUtils
from features_info import get_feature_columns
from spark_session_utils import SparkSessionHelper

from config import train_data_df
from features_info import get_features_info
from features_info import get_feature_columns
from aibrain_common.utils import date_convert_utils
from aibrain_common.component import tools

from metric import auc, gauc, labels_mean, logits_mean, topk_acc, mse

from ple import PleLayer # ple模型
from fm import FM # FM模型


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_df_iterator(input_table, partitions, batch_size):
    dataset_builder = DatasetBuilder(input_table=input_table, partitions=partitions)
    hdfs_reader = dataset_builder.get_reader()
    tf_env = env_utils.prepare_tf_env()
    logger.warning(f'tf_env: {tf_env}')
    num_worker = tf_env['WORKER_SIZE']
    index_worker = tf_env['TASK_ID']
    df_iterator = hdfs_reader.to_iterator_for_pdf(num_worker=num_worker, index_worker=index_worker, batch_size=batch_size,\
                                                  shuffle=False, columns=[i.feature_name for i in get_features_info()] + ['click_label'] + ['order_label'] + ['gmv_label'])
    return df_iterator


def generator(input_table, partitions, batch_size):
    df_iterator = get_df_iterator(input_table, partitions, batch_size)
    for pandas_df in df_iterator:
        click_labels =  np.expand_dims(pandas_df['click_label'].to_numpy(), axis=-1)
        order_labels = np.expand_dims(pandas_df['order_label'].to_numpy(), axis=-1)
        gmv_labels = np.expand_dims(pandas_df['gmv_label'].to_numpy(), axis=-1)
        
        pandas_df = pandas_df.drop(columns=['click_label','order_label','gmv_label']).applymap(lambda x: [x])
        features = pandas_df.to_dict('list')
        yield features, {'click_labels': click_labels, 'order_labels': order_labels, 'gmv_labels':gmv_labels}
        
def input_fn(input_table, partitions, num_epochs=None, batch_size=1024):
    dataset = tf.data.Dataset.from_generator(
        generator = lambda: generator(input_table, partitions, batch_size),
        output_types=(
            {i.feature_name: i.dtype for i in get_features_info()},
            {"click_labels": tf.float32 , "order_labels": tf.float32, "gmv_labels": tf.float32}
            
        ),
        output_shapes=(
            {i.feature_name: tf.TensorShape([None, 1]) for i in get_features_info()},
            {"click_labels": tf.TensorShape([None, 1]) , "order_labels": tf.TensorShape([None, 1]), "gmv_labels": tf.TensorShape([None, 1])}
        )
    )

    dataset = dataset.repeat(num_epochs)

    features, labels = dataset.make_one_shot_iterator().get_next()
    return features, labels



"""PLE"""
def model_fn(features, labels, mode, params, config):
    logger.warning(f'model_fn params: {params}')
    
    hidden_dims = params['hidden_dims']
    feature_columns = params['feature_columns']
    alpha = params['alpha']

    example_name = 'car_name'  
    batch_size = tf.shape(input=features[example_name])[0] # 360
    
    n_task = 3
    n_experts = [4,4,4]
    n_expert_share = 4
    expert_dim = 32
    dnn_reg_l2 = 0.0005  # 1e-6

    dnn_hidden_units = (512,128)
    drop_rate = 0.1
    targets = ['click_labels','order_labels','gmv_labels']

    reshaped_features = {}

    for feature_info in get_features_info():
        # [batch_size, 1] -> [batch_size, 1]
        reshaped_features[feature_info.feature_name] = tf.reshape(features[feature_info.feature_name], [batch_size, 1])

    x = tf.keras.layers.DenseFeatures(feature_columns)(reshaped_features) # (?, 1368)
    
    for num in dnn_hidden_units:
        input_embed = Dropout(drop_rate)(Dense(num,activation = 'relu', kernel_regularizer=regularizers.l2(dnn_reg_l2))(x)) # (?, 128)
    
    # 添加BN层
    bn = tf.layers.BatchNormalization(axis=-1, center=True, scale=True, trainable=True)
    input_embed_bn = bn(input_embed) # (?, 128)
    # 调用ple模型
    towers = PleLayer(n_task,n_experts,expert_dim,n_expert_share)(input_embed_bn)
    # 输出每个任务的output   shape=(?, 1)  # outputs[0]  click pred ; outputs[1] order pred
    # 分类任务：是否点击/下单，二分类
    outputs_cls = [Dense(1,activation = 'sigmoid',kernel_regularizer=regularizers.l2(dnn_reg_l2), name = f,use_bias = True)(_t) for f,_t in zip(['click_labels','order_labels'],towers)]
    # 回归任务：gmv
    outputs_reg = [Dense(1,activation = 'relu',kernel_regularizer=regularizers.l2(dnn_reg_l2), name = f,use_bias = True)(_t) for f,_t in zip(['gmv_labels'],towers)]
    
    click_pre = outputs_cls[0]
    order_pre = outputs_cls[1]
    gmv_pre = outputs_reg[0]
    
    y = order_pre
    threshold = 0.5  # 标签阈值设置
    one = tf.ones_like(y)  # 生成与y大小一致的值全部为1的矩阵
    zero = tf.zeros_like(y)
    pred = tf.where(y < threshold, x=zero, y=one)  # 数值小于0.5置0，大于0.5置1

    predictions = {
        "click_pre": click_pre,
        "order_pre": order_pre,
        "gmv_pre": gmv_pre,
        "classes": pred,
        "y": y
    }

    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=y)  # [batch_size, 2]
    
    """BinaryCrossentropy"""
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    click_cross_entropy = tf.reduce_mean(bce(labels['click_labels'], click_pre)) 
    order_cross_entropy = tf.reduce_mean(bce(labels['order_labels'], order_pre))
    gmv_mse = tf.losses.mean_squared_error(labels['gmv_labels'], gmv_pre)

    
    print("成功")
    """
    损失函数：
        1、三个任务直接相加
        2、三个任务手动设置权重：点击*0.2 + 下单*0.6 + gmv*0.2
        3、三个任务通过线性规划进行动态设置权重
    """
    # 手动设置权重
    # loss1 = 0.2*click_cross_entropy + 0.6*order_cross_entropy + 0.2*gmv_mse
    loss1 = 0.2*click_cross_entropy + 0.8*order_cross_entropy
    
    tf.summary.scalar('loss1', click_cross_entropy)
    tf.summary.scalar('loss2', order_cross_entropy)
    tf.summary.scalar('loss3', gmv_mse)
    
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            'click_metric/auc': auc(labels['click_labels'], click_pre),
            'order_metric/auc': auc(labels['order_labels'], order_pre),
            'gmv_metric/mse': mse(labels['gmv_labels'], gmv_pre),
        }
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss1, eval_metric_ops=eval_metric_ops)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=3e-5) # 3e-4
        train_op = optimizer.minimize(
            loss=loss1,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss1, train_op=train_op)

    raise ValueError('mode={} unrecognized'.format(mode))
    

def serving_input_receiver_fn():
    feature_tensors = {
        i.feature_name: tf.placeholder(i.dtype, shape=(None, 1), name=i.feature_name) for i in get_features_info()
    }
    return tf.estimator.export.ServingInputReceiver(feature_tensors, feature_tensors)

def save_and_update(estimator, update=False, model_name=None):
    ''' 保存模型 & 更新线上模型（可选）'''
    if estimator.config.is_chief or estimator.config.task_id == 0:
        export_path = os.path.join(estimator.model_dir, 'export')
        export_dir = estimator.export_saved_model(
            export_dir_base=export_path,
            serving_input_receiver_fn=lambda: serving_input_receiver_fn()
        )
        logger.warning(f'export dir: {export_dir.decode()}')

        tar_file = os.path.join(os.path.dirname(export_dir.decode()), os.path.basename(export_dir.decode()) + ".tar.gz")
        archive_utils.tar_archive_with_file(export_dir.decode(), tar_file)
        oss_key = oss_utils.generate_oss_key(os.path.basename(export_dir.decode()) + ".tar.gz")
        http_url = oss_utils.put_file_to_oss(tar_file, oss_key)
        logger.warning(f'tensorflow model saved to: {http_url}')

        if (update):
            if (model_name is None):
                raise ValueError('model name cannot be None if update is True')
            os.chdir(os.path.dirname(export_dir.decode()))
            os.rename(os.path.basename(export_dir.decode()), f'{model_name}')
            os.system(f'zip -r {model_name}.zip {model_name}')
            url = f'https://aiengine-aibrain-inner.hellobike.cn/aibrain/api/v1/modelversion/internal/autoupdate/{model_name}?updatedBy=lizhiqiang195'
            response = requests.request(method='PATCH', url=url, files=[('uploadFile', open(f'{model_name}.zip', 'rb'))])
            logger.warning('response text: ' + response.text)

    return





def main(unused_argv):
    """参数设置""" 
    # estimator_params = param_utils.get_object_from_input_param('estimator_params')
    # train_partition = param_utils.get_object_from_input_param('train_partition')
    # eval_partition = param_utils.get_object_from_input_param('eval_partition')
    # max_steps = param_utils.get_object_from_input_param('max_steps')
    # batch_size = param_utils.get_object_from_input_param('batch_size')
    
    max_steps = 50000   # 200000
    batch_size = 32

    """一天训练一天预测"""
#     train_partition = [['pt=20241222']]
#     eval_partition = [['pt=20241224']]
#     estimator_params = {
#         'hidden_dims': [128, 64, 32],
#         'feature_columns': get_feature_columns(['20241222','20241224']),
#         'alpha': 0.5
#     }
    
    
    date_converter = date_convert_utils.DateConvertUtils()
    train_pt = date_converter.parse_data_date("${yyyymmdd-2}")
    eval_pt = date_converter.parse_data_date("${yyyymmdd-1}")
    train_partition = [['pt=' + str(train_pt)]] # [['pt=20240709']]
    eval_partition = [['pt=' + str(eval_pt)]] # [['pt=20240710']]
    
    print("训练数据pt:",train_partition)
    print("测试数据pt:",eval_partition)
    
    estimator_params = {
        'hidden_dims': [128, 64, 32],
        'feature_columns': get_feature_columns([train_pt, eval_pt]), # '20240709','20240710'
        'alpha': 0.5
    }
    
    
    seed = 2024
    tf.set_random_seed(seed)  # 为TensorFlow设置全局随机种子

    train_table = train_data_df
    eval_table = train_data_df

    config, _ = tf_context.get_tf_config(
        save_checkpoints_secs=None,
        save_checkpoints_steps=500,
        keep_checkpoint_max=5
    )

    ranker = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=config.model_dir,
        config=config,
        params=estimator_params
    )

    train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(train_table, train_partition, batch_size=batch_size), max_steps=max_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(eval_table, eval_partition, num_epochs=1, batch_size=batch_size))

    print("result 开始")
    tf.estimator.train_and_evaluate(ranker, train_spec, eval_spec)
    print("result 结束")
    
    
    
    print("train end...")
    print("*"*100)
    print("*"*100)
    save_and_update(ranker, update=False)
        
    # 模型自动更新
#     online_model_name = 'RentCarIndividualSortModelV3AutoUpdate'
#     if accuracy[0]['order_metric/auc'] > 0.85:
#         save_and_update(ranker, update = True, model_name = online_model_name)
#     else:
#         save_and_update(ranker, update=False)    
#     saved_model_dir = save_and_update(ranker, update=False)

    online_model_name = 'RentCarIndividualSortModelV3AutoUpdate'
    save_and_update(ranker, update = True, model_name = online_model_name)
    saved_model_dir = save_and_update(ranker, update=False)
    logging.info(f"============================================================模型迭代完成============================================================")

    logger.warning(f'model size is {sum([ranker.get_variable_value(var).size for var in ranker.get_variable_names()])}')
    
    
    return



if __name__ == '__main__':
    tf.app.run()