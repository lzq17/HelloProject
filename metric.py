import tensorflow as tf
from tfr2 import metrics as tfr2_metrics


def labels_mean(labels, logits):
    is_label_valid = tf.reshape(tf.greater_equal(labels, 0.), [-1])
    m = tf.keras.metrics.Mean()
    m.update_state(tf.boolean_mask(tensor=tf.reshape(labels, [-1]), mask=is_label_valid))
    return m

def logits_mean(labels, logits):
    is_label_valid = tf.reshape(tf.greater_equal(labels, 0.), [-1])
    m = tf.keras.metrics.Mean()
    m.update_state(tf.boolean_mask(tensor=tf.reshape(logits, [-1]), mask=is_label_valid))
    return m

def auc(labels, logits):
    """Returns metrics for labels and logits."""
    is_label_valid = tf.reshape(tf.greater_equal(labels, 0.), [-1])
    m = tf.keras.metrics.AUC()
    m.update_state(
        tf.boolean_mask(tensor=tf.reshape(labels, [-1]), mask=is_label_valid),
        tf.sigmoid(tf.boolean_mask(tensor=tf.reshape(logits, [-1]), mask=is_label_valid))
    )
    return m

def gauc(labels, logits):
    """Returns metrics for labels and logits."""
    is_label_valid = tf.greater_equal(labels, 0.)
    m = tfr2_metrics.OPAMetric(ragged=True)
    m.update_state(
        tf.ragged.boolean_mask(labels, is_label_valid),
        tf.ragged.boolean_mask(logits, is_label_valid)
    )
    return m

def topk_acc(labels, logits, top_k):
    # probabilities = tf.nn.softmax(logits)
    # accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels, top_k), tf.float32))
    accuracy_in_top_k = tf.reduce_mean(tf.nn.in_top_k(logits, labels, top_k))
    return accuracy_in_top_k
    
# def mse(labels, logits):
#     is_label_valid = tf.reshape(tf.greater_equal(labels, 0.), [-1])
#     m = tf.keras.metrics.MeanSquaredError()
#     m.update_state(tf.boolean_mask(tensor=tf.reshape(logits, [-1]), mask=is_label_valid))
#     return m


def mse(labels, logits):
    # 创建一个布尔掩码，指示 labels 中哪些元素是有效的（即大于或等于0）
    is_label_valid = tf.reshape(tf.greater_equal(labels, 0.), [-1])
    # 使用 tf.boolean_mask 筛选有效的 labels 和 logits
    valid_labels = tf.boolean_mask(labels, is_label_valid)
    valid_logits = tf.boolean_mask(logits, is_label_valid)
    # 创建 MeanSquaredError 指标对象
    m = tf.keras.metrics.MeanSquaredError()
    # 更新指标状态
    m.update_state(y_true=valid_labels, y_pred=valid_logits)
    # 返回计算得到的 MSE 值
    return m



    