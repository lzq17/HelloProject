import tensorflow as tf
from pyspark.sql import SparkSession

from config import train_data_df
from spark_session_utils import SparkSessionHelper


class Feature():
    def __init__(self, feature_name, dtype, default_value, feature_type):
        """
        Args:
            feature_name (string): 特征名称
            dtype (tf.dtypes.DType): 特征原始数据类型
            default_value: 特征默认值
        """
        self.feature_name = feature_name
        self.dtype = dtype
        self.default_value = default_value
        self.feature_type = feature_type

    def get_feature_column(self):
        raise NotImplementedError('Calling an abstract method.')

# 分箱
class Bucketize(Feature):
    # boundaries=[0., 1., 2.] generates buckets (-inf, 0.), [0., 1.), [1., 2.), and [2., +inf).
    def __init__(self, feature_name, boundaries, default_value, feature_type):
        self.boundaries = boundaries
        super().__init__(feature_name=feature_name, dtype=tf.float32, default_value=default_value, feature_type=feature_type)

    # bucketized_column：分箱，把一个连续的数字范围分成几段
    # numeric_column：用于抽取数值类型的特征，即dense特征
    # boundaries：指定边界的已排序列表或浮点数元组
    # # dimension 指定嵌入维度的整数
    def get_feature_column(self):
        return tf.feature_column.embedding_column(
            tf.feature_column.bucketized_column(
                tf.feature_column.numeric_column(self.feature_name),
                self.boundaries
            ),
            dimension=int(round(6 * (len(self.boundaries) ** 0.25)))
        )

# 处理成ID类特征0、1、2、3...
# vocabulary_list 定义词汇表的有序迭代。每个特征都映射到 vocabulary_list 中其值的索引(如果存在)
class Identity(Feature):
    def __init__(self, feature_name, max_value, default_value, feature_type):
        self.vocabulary_list = list(range(max_value + 1))
        if (default_value not in self.vocabulary_list):
            self.vocabulary_list.append(default_value)
        super().__init__(feature_name=feature_name, dtype=tf.int32, default_value=default_value, feature_type=feature_type)

    def get_feature_column(self):
        return tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_vocabulary_list(
                self.feature_name,
                self.vocabulary_list
            ),
            dimension=int(round(6 * (len(self.vocabulary_list) ** 0.25)))
        )

# 类别特征
class Category(Feature):
    def __init__(self, feature_name, default_value, feature_type):
        super().__init__(feature_name=feature_name, dtype=tf.string, default_value=default_value, feature_type=feature_type)

    def get_feature_column(self, pt):
        self.vocabulary_list = self.gen_vocabulary(pt)
        if (self.default_value not in self.vocabulary_list):
            self.vocabulary_list.append(self.default_value)
        return tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_vocabulary_list(
                self.feature_name,
                self.vocabulary_list
            ),
            dimension=int(round(6 * (len(self.vocabulary_list) ** 0.25)))
        )

    def gen_vocabulary(self, pt):
        pt = ['\'' + i + '\'' for i in pt]
        spark = SparkSessionHelper().get()
        return spark.sql(f'''
            select distinct
                {self.feature_name}
            from
                {train_data_df}
            where
                pt in ({','.join(pt)})
        ''').toPandas()[self.feature_name].tolist() # .map(lambda x: x[0]).unique().tolist()

class Rescaling(Feature):  
    def __init__(self, feature_name, max_value, default_value, feature_type):
        self.max_value = max_value
        super().__init__(feature_name=feature_name, dtype=tf.float32, default_value=default_value, feature_type=feature_type)

    def normalizer_fn(self, value):
        return tf.keras.backend.maximum(tf.keras.backend.minimum(value / self.max_value, 1), 0)

    def get_feature_column(self):
        return tf.feature_column.numeric_column(
            self.feature_name, 
            normalizer_fn=self.normalizer_fn
        )

class LogRescaling(Feature):
    def __init__(self, feature_name, max_value, default_value, feature_type):
        self.max_value = max_value
        super().__init__(feature_name=feature_name, dtype=tf.float32, default_value=default_value, feature_type=feature_type)

    def normalizer_fn(self, value):
        return tf.keras.backend.maximum(tf.keras.backend.minimum(tf.math.log1p(tf.keras.backend.cast(value, tf.float32)) / tf.math.ceil(tf.math.log1p(tf.keras.backend.cast(self.max_value, tf.float32))), 1), 0)

    def get_feature_column(self):
        return tf.feature_column.numeric_column(
            self.feature_name, 
            normalizer_fn=self.normalizer_fn
        )


def get_features_info():
    return [
        
        Category('licensetag', '', ['item']),  # 粤A牌/沪牌  【string】
        Category('platform','others',['user']), # 应用平台 Android iOS H5 Alipay 【string】
        Category('city_code', '', ['user']),  # 城市code 【string】
        Category('usr_pub_usual_active_city_code', '', ['user']),  # 城市code 【string】
        Category('ad_code','',['user']), #  GPS定位县级code  【string】
        Category('adsource','',['item']), #  adsource  【string】
        Category('displacement','',['item']), #  排量 【string】
        Category('car_name','',['item']), #  车名 【string】
        
        Category('first_pay_group_name', '', ['item']),  
        Category('last_pay_group_name', '', ['item']), 
        Category('last_complete_group_name', '', ['item']), 
        
        Bucketize('zu_qi_deal', [0, 10, 20, 30, 40, 50, 100, 120, 140, 160, 200], 0, ['item']), # 租期 【int】
        Bucketize('user_label', [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 0, ['item']), 
        Bucketize('min_diff', [0, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500], 0, ['item']), 
        Bucketize('order_create', [0, 50, 100, 200, 400, 600, 800, 1000], 0, ['item']), 
        Bucketize('order_pay', [0, 50, 100, 200, 400, 600, 800, 1000], 0, ['item']), 
        Bucketize('order_finish', [0, 50, 100, 200, 400, 600, 800, 1000], 0, ['item']), 
        Bucketize('homepage_pv', [0, 500, 1000, 2000, 3000, 4000, 5000], 0, ['user']),
        Bucketize('last_view_diff', [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 0, ['item']), 
        Bucketize('click_price_cnt', [0, 500, 1000, 2000, 4000, 6000, 8000, 10000], 0, ['item']), 
        Bucketize('click_group_cnt', [0, 500, 1000, 2000, 4000, 6000, 8000, 10000], 0, ['item']), 
        Bucketize('order_price_cnt', [0, 500, 1000, 2000, 4000, 6000, 8000, 10000], 0, ['item']), 
        Bucketize('order_cartype_cnt', [0, 500, 1000, 2000, 4000, 6000, 8000, 20000], 0, ['item']), 
        Bucketize('order_group_cnt', [0, 500, 1000, 2000, 4000, 6000, 8000, 10000], 0, ['item']), 
        Bucketize('click_price_cnt_7_day', [0, 500, 1000, 1500, 2000, 3000, 4000, 5000], 0, ['item']), 
        Bucketize('click_cartype_cnt_7_day', [0, 500, 1000, 2000, 4000, 6000, 8000, 10000], 0, ['item']), 
        Bucketize('click_group_cnt_7_day', [0, 500, 1000, 1500, 2000, 3000, 4000, 5000], 0, ['item']), 
        Bucketize('order_price_cnt_7_day', [0, 500, 1000, 1500, 2000, 3000, 4000, 5000], 0, ['item']), 
        Bucketize('order_cartype_cnt_7_day', [0, 500, 1000, 2000, 4000, 6000, 8000, 10000], 0, ['item']), 
        Bucketize('order_group_cnt_7_day', [0, 500, 1000, 1500, 2000, 3000, 4000, 5000], 0, ['item']), 
        Bucketize('click_price_cnt_14_day', [0, 500, 1000, 2000, 4000, 6000, 8000, 10000], 0, ['item']), 
        Bucketize('click_cartype_cnt_14_day', [0, 500, 1000, 2000, 4000, 6000, 8000, 15000], 0, ['item']),
        Bucketize('click_group_cnt_14_day', [0, 500, 1000, 2000, 4000, 6000, 8000, 10000], 0, ['item']), 
        Bucketize('order_price_cnt_14_day', [0, 500, 1000, 2000, 4000, 6000, 8000, 10000], 0, ['item']), 
        Bucketize('order_cartype_cnt_14_day', [0, 500, 1000, 2000, 4000, 6000, 8000, 15000], 0, ['item']),
        Bucketize('order_group_cnt_14_day', [0, 500, 1000, 2000, 4000, 6000, 8000, 10000], 0, ['item']), 
        Bucketize('first_pay_diff_dt', [0, 500, 1000, 1500, 2000, 3000, 4000, 5000], 0, ['item']), 
        Bucketize('last_pay_diff_dt', [0, 500, 1000, 1500, 2000, 3000, 4000, 5000], 0, ['item']), 
        Bucketize('rentcars_homepage_cnt_3d', [0, 50, 100, 200, 300, 400, 500], 0, ['item']),
        Bucketize('rentcars_homepage_cnt_7d', [0, 50, 100, 200, 300, 400, 500], 0, ['item']),
        Bucketize('rentcars_homepage_cnt_14d', [0, 50, 100, 200, 400, 600, 800, 1000], 0, ['item']),
        Bucketize('rentcars_homepage_cnt_30d', [0, 50, 100, 200, 400, 500, 1000, 2000], 0, ['item']),
        Bucketize('rentcars_cars_cnt_3d', [0, 50, 100, 200, 300, 400, 500], 0, ['item']),
        Bucketize('rentcars_cars_cnt_7d', [0, 50, 100, 200, 300, 400, 500], 0, ['item']),
        Bucketize('rentcars_cars_cnt_14d', [0, 50, 100, 200, 400, 600, 800, 1000], 0, ['item']),
        Bucketize('rentcars_cars_cnt_30d', [0, 50, 100, 200, 400, 600, 800, 1000], 0, ['item']),
        Bucketize('rentcars_order_cnt_3d', [0, 50, 100, 200, 300, 400, 500], 0, ['item']),
        Bucketize('rentcars_order_cnt_7d', [0, 50, 100, 200, 400, 600, 800, 1000], 0, ['item']),
        Bucketize('rentcars_order_cnt_14d', [0, 100, 200, 500, 1000, 2000, 3000], 0, ['item']),
        Bucketize('rentcars_order_cnt_30d', [0, 500, 1000, 1500, 2000, 3000, 4000, 5000], 0, ['item']), 
        Bucketize('rentcars_orderdetails_cnt_3d', [0, 50, 100, 120, 140, 160, 200], 0, ['item']),
        Bucketize('rentcars_orderdetails_cnt_7d', [0, 50, 100, 120, 140, 160, 200], 0, ['item']),
        Bucketize('rentcars_orderdetails_cnt_14d', [0, 50, 100, 200, 300, 400, 500], 0, ['item']),
        Bucketize('rentcars_orderdetails_cnt_30d', [0, 50, 100, 200, 300, 400, 500], 0, ['item']),
        Bucketize('rentcars_homepage_day_3d', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0, ['item']), 
        Bucketize('rentcars_homepage_day_7d', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0, ['item']), 
        Bucketize('rentcars_homepage_day_14d', [0, 2, 4, 6, 8, 10, 15, 20, 25, 30], 0, ['item']), 
        Bucketize('rentcars_homepage_day_30d', [0, 2, 4, 6, 8, 10, 20, 30, 40, 50], 0, ['item']), 
        Bucketize('rentcars_cars_day_3d', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0, ['item']), 
        Bucketize('rentcars_cars_day_7d', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0, ['item']), 
        Bucketize('rentcars_cars_day_14d', [0, 2, 4, 6, 8, 10, 15, 20, 25, 30], 0, ['item']), 
        Bucketize('rentcars_cars_day_30d', [0, 2, 4, 6, 8, 10, 20, 30, 40, 50], 0, ['item']), 
        Bucketize('rentcars_order_day_3d', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0, ['item']), 
        Bucketize('rentcars_order_day_7d', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0, ['item']), 
        Bucketize('rentcars_order_day_14d', [0, 2, 4, 6, 8, 10, 15, 20, 25, 30], 0, ['item']), 
        Bucketize('rentcars_order_day_30d', [0, 2, 4, 6, 8, 10, 20, 30, 40, 50], 0, ['item']), 
        Bucketize('rentcars_orderdetails_day_3d', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0, ['item']), 
        Bucketize('rentcars_orderdetails_day_7d', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0, ['item']), 
        Bucketize('rentcars_orderdetails_day_14d', [0, 2, 4, 6, 8, 10, 15, 20, 25, 30], 0, ['item']), 
        Bucketize('rentcars_orderdetails_day_30d', [0, 2, 4, 6, 8, 10, 20, 30, 40, 50], 0, ['item']), 
        Bucketize('sex_age_level_cross_group', [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000], 0, ['item']),
        Bucketize('sex_is_stu_user_cross_group', [0, 500000, 1000000, 1500000, 2000000, 2500000, 3000000], 0, ['item']),
        Bucketize('sex_consume_level_cross_group', [0, 400000, 800000, 1000000, 1400000, 1800000, 2000000], 0, ['item']),
        Bucketize('sex_if_yidi_rent_cross_group', [0, 400000, 800000, 1000000, 1400000, 1800000, 2000000], 0, ['item']),
        Bucketize('sex_age_level_is_stu_cross_group', [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000], 0, ['item']),
        Bucketize('sex_age_level_consume_level_cross_group', [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000], 0, ['item']),
        Bucketize('sex_age_level_yidi_cross_group', [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000], 0, ['item']),
        Bucketize('sex_age_level_consume_level_yidi_cross_group', [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000], 0, ['item']),
        Bucketize('sex_age_level_cross_price', [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1500000], 0, ['item']),
        Bucketize('sex_is_stu_user_cross_price', [0, 500000, 1000000, 2000000, 3000000, 4000000, 5000000], 0, ['item']),
        Bucketize('sex_consume_level_cross_price', [0, 500000, 1000000, 2000000, 3000000, 4000000, 5000000], 0, ['item']),
        Bucketize('sex_if_yidi_rent_cross_price', [0, 500000, 1000000, 2000000, 3000000, 4000000, 5000000], 0, ['item']),
        Bucketize('sex_age_level_is_stu_cross_price', [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1500000], 0, ['item']),
        Bucketize('sex_age_level_consume_level_cross_price', [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000], 0, ['item']),
        Bucketize('sex_age_level_yidi_cross_price', [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1500000], 0, ['item']),
        Bucketize('sex_age_level_consume_level_yidi_cross_price', [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000], 0, ['item']),
        Bucketize('new_old_cross_group', [0, 400000, 800000, 1000000, 1400000, 1800000, 2000000], 0, ['item']),
        Bucketize('new_old_cross_price', [0, 400000, 800000, 1000000, 1400000, 1800000, 2500000], 0, ['item']),
        Bucketize('new_old_sex_cross_group', [0, 400000, 800000, 1000000, 1400000, 1800000, 2000000], 0, ['item']),
        Bucketize('new_old_sex_cross_price', [0, 400000, 800000, 1000000, 1400000, 1800000, 2000000], 0, ['item']),
        Bucketize('new_old_age_level_cross_group', [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000], 0, ['item']),
        Bucketize('new_old_age_level_cross_price', [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000], 0, ['item']),
        Bucketize('new_old_stu_cross_group', [0, 400000, 800000, 1000000, 1400000, 1800000, 2000000], 0, ['item']),
        Bucketize('new_old_stu_cross_price', [0, 400000, 800000, 1000000, 1400000, 1800000, 2500000], 0, ['item']),
        Bucketize('new_old_yidi_cross_group', [0, 400000, 800000, 1000000, 1400000, 1800000, 2000000], 0, ['item']),
        Bucketize('new_old_yidi_cross_price', [0, 400000, 800000, 1000000, 1400000, 1800000, 2500000], 0, ['item']),
        Bucketize('new_old_consume_cross_group', [0, 400000, 800000, 1000000, 1400000, 1800000, 2000000], 0, ['item']),
        Bucketize('new_old_consume_cross_price', [0, 400000, 800000, 1000000, 1400000, 1800000, 2000000], 0, ['item']),
        Bucketize('new_old_sex_age_cross_group', [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000], 0, ['item']),
        Bucketize('new_old_sex_age_cross_price', [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000], 0, ['item']),
        Bucketize('new_old_sex_stu_cross_group', [0, 400000, 800000, 1000000, 1400000, 1800000, 2000000], 0, ['item']),
        Bucketize('new_old_sex_stu_cross_price', [0, 400000, 800000, 1000000, 1400000, 1800000, 2000000], 0, ['item']),
        Bucketize('new_old_sex_yidi_cross_group', [0, 400000, 800000, 1000000, 1400000, 1800000, 2000000], 0, ['item']),
        Bucketize('new_old_sex_yidi_cross_price', [0, 400000, 800000, 1000000, 1400000, 1800000, 2000000], 0, ['item']),
        Bucketize('new_old_sex_consume_cross_group', [0, 400000, 800000, 1000000, 1400000, 1800000, 2000000], 0, ['item']),
        Bucketize('new_old_sex_consume_cross_price', [0, 400000, 800000, 1000000, 1400000, 1800000, 2000000], 0, ['item']),
        Bucketize('new_old_sex_age_stu_cross_group', [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000], 0, ['item']),
        Bucketize('new_old_sex_age_stu_cross_price', [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000], 0, ['item']),
        Bucketize('new_old_sex_age_yidi_cross_group', [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000], 0, ['item']),
        Bucketize('new_old_sex_age_yidi_cross_price', [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000], 0, ['item']),
        Bucketize('new_old_sex_age_consume_cross_group', [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000], 0, ['item']),
        Bucketize('new_old_sex_age_consume_cross_price', [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000], 0, ['item']),
        Bucketize('xl_sex_age_stu_yd_cross_group', [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000], 0, ['item']),
        Bucketize('xl_sex_age_stu_yd_cross_price', [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000], 0, ['item']),
        Bucketize('xl_sex_age_stu_co_cross_group', [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000], 0, ['item']),
        Bucketize('xl_sex_age_stu_co_cross_price', [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000], 0, ['item']),
        Bucketize('xl_sex_age_stu_yd_co_cross_group', [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000], 0, ['item']),
        Bucketize('xl_sex_age_stu_yd_co_cross_price', [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000], 0, ['item']),
        Bucketize('evaluate_add_info', [0, 500, 1000, 2000, 4000, 6000, 8000, 10000], 0, ['item']), # 评价条数 【int】
        Bucketize('exp_uv_7d', [0, 50000, 100000, 150000, 200000, 250000, 300000], 0, ['item']),# 7日曝光uv  【bigint】
        Bucketize('exp_uv_14d', [0, 100000, 200000, 300000, 400000, 500000, 600000], 0, ['item']),# 14日曝光uv  【bigint】
        Bucketize('exp_uv_30d', [0, 200000, 400000, 600000, 800000, 900000, 1000000], 0, ['item']),# 30日曝光uv  【bigint】
        Bucketize('exp_pv_7d', [0, 200000, 400000, 600000, 800000, 900000, 1000000], 0, ['item']),# 7日曝光pv  【bigint】
        Bucketize('exp_pv_14d', [0, 400000, 800000, 1000000, 1400000, 1800000, 2000000], 0, ['item']),# 14日曝光pv  【bigint】
        Bucketize('exp_pv_30d', [0, 800000, 1000000, 2000000, 3000000, 40000000, 50000000], 0, ['item']),# 30日曝光pv  【bigint】
        Bucketize('clk_uv_7d', [0, 8000, 10000, 20000, 30000, 40000, 50000], 0, ['item']),# 7日曝光uv  【bigint】
        Bucketize('clk_uv_14d', [0, 10000, 30000, 50000, 60000, 70000, 80000], 0, ['item']),# 14日曝光uv  【bigint】
        Bucketize('clk_uv_30d', [0, 30000, 60000, 90000, 120000, 150000, 180000], 0, ['item']),# 30日曝光uv  【bigint】
        Bucketize('clk_pv_7d', [0, 10000, 20000, 40000, 60000, 80000, 100000], 0, ['item']),# 7日曝光pv  【bigint】
        Bucketize('clk_pv_14d', [0, 40000, 80000, 120000, 160000, 180000, 200000], 0, ['item']),# 14日曝光pv  【bigint】
        Bucketize('clk_pv_30d', [0, 60000, 120000, 180000, 240000, 300000, 360000], 0, ['item']),# 30日曝光pv  【bigint】
        Bucketize('ordercreate_cnt_7d', [0, 500, 1000, 2000, 4000, 6000, 8000, 10000], 0, ['item']),# 7日创单  【bigint】
        Bucketize('ordercreate_cnt_14d', [0, 4000, 8000, 12000, 14000, 18000, 22000], 0, ['item']),# 14日创单  【bigint】
        Bucketize('ordercreate_cnt_30d', [0, 8000, 12000, 18000, 22000, 28000, 38000], 0, ['item']),# 30日创单  【bigint】
        Bucketize('ordercreate_cnt_60d', [0, 10000, 30000, 40000, 50000, 60000, 70000], 0, ['item']),# 60日创单  【bigint】
        Bucketize('ordercreate_cnt_90d', [0, 10000, 30000, 50000, 70000, 90000, 110000], 0, ['item']),# 90日创单  【bigint】
        Bucketize('ordercreate_cnt', [0, 100000, 200000, 300000, 400000, 500000, 600000], 0, ['item']),# 创单  【bigint】
        Bucketize('orderpay_cnt_7d', [0, 500, 1000, 2000, 4000, 6000, 8000, 10000], 0, ['item']),# 7日完单  【bigint】
        Bucketize('orderpay_cnt_14d', [0, 4000, 8000, 12000, 14000, 18000, 22000], 0, ['item']),# 14日完单  【bigint】
        Bucketize('orderpay_cnt_30d', [0, 4000, 8000, 12000, 20000, 25000, 30000], 0, ['item']),# 30日完单  【bigint】
        Bucketize('orderpay_cnt_60d', [0, 10000, 20000, 30000, 40000, 50000, 60000], 0, ['item']),# 60日完单  【bigint】
        Bucketize('orderpay_cnt_90d', [0, 10000, 20000, 30000, 40000, 60000, 80000], 0, ['item']),# 90日完单  【bigint】
        Bucketize('orderpay_cnt', [0, 50000, 100000, 150000, 200000, 300000, 400000], 0, ['item']),# 完单  【bigint】
        Bucketize('rentcars_orderpay_cnt_7d', [0, 4, 8, 12, 16, 20, 24], 0, ['user']),# 用户7日完单  【bigint】
        Bucketize('rentcars_orderpay_cnt_14d', [0, 4, 8, 12, 18, 24, 30], 0, ['user']),# 用户14日完单  【bigint】
        Bucketize('rentcars_orderpay_cnt_30d', [0, 5, 10, 20, 30, 40, 50], 0, ['user']),# 用户30日完单  【bigint】
        Bucketize('rentcars_orderpay_cnt_60d', [0, 10, 20, 40, 60, 80, 100], 0, ['user']),# 用户60日完单  【bigint】
        Bucketize('rentcars_orderpay_cnt_90d', [0, 20, 40, 60, 80, 100, 120], 0, ['user']),# 用户90日完单  【bigint】
        Bucketize('rentcars_orderpay_cnt', [0, 50, 100, 200, 300, 400, 500], 0, ['user']),# 用户完单  【bigint】
        Bucketize('sex_cross_group', [0, 400000, 800000, 1200000, 1600000, 2000000, 3000000], 0, ['user']), # 交叉特征
        Bucketize('sex_cross_price', [0, 400000, 800000, 1200000, 1600000, 2000000, 3000000], 0, ['user']), # 交叉特征
        Bucketize('age_level_cross_group', [0, 100000, 200000, 400000, 600000, 800000, 1000000], 0, ['user']), # 交叉特征
        Bucketize('age_level_cross_price', [0, 100000, 200000, 400000, 600000, 800000, 1000000], 0, ['user']), # 交叉特征
        Bucketize('is_stu_user_cross_group', [0, 400000, 800000, 1200000, 1600000, 2000000, 3000000], 0, ['user']), # 交叉特征
        Bucketize('is_stu_user_cross_price', [0, 400000, 800000, 1200000, 1600000, 2000000, 3000000], 0, ['user']), # 交叉特征
        Bucketize('consume_level_cross_group', [0, 400000, 800000, 1200000, 1600000, 2000000, 3000000], 0, ['user']), # 交叉特征
        Bucketize('consume_level_cross_price', [0, 400000, 800000, 1200000, 1600000, 2000000, 3000000], 0, ['user']), # 交叉特征
        Bucketize('yidi_rent_cross_group', [0, 400000, 800000, 1200000, 1600000, 2000000, 3000000], 0, ['user']), # 交叉特征
        Bucketize('yidi_rent_cross_price', [0, 400000, 800000, 1200000, 1600000, 2000000, 3000000], 0, ['user']), # 交叉特征
        
        
        Identity('if_yidi_rent', 1, -1, ['user']),  # 是否异地 【int】
        Identity('usr_pub_sex', 1, -1, ['user']),  # '性别 0男 1女 -1未知' 【bigint】
        Identity('usr_pub_is_stu_user', 1, -1, ['user']),  # 是否学生 【bigint】
        Identity('back_method',3, -1,['item']), #  还车方式（1-用户自行还车,2-商家免费上门取车,3-商家上门取车（收费）,4-用户还车到门店后商家送回目的地） 【int】
        Identity('pickup_method',3, -1,['item']), #  取车类型 1:接用户到门店取车,2:送车上门,3:自行取车 【int】
        Identity('transmission_type', 2, -1, ['item']),  # 1:自动,2:手动 【int】
        Identity('passenger_no', 20, 0, ['item']),  # 乘客数 【int】
        Identity('door_no', 6, 2, ['item']),  # 车门数 【int】
        Identity('usr_pub_consume_level', 3, 0, ['user']),  # 消费能力等级，0-未知,1:低消费,2:中消费,3:高消费 【int】
        Identity('site_score', 5, -9999, ['item']),  # 评分
        Identity('total_price_level', 16, 0, ['item']),  # 总价分层
        Identity('avg_price_level', 7, 0, ['item']),  # 平均价分层
        Identity('usr_pub_age_level', 7, 0, ['item']),  # 年龄分层
        Identity('num_terms', 4, 0, ['item']),  # terms数
        Identity('usr_pub_platform', 3, -1, ['item']),  # platform
        Identity('usr_pub_usual_active_city_lvl', 4, -9999, ['item']),  # 常活跃城市等级
        Identity('is_new_user', 1, -1, ['user']),  # 新老客
        
        
        Rescaling('last_pay_day_avg_real_pay',20000,0,['item']),
        Rescaling('sum_rent_day_no_cancel',1000,0,['item']),
        Rescaling('avg_rent_day_no_cancel',100,0,['item']),
        
        LogRescaling('total_price', 10000000.0, 0, ['item']), #总价 【double】
        LogRescaling('ave_price', 1000000.0, 0, ['item']), #平均价 【double】
        LogRescaling('orderpay_gmv_7d', 3000000.0, 0, ['item']), #7日gmv 【double】
        LogRescaling('orderpay_gmv_14d', 5000000.0, 0, ['item']), #14日gmv 【double】
        LogRescaling('orderpay_gmv_30d', 10000000.0, 0, ['item']), #30日gmv 【double】
        LogRescaling('orderpay_gmv_60d', 20000000.0, 0, ['item']), #60日gmv 【double】
        LogRescaling('orderpay_gmv_90d', 30000000.0, 0, ['item']), #90日gmv 【double】
        LogRescaling('orderpay_gmv', 200000000.0, 0, ['item']), #gmv 【double】
        LogRescaling('rentcars_orderpay_gmv_7d', 30000.0, 0, ['user']), #用户7日gmv 【double】
        LogRescaling('rentcars_orderpay_gmv_14d', 30000.0, 0, ['user']), #用户14日gmv 【double】
        LogRescaling('rentcars_orderpay_gmv_30d', 40000.0, 0, ['user']), #用户30日gmv 【double】
        LogRescaling('rentcars_orderpay_gmv_60d', 70000.0, 0, ['user']), #用户60日gmv 【double】
        LogRescaling('rentcars_orderpay_gmv_90d', 500000.0, 0, ['user']), #用户90日gmv 【double】
        LogRescaling('rentcars_orderpay_gmv', 700000.0, 0, ['user']), #用户gmv 【double】

    ]




# def get_feature_columns(features_info_partition):
#     feature_columns = [] # {}

#     for feature_info in get_features_info():
#         if (isinstance(feature_info, Category)):
#             feature_column = feature_info.get_feature_column(features_info_partition)
#         else:
#             feature_column = feature_info.get_feature_column()
#         for feature_type in feature_info.feature_type:
#             if feature_type not in feature_columns:
#                 feature_columns[feature_type] = [feature_column]
#             else:
#                 feature_columns[feature_type].append(feature_column)
#     return feature_columns



def get_feature_columns(features_info_partition):
    feature_columns = []

    for feature_info in get_features_info():
        if (isinstance(feature_info, Category)):
            feature_column = feature_info.get_feature_column(features_info_partition)
        else:
            feature_column = feature_info.get_feature_column()

        feature_columns.append(feature_column)

    return feature_columns