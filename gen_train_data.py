import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats as spt
from pyspark import SparkContext
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import Row, SparkSession, SQLContext, Window
import os, math
import arrow, pendulum, time
import json, random
from collections import OrderedDict
from ast import literal_eval
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors, VectorUDT
import pyspark.ml.feature as ft
from features_info import get_features_info

os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3.7"
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
from copy import copy

pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
from IPython.core.display import display, HTML

# display(HTML("<style>.container { width:100% !important; }</style>"))
display(HTML("<style>.container { width:100%; }</style>"))

spark = SparkSession.builder. \
    config('spark.executor.memory', '14g'). \
    config('spark.executor.cores', '6'). \
    config('spark.driver.memory', '10g'). \
    config('spark.dynamicAllocation.minExecutors', '8'). \
    config('spark.dynamicAllocation.maxExecutors', '100'). \
    config('spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version', '2'). \
    config('spark.sql.execution.arrow.enabled', 'false'). \
    config('spark.driver.maxResultSize', '200g'). \
    config('spark.sql.shuffle.partitions', '2000'). \
    config('spark.default.parallelism', '800'). \
    config("spark.sql.broadcastTimeout", "800"). \
    config('spark.sql.crossJoin.enabled', 'true'). \
    appName('rent car recommend daily sample lizhiqiang195'). \
    enableHiveSupport().getOrCreate()
# config('hive.exec.orc.split.strategy', 'ETL'). \
sc = spark.sparkContext
sqlContext = SQLContext(sc)



def trainData(pt_):
    yuchuli = spark.sql(f'''
        select * from turing_dev.rent_cars_all_feature_old_new_model where pt={pt_}
    ''')
    drop_list = [
        'user_new_id',
        'user_guid',
        'car_type',
        'avg_price',
        'group_name',
        'vehicle_display_group_id',
        'item_id',
        'rec_trace_id',
        'goods_id',
        'company_name',
        'company_code',
        'usr_pub_age',
        'item_id_click_list',
        'is_first_click_item',
        'is_last_click_item',
        'pickup_site_guid', 
        'location', 
        'activity_daily_price',
        'fixed_price',
        'coupon_amount', 
        'promotion_detail', 
        'term_type',
        'view_type',
        'term_name',
        'term_code',
        'store_distance',
        'is_zong_jia_zui_di',
        'is_shou_fei_song_qu_che',
        'is_dao_che_ying_xiang',
        'is_dao_che_lei_da',
        'is_yi_che_yi_xi',
        'is_bu_xian_li_cheng',
        'is_ban_nian_nei_xin_che',
        'is_liang_nian_nei_xin_che',
        'is_bu_xian_jia_ling',
        'is_you_liang_bao_zhang',

        'shouye_staytime',
        'carlist_staytime',
        'ord_staytime',
        'carsprice_staytime',

        'quotation_num',
        'is_mian_fei_song_qu_che',
        'is_zhi_ma_mian_ya_jin',
        'is_hl_ping_pai_dian',
        'is_fang_xin_zu',
        'is_new_energy'
    ]
    yuchuli_tmp1 = yuchuli.drop(*drop_list)
    
    yuchuli_tmp2 = yuchuli_tmp1.select([F.col(col).cast(T.IntegerType()) \
                           if 'back_method' in col \
                           or 'pickup_method' in col \
                           or 'transmission_type' in col \
                           or 'passenger_no' in col \
                           or 'door_no' in col \
                           else F.col(col) for col, dtype in yuchuli_tmp1.dtypes])
    yuchuli_tmp3 = yuchuli_tmp2.select([F.col(col).cast(T.DoubleType()) \
                               if 'site_score' in col \
                               else F.col(col) for col, dtype in yuchuli_tmp2.dtypes])
    
    yuchuli_tmp4 = yuchuli_tmp3.select([F.col(col).cast(T.DoubleType()) if 'decimal' in dtype else F.col(col) for col, dtype in yuchuli_tmp3.dtypes])


    fillna_dict = {feature_info.feature_name: feature_info.default_value for feature_info in get_features_info()}
    
    yuchuli_fin = yuchuli_tmp4.fillna(fillna_dict)
    
    
    # 落表
    tools.save_to_hive(yuchuli_fin, mode='overwrite', 
                       table_name='turing_dev.rent_cars_all_feature_old_new_model_train_data', partition_cols=['pt'])
    
    
    
if __name__ == '__main__': 
    
    from aibrain_common.utils import date_convert_utils

    from aibrain_common.component import tools
    date_converter = date_convert_utils.DateConvertUtils()
    # date_converter.set_biz_date("20240710")
    
    # date = date_converter.parse_data_date("${yyyymmdd}")
    date = '20241223'
    pt_ = date
    logging.info(f'remake train_data {date} ...')

    trainData(pt_) 