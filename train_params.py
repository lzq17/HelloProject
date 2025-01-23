import logging
from aibrain_common.utils.date_convert_utils import DateConvertUtils
from aibrain_common.utils import date_convert_utils
from aibrain_job.utils import param_utils

from features_info import get_feature_columns
from spark_session_utils import SparkSessionHelper


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    
    date_converter = date_convert_utils.DateConvertUtils()
    train_pt = date_converter.parse_data_date("${yyyymmdd-3}")
    eval_pt = date_converter.parse_data_date("${yyyymmdd-2}")
    train_partition = [['pt=' + str(train_pt)]] # [['pt=20240709']]
    eval_partition = [['pt=' + str(eval_pt)]] # [['pt=20240710']]
    
    print("训练数据pt:",train_partition)
    print("测试数据pt:",eval_partition)
    
    estimator_params = {
        'hidden_dims': [128, 64, 32],
        'feature_columns': get_feature_columns([train_pt, eval_pt]), # '20240709','20240710'
        'alpha': 0.5
    }
    
    param_utils.put_object_to_output_param('estimator_params', estimator_params)
    param_utils.put_object_to_output_param('train_partition', train_partition)
    param_utils.put_object_to_output_param('eval_partition', eval_partition)
    param_utils.put_object_to_output_param('max_steps', 100000) # 50000
    param_utils.put_object_to_output_param('batch_size', 32)
    

    logger.warning('train params success')

    return


if __name__ == '__main__':
    main()
