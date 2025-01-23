# 这里存放哈啰租车业务的精排项目

- config.py &nbsp; &nbsp; &nbsp; hive表
- gen_train_data.py &nbsp; &nbsp; &nbsp; 生成训练数据（从all_feature.py中删除部分字段&null值填充）
- features_info.py &nbsp; &nbsp; &nbsp; 特征工程（数据预处理）
- ple.py &nbsp; &nbsp; &nbsp; PLE模型
- train_params.py &nbsp; &nbsp; &nbsp; 训练参数
- metric.py &nbsp; &nbsp; &nbsp; 效果指标
- spark_session_utils.py &nbsp; &nbsp; &nbsp; spark设置
- train.py 模型训练