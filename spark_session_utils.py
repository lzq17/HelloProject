from pyspark.sql.session import SparkSession
from pyspark import SparkContext
import logging

class SparkSessionHelper(object):
    def __init__(self, init_flag=True):
        if init_flag:
            sc = SparkContext._active_spark_context
            if sc is not None:
                self.spark = SparkSession.builder.getOrCreate()
                return
            self.spark = SparkSession.builder \
                .config('spark.default.parallelism', '800') \
                .config('spark.driver.maxResultSize', '200g') \
                .config('spark.driver.memory', '200g') \
                .config('spark.dynamicAllocation.maxExecutors', '100') \
                .config('spark.dynamicAllocation.minExecutors', '5') \
                .config('spark.executor.cores', '6') \
                .config('spark.executor.memory', '14g') \
                .config('spark.hadoop.hive.exec.dynamic.partition.mode', 'nonstrict') \
                .config('spark.hadoop.hive.exec.dynamic.partition', 'true') \
                .config('spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version', '2') \
                .config('spark.sql.execution.arrow.enabled', 'false') \
                .config('spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation', 'true') \
                .config('spark.sql.shuffle.partitions', '2000') \
                .config('spark.sql.sources.default', 'orc') \
                .config('spark.sql.sources.partitionOverwriteMode', 'dynamic') \
                .appName('lizhiqiang195') \
                .enableHiveSupport().getOrCreate()
        else:
            self.spark = None

    def __del__(self):
        pass
        #self.spark.stop()

    def get(self):
        return self.spark
