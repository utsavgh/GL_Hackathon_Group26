import os
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, functions as F, DataFrame
from pyspark.ml.feature import StringIndexer, VectorAssembler, Imputer, VectorIndexer, Bucketizer, OneHotEncoder
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, GeneralizedLinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from flask import Flask, render_template, request, redirect, make_response

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html', flag='display: none;')

@app.route("/predict",  methods = ['GET', 'POST'])
def main():
    spark = SparkSession.builder.getOrCreate()
    print(f'Spark Version : {spark.version}')
    print(f'Spark Context : {spark.sparkContext.appName}, {spark.sparkContext.master}')
    print(f'Spark URL : {spark.sparkContext.uiWebUrl}')
    for dirname, _, filenames in os.walk('static'):
        for filename in filenames:
            print(os.path.join(dirname, filename))
    PATH='static'
    sparkdf_train = spark.read.csv(f'{PATH}/train.csv', inferSchema=True, header=True)
    sparkdf_test = spark.read.csv(f'{PATH}/test.csv', inferSchema=True, header=True)
    sparkdf_sample_submission = spark.read.csv(f'{PATH}/sample_submission.csv', inferSchema=True, header=True)
    col_sample_submission = ['Id', 'SalePrice']
    pdf = sparkdf_train.toPandas()
    print(sparkdf_train.printSchema())
    # Feature Selection
    str_features = []
    int_features = []
    for col in sparkdf_train.dtypes:
        if col[1] == 'string':
            str_features += [col[0]]
        else:
            int_features += [col[0]]
    print(f'str_features : {str_features}')
    print(f'int_features: {int_features}')
    sparkdf_train_filter = sparkdf_train.select(int_features + str_features)
    int_features.remove('SalePrice')
    sparkdf_test_filter = sparkdf_test.select(int_features + str_features)
    print(sparkdf_train.select(str_features).limit(15).toPandas())
    def cast_to_int(_sparkdf: DataFrame, col_list: list) -> DataFrame:
        for col in col_list:
            _sparkdf = _sparkdf.withColumn(col, _sparkdf[col].cast('int'))
        return _sparkdf
    sparkdf_test_typecast = cast_to_int(sparkdf_test_filter, int_features)
    stage_list = []
    str_indexer = [StringIndexer(inputCol=column, outputCol=f'{column}_StringIndexer', handleInvalid='keep') for column
                   in str_features]
    stage_list += str_indexer
    assembler_input = [f for f in int_features]
    assembler_input += [f'{column}_StringIndexer' for column in str_features]
    feature_vector = VectorAssembler(inputCols=assembler_input, outputCol='features', handleInvalid='keep')
    stage_list += [feature_vector]
    vect_indexer = VectorIndexer(inputCol='features', outputCol='features_indexed', handleInvalid='keep')
    stage_list += [vect_indexer]
    LR = LinearRegression(featuresCol='features_indexed', labelCol='SalePrice', maxIter=10, regParam=0.3,
                          elasticNetParam=0.8)
    stage_list += [LR]
    # Start predicting
    ml_pipeline = Pipeline(stages=stage_list)
    model = ml_pipeline.fit(sparkdf_train_filter)
    sparkdf_predict = model.transform(sparkdf_test_typecast)
    print(sparkdf_predict.limit(5).toPandas().T)
    trainingSummary = model.stages[-1].summary
    print("-" * 100)
    print("Linear Regression")
    print("-" * 100)
    print(f'Root Mean Squared Error (RMSE) : {trainingSummary.rootMeanSquaredError}')
    print(f'R2: {trainingSummary.r2}')
    print(f'Mean Absolute Error (MAE) : {trainingSummary.meanAbsoluteError}')
    print(f'Mean Squared Error (MSE) : {trainingSummary.meanSquaredError}')
    print("-" * 100)
    sparkdf_predict = sparkdf_predict.withColumnRenamed('prediction', 'SalePrice').select('Id', 'SalePrice')
    sparkdf_predict.show()
    sparkdf_predict = sparkdf_predict.toPandas()
    sparkdf_predict.to_csv('static/gl_hackathon_submission.csv', index=False)
    return render_template('index.html', flag='display: block;', sversion=spark.version, scontext=spark.sparkContext.appName+','+spark.sparkContext.master, surl=spark.sparkContext.uiWebUrl, rmse=trainingSummary.rootMeanSquaredError, r2=trainingSummary.r2, mae=trainingSummary.meanAbsoluteError, mse=trainingSummary.meanSquaredError )

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=8000)