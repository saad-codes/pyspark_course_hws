from __future__ import print_function
import time
import re
import sys
import findspark
findspark.init()
import numpy as np
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
from pyspark.sql.functions import *
spark = SparkSession.builder.getOrCreate()
from pyspark.ml.feature import  IDF, Tokenizer, CountVectorizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import PCA
from pyspark.ml.classification import LogisticRegression
from pyspark.mllib.evaluation import MulticlassMetrics

from pyspark.ml.classification import LinearSVC

from pyspark.ml import Pipeline
def m_metrics_l(ml_model,test_data):
    predictions = ml_model.transform(test_data).cache()
    predictionAndLabels = predictions.select("label","prediction").rdd.map(lambda x: (float(x[0]), float(x[1]))).cache()
    

    metrics = MulticlassMetrics(predictionAndLabels)
    
    # Overall statistics
    precision = metrics.precision(1.0)
    recall = metrics.recall(1.0)
    f1Score = metrics.fMeasure(1.0)
    print(f"Precision = {precision:.4f} Recall = {recall:.4f} F1 Score = {f1Score:.4f}")
    print("Confusion matrix \n", metrics.confusionMatrix().toArray().astype(int))



if __name__ == "__main__":


    ##TRAIN
    # Use this code to reade the data
    corpus = spark.sparkContext.textFile('SmallTrainingData.txt')
    keyAndText = corpus.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6])).map(lambda x: (x[0], int(x[0].startswith("AU")),x[1]))   
    # Spark DataFrame to be used wiht MLlib 
    df = spark.createDataFrame(keyAndText).toDF("id","label","text")
    # print(df.select(col("label") , col("text")).show())    
    # Create a weight of each class
    p_weight = df.filter('label == 1').count()/ df.count()
    n_weight = df.filter('label == 0').count()/ df.count()
    df = df.withColumn("weight", when(col("label")==1,n_weight).otherwise(p_weight))
    # Preprocessing pipeline
    tokenizer = Tokenizer(inputCol="text", outputCol="words",)
    remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="filtered" )
    countVectorizer = CountVectorizer(inputCol=remover.getOutputCol(), outputCol="rawFeatures", vocabSize=5000)
    idf = IDF(inputCol=countVectorizer.getOutputCol(), outputCol="featuresIDF")
    pipeline_p = Pipeline(stages=[tokenizer,remover, countVectorizer, idf])
    start = time.time()

    data_model = pipeline_p.fit(df)
    transformed_data = data_model.transform(df)
    transformed_data.cache()

    print(f"Count Vector created in {time.time()-start:.2f}s.")
    print(transformed_data.show(3))
    # print(data_model.stages[2].vocabulary[:10])

    transformed_data.cache()

    #TEST
    #Use this code to read the data
    corpust = spark.sparkContext.textFile('SmallTestingData.txt')
    keyAndTextt = corpust.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6])).map(lambda x: (x[0], int(x[0].startswith("AU")),x[1]))   
    # Spark DataFrame to be used wiht MLlib 
    test = spark.createDataFrame(keyAndTextt).toDF("id","label","text")
    transformed_test = data_model.transform(test)
    # ### Task 2
    ### Build your learning model using Logistic Regression
    cassifier = LogisticRegression(maxIter=20, featuresCol = "featuresIDF", weightCol="weight")
    pipeline = Pipeline(stages=[cassifier])
    start = time.time()
    model = pipeline.fit(transformed_data)
    print(f"Model created in {time.time()-start:.2f}s.")
    m_metrics_l(model,transformed_test)
    print(f"Total time {time.time()-start:.2f}s.")


 

    ### Task 3
    ### Build your learning model using SVM
    cassifier = LinearSVC(maxIter=20, featuresCol = "featuresIDF", weightCol="weight")
    pipeline = Pipeline(stages=[cassifier])
    start = time.time()
    print(f"Training started.")
    model = pipeline.fit(transformed_data)
    print(f"Model created in {time.time()-start:.2f}s.")
    m_metrics_l(model,transformed_test)
    print(f"Total time {time.time()-start:.2f}s.")





    # # ### Task 4
    # ### Rebuild your learning models using 200 words instead of 5000
    # Preprocessing pipeline
    tokenizer = Tokenizer(inputCol="text", outputCol="words",)
    remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="filtered" )
    countVectorizer = CountVectorizer(inputCol=remover.getOutputCol(), outputCol="rawFeatures", vocabSize=5000)
    pca = PCA(k=200, inputCol=countVectorizer.getOutputCol(), outputCol="pcaFeatures")
    idf = IDF(inputCol=pca.getOutputCol(), outputCol="featuresIDF")
    pca_feature = Pipeline(stages=[tokenizer,remover, countVectorizer,pca, idf])
    start = time.time()

    data_model = pca_feature.fit(df)
    transformed_data_pca= data_model.transform(df)
    transformed_data_pca.cache()
    print(f"Count Vector created in {time.time()-start:.2f}s.")
    print(transformed_data_pca.show(3))
    transformed_test = data_model.transform(test)
    cassifier = LogisticRegression(maxIter=20, featuresCol = "featuresIDF", weightCol="weight")
    pipeline = Pipeline(stages=[cassifier])
    start = time.time()
    model = pipeline.fit(transformed_data_pca)
    print(f"Model created in {time.time()-start:.2f}s.")
    m_metrics_l(model,transformed_test)
    print(f"Total time {time.time()-start:.2f}s.")







    cassifier = LinearSVC(maxIter=20, featuresCol = "featuresIDF", weightCol="weight")
    pipeline = Pipeline(stages=[cassifier])
    start = time.time()
    print(f"Training started.")
    model = pipeline.fit(transformed_data_pca)
    print(f"Model created in {time.time()-start:.2f}s.")
    m_metrics_l(model,transformed_test)
    print(f"Total time {time.time()-start:.2f}s.")



    spark.sparkContext.stop()

    
