# from __future__ import print_function

# import os
# import sys
# from typing import OrderedDict
# # import requests
import findspark
findspark.init()

# #!pip install pyspark

# from pyspark import SparkConf,SparkContext
# from pyspark.streaming import StreamingContext

# from pyspark.sql import SparkSession
# from pyspark.sql import SQLContext

# from pyspark.sql.types import *
# from pyspark.sql import functions as func
from pyspark.sql.functions import *
# import spark
# from pyspark.sql import Row

from pyspark.sql import SparkSession


spark = SparkSession.builder.master('local[*]').config("spark.driver.memory", "3g").getOrCreate()    
spark.sparkContext.setLogLevel("OFF")
#Exception Handling and removing wrong datalines
def isfloat(value):
    try:
        float(value)
        return True
 
    except:
         return False

#Function - Cleaning
#For example, remove lines if they don’t have 16 values and 
# checking if the trip distance and fare amount is a float number
# checking if the trip duration is more than a minute, trip distance is more than 0.1 miles, 
# fare amount and total amount are more than 0.1 dollars
def correctRows(p):
    if(len(p)==17):
        if(isfloat(p[5]) and isfloat(p[11])):
            if(float(p[4])> 60 and float(p[5])>0 and float(p[11])> 0 and float(p[16])> 0):
                return p


#Main
if __name__ == "__main__":

    from pyspark.sql import SparkSession, Row

    spark = SparkSession.builder.getOrCreate() 
    from operator import countOf
    rdd = spark.read.csv("tested.csv").rdd
    print(rdd.count())
    rdd = rdd.filter(lambda x: correctRows(x))
    print('RDD Count:', rdd.count())
    task1 = rdd.toDF()
    task1 = task1.select(col("_c0"),col("_c1")).groupBy("_c0").count()
    task1 = task1.sort(task1['count'], ascending=False).collect()
    task1 = spark.sparkContext.parallelize(task1).top(10,lambda x: x[1])
    print("task1")
    # print(spark.sparkContext.parallelize(task1).top(10, key = lambda x: x[1] ))
    
    print(task1)

    

    
    
    
    
    
    
    
    

    
    