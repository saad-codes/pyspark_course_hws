from __future__ import print_function
from calendar import c

import os
from re import X
import sys
import numpy as np
from operator import add

from pyspark import SparkConf,SparkContext
from pyspark.streaming import StreamingContext
import findspark 
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext

from pyspark.sql.types import *
from pyspark.sql import functions as func
from pyspark.sql.functions import *


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
            if(float(p[4])> 60 and float(p[5])>0 and float(p[11])> 0 and float(p[16])> 1 and float(p[16])< 600):
                return p

#Main
if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()
    sc = spark.sparkContext
    rdd = spark.read.csv(r"tested.csv").rdd
    rdd = rdd.filter(lambda x: correctRows(x))
    df = rdd.toDF()
    num_iteration = 1
    n = rdd.count()
    learningRate_mu = 0.0000001
    cost = [0]
    print("########### TASK 3 ############")
    task3_rdd = df.select(array(col('_c4').cast("float"),col('_c5').cast("float"),col('_c11').cast("float"),col('_c12').cast("float")),col('_c16').cast("float")).rdd
    task3_rdd = task3_rdd.map(lambda x: ((x[1]), np.array(x[0])))
    task3_rdd = task3_rdd.map(lambda x: (x[0] , np.append(x[1],1)))

    task3_rdd.cache()
    weights = np.zeros(5)
    for i in range(num_iteration):
        gradient_costs = task3_rdd.map(lambda x: (x[1] , x[0]-np.dot(x[1] ,weights ))).map(lambda x: (x[0]*x[1] ,x[1]**2)).reduce(lambda x,y: (x[0]+y[0] , x[1]+y[1]))
        
        cost.append(gradient_costs[1])
        
        weights += (2/n)*(gradient_costs[0])  
        print( "slope/weight ", weights[:-1], " Intercept  ", weights[-1] , ' cost ', cost[i+1] )

        if cost[i+1]<cost[i]:
            learningRate_mu = learningRate_mu*1.15 # increasing 5%
        else:
            learningRate_mu = learningRate_mu/2 # decreasing 50%

    task3_rdd.unpersist()
    
    print( "slope/weight ", weights[:-1], " Intercept ", weights[-1])
    sc.stop()

    