import findspark
findspark.init()
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
sc = SparkContext.getOrCreate()
spark = SparkSession.builder.getOrCreate()


#################################################

wikiCategoryFile="wiki-categorylinks-small.csv"

df = spark.read.csv(wikiCategoryFile)
################################################

############ TASK 3 ############################

#### PART A
print("PART A")
grouped_dataA = df.groupBy(col('_c0')).count()
print(grouped_dataA.agg({"count":"avg" }).show())
print(grouped_dataA.agg({"count":"max" }).show())
print(grouped_dataA.agg({"count":"std" }).show())
print('median is ' ,grouped_dataA.approxQuantile('count',[0.5],0.1))

#### PART B
print("PART B")
grouped_dataB = df.groupBy(col('_c1')).count()
grouped_dataB = grouped_dataB.orderBy(col('count').desc())
print(grouped_dataB.withColumnRenamed('_c1','category').show(10))
