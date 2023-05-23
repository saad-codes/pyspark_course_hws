from pyspark.sql import SparkSession
import findspark
findspark.init()
import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import GBTClassifier
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import MulticlassMetrics


spark = SparkSession.builder.appName("project").getOrCreate()



def m_metrics_l(ml_model,test_data):
    predictions = ml_model.transform(test_data).cache()
    predictionAndLabels = predictions.select("label","prediction").rdd.map(lambda x: (float(x[0]), float(x[1]))).cache()
    
    # Print some predictions vs labels
    # print(predictionAndLabels.take(10))
    metrics = MulticlassMetrics(predictionAndLabels)
    
    # Overall statistics
    precision = metrics.precision(1.0)
    recall = metrics.recall(1.0)
    f1Score = metrics.fMeasure(1.0)
    print(f"Precision = {precision:.4f} Recall = {recall:.4f} F1 Score = {f1Score:.4f}")
    print("Confusion matrix \n", metrics.confusionMatrix().toArray().astype(int))


col_names = ['Gender', "Age", 'Debt', "Marital_status", "Bank_Customer", "Education", 'Ethnicity', "Year_of_Employment", "Prior_Default", "Employed",
                    "Credit_Score", "Drivers_License", "Citizen", "Zip_Code", "Income", "class"]
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data', names= col_names)


df = spark.createDataFrame(df)
print(df.show())

from pyspark.sql.functions import col
df = df.filter(col("Age") != "?")

df = df.na.drop()

indexer = StringIndexer(inputCol="Gender", outputCol="GenderIndex")
indexed = indexer.fit(df).transform(df)
indexer = StringIndexer(inputCol="Marital_status", outputCol="MaritalStatusIndex")
indexed = indexer.fit(indexed).transform(indexed)
indexer = StringIndexer(inputCol="Bank_Customer", outputCol="BankCustomerIndex")
indexed = indexer.fit(indexed).transform(indexed)
indexer = StringIndexer(inputCol="Education", outputCol="EducationIndex")
indexed = indexer.fit(indexed).transform(indexed)
indexer = StringIndexer(inputCol="Ethnicity", outputCol="EthnicityIndex")
indexed = indexer.fit(indexed).transform(indexed)
indexer = StringIndexer(inputCol="Prior_Default", outputCol="PriorDefaultIndex")
indexed = indexer.fit(indexed).transform(indexed)
indexer = StringIndexer(inputCol="Employed", outputCol="EmployedIndex")
indexed = indexer.fit(indexed).transform(indexed)
indexer = StringIndexer(inputCol="Drivers_License", outputCol="DriversLicenseIndex")
indexed = indexer.fit(indexed).transform(indexed)
indexer = StringIndexer(inputCol="Citizen", outputCol="CitizenIndex")
indexed = indexer.fit(indexed).transform(indexed)
indexer = StringIndexer(inputCol="class", outputCol="label")
indexed = indexer.fit(indexed).transform(indexed)

indexed = indexed.select("GenderIndex",col("Age").cast('float') , "Credit_Score", 'Debt', "MaritalStatusIndex", "Income" , "BankCustomerIndex" , "EducationIndex" ,"EthnicityIndex" ,"PriorDefaultIndex" ,"EmployedIndex","DriversLicenseIndex","CitizenIndex","label" )

featureCols = ["GenderIndex" ,"Age" , "Credit_Score", 'Debt', "MaritalStatusIndex", "Income" , "BankCustomerIndex" , "EducationIndex" ,"EthnicityIndex" ,"PriorDefaultIndex" ,"EmployedIndex","DriversLicenseIndex","CitizenIndex"] 

# put features into a feature vector column
assembler = VectorAssembler(inputCols=featureCols, outputCol="features") 

assembled_df = assembler.transform(indexed)
# print(assembled_df.show(10))

# Split the data into train and test sets
train_data, test_data = assembled_df.randomSplit([.8,.2], seed=2)
train_data.cache()
test_data.cache()

# Logistics regression

print("### Logistic Regression ###")
cassifier = LogisticRegression(maxIter=10, featuresCol = "features")

pipeline = Pipeline(stages=[cassifier])
print(f"Training started.")
model = pipeline.fit(train_data)
print(m_metrics_l(model,test_data))

print("### GBTClassifier  ###")
cassifier = GBTClassifier(maxIter=10, featuresCol = "features", maxDepth=10)
pipeline = Pipeline(stages=[cassifier])
print(f"Training started.")
model = pipeline.fit(train_data)
print(m_metrics_l(model,test_data))

print("### LinearSVC  ###")
cassifier = LinearSVC(maxIter=10, featuresCol = "features")
pipeline = Pipeline(stages=[cassifier])
print(f"Training started.")
model = pipeline.fit(train_data)

print(m_metrics_l(model,test_data))
