from __future__ import print_function

import re
import numpy as np
from operator import add
import findspark
findspark.init()
from pyspark import SparkContext
numTopWords = 10000
def freqArray (listOfIndices):
	global numTopWords
	returnVal = np.zeros (numTopWords)
	for index in listOfIndices:
		returnVal[index] = returnVal[index] + 1
	mysum = np.sum (returnVal)
	returnVal = np.divide(returnVal, mysum)
	return returnVal


def buildArray(listOfIndices):

    

    returnVal = np.zeros(numTopWords)

    

    for index in listOfIndices:

        returnVal[index] = returnVal[index] + 1

    

    mysum = np.sum(returnVal)

    

    returnVal = np.divide(returnVal, mysum)

    

    return returnVal

if __name__ == "__main__":

	sc = SparkContext(appName="Assignment-4")

	## Task 1
	## Data Preparation
	train_corpus = sc.textFile("SmallTrainingData.txt")
	
	numberOfDocs = train_corpus.count()
	keyAndText = train_corpus.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))
	regex = re.compile('[^a-zA-Z]')
	keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))
	newRdd = keyAndListOfWords.flatMap(lambda x:  [(word , 1) for word in x[1][:]])
	allCounts = newRdd.reduceByKey(lambda x,y:x+y)
	topWords = allCounts.top(10000, key=lambda x : x[1])
	topWordsK = sc.parallelize(range(numTopWords))
	dictionary = topWordsK.map (lambda x : (topWords[x][0], x))

	# print("Index for 'applicant' is",dictionary.filter(lambda x: x[0]=='applicant').take(1)[0][1])
	# print("Index for 'and' is",dictionary.filter(lambda x: x[0]=='and').take(1)[0][1])
	# print("Index for 'attack' is",dictionary.filter(lambda x: x[0]=='attack').take(1)[0][1])
	# print("Index for 'protein' is",dictionary.filter(lambda x: x[0]=='protein').take(1)[0][1])
	# print("Index for 'car' is",dictionary.filter(lambda x: x[0]=='car').take(1)[0][1])
	# print("Index for 'in' is",dictionary.filter(lambda x: x[0]=='in').take(1)[0][1])




	# ### Task 2
	# ### Build your learning model
	allWordsWithDocID = keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))
	allDictionaryWords =  dictionary.join(allWordsWithDocID)
	justDocAndPos = allDictionaryWords.map(lambda x: (x[1][1],x[1][0]))
	allDictionaryWordsInEachDoc = justDocAndPos.groupByKey()
	# # TF 
	TF_train = allDictionaryWordsInEachDoc.map(lambda x: (1 if x[0].startswith('AU') else 0, np.append(buildArray(x[1]),1)))
	n = TF_train.count()
	n0 = TF_train.filter(lambda x: x[0]==0).count()
	n1 = TF_train.filter(lambda x: x[0]==1).count()
	w0 = n0/(2*n)
	w1 = 20*n1/(n)
	print(w0,w1)
	iters = 5				
	lr = 0.0001	
	regval = 0.001
	batchSize = 512
	tolerance = 10e-8
	TF_train.cache()
	n = TF_train.count()

	def LogisticRegression_weighted(traindata=TF_train,
						max_iteration = 15,
						learningRate = 0.01,
						regularization = 0.01,
						mini_batch_size = 64,
						tolerance = 10e-8,
						optimizer = 'SGD',  
						train_size=1
						):

		# initialization
		prev_cost = 0
		L_cost = []
		prev_validation = 0
		
		parameter_size = len(traindata.take(1)[0][1])
		np.random.seed(0)
		parameter_vector = np.random.normal(0, 0.1, parameter_size)
		momentum = np.zeros(parameter_size)
		prev_mom = np.zeros(parameter_size)
		second_mom = np.array(parameter_size)
		gti = np.zeros(parameter_size)
		epsilon = 10e-8
		
		for i in range(max_iteration):

			bc_weights = parameter_vector
			min_batch = traindata.sample(False, mini_batch_size / train_size, 1 + i)
			res1 = min_batch.filter(lambda x: x[0]==1).treeAggregate(
				(np.zeros(parameter_size), 0, 0),
				lambda x, y:(x[0]+\
							(y[1])*(-y[0]+(1/(np.exp(-np.dot(y[1], bc_weights))+1))),\
							x[1]+\
							y[0]*(-(np.dot(y[1], bc_weights)))+np.log(1 + np.exp(np.dot(y[1],bc_weights))),\
							x[2] + 1),
				lambda x, y:(x[0] + y[0], x[1] + y[1], x[2] + y[2])
				)        
			# Calcualtion of negative class. Only the samples labeled as 0 are filtered and then processed
			res0 = min_batch.filter(lambda x: x[0]==0).treeAggregate(
				(np.zeros(parameter_size), 0, 0),
				lambda x, y:(x[0]+\
							(y[1])*(-y[0]+(1/(np.exp(-np.dot(y[1], bc_weights))+1))),\
							x[1]+\
							y[0]*(-(np.dot(y[1], bc_weights)))+np.log(1 + np.exp(np.dot(y[1],bc_weights))),\
							x[2] + 1),
				lambda x, y:(x[0] + y[0], x[1] + y[1], x[2] + y[2])
				)        
			
			# The total gradients are a weighted sum
			gradients = w0*res0[0]+w1*res1[0]
			sum_cost = w0*res0[1]+w1*res1[1]
			num_samples = res0[2]+res1[2]
			
			cost =  sum_cost/num_samples + regularization * (np.square(parameter_vector).sum())

			# calculate gradients
			gradient_derivative = (1.0 / num_samples) * gradients + 2 * regularization * parameter_vector
			
			if optimizer == 'SGD':
				parameter_vector = parameter_vector - learningRate * gradient_derivative

				
				
			print("Iteration No.", i, " Cost=", cost, "graients=",gradients)
			
			# Stop if the cost is not descreasing
			if abs(cost - prev_cost) < tolerance:
				print("cost - prev_cost: " + str(cost - prev_cost))
				break
			prev_cost = cost
			L_cost.append(cost)
			
		return parameter_vector, L_cost
	# def LogisticRegression(dataset):
	# 	prev_cost = 0
	# 	L_cost = []		
	# 	parameter_size = len(dataset.take(1)[0][1])
	# 	np.random.seed(0)
	# 	parameter_vector = np.random.normal(0, 0.1, parameter_size)		
	# 	for i in range(iters):

	# 		bc_weights = parameter_vector
	# 		min_batch = dataset.sample(False, batchSize / n, 1 + i)

	# 		res = min_batch.treeAggregate((np.zeros(parameter_size), 0, 0),  lambda x, y:(x[0] + (y[1]) * (-y[0] + (1/(np.exp(-np.dot(y[1], bc_weights))+1))),x[1] + y[0] * (-(np.dot(y[1], bc_weights))) \
	# 						+ np.log(1 + np.exp(np.dot(y[1],bc_weights))),\
	# 						x[2] + 1),lambda x, y:(x[0] + y[0], x[1] + y[1], x[2] + y[2]))        

	# 		# Optimise
	# 		gradients = res[0]
	# 		sum_cost = res[1]
	# 		num_samples = res[2]
	# 		cost =  sum_cost/num_samples + regval * (np.square(parameter_vector).sum())
	# 		# calculate gradients
	# 		gradient_derivative = (1.0 / num_samples) * gradients + 2 * regval * parameter_vector
	# 		parameter_vector = parameter_vector - lr * gradient_derivative


				
				
	# 		print("Iteration No.", i, " Cost=", cost, "graients=",gradients)
			
	# 		# Stop if the cost is not descreasing
	# 		if abs(cost - prev_cost) < tolerance:
	# 			print("cost - prev_cost: " + str(cost - prev_cost))
	# 			break
	# 		prev_cost = cost
	# 		L_cost.append(cost)
			
	# 	return parameter_vector, L_cost
	# ### Print the top 5 words with the highest coefficients
	parameter_vector_sgd_w, L_cost_sgd_w = LogisticRegression_weighted()
	listed = dictionary.collect()
	listCoefficient = sc.parallelize([ (listed[i][0],parameter_vector_sgd_w[i])  for  i in range(len(listed)) ]).top(5,key = lambda x: x[1] )
	print("top 5 words coefficeinets ", listCoefficient)
	# ### Task 3
	# ### Use your model to predict the category of each document

	test_corpus = sc.textFile("SmallTestingData.txt")
	
	numberOfTestDocs = test_corpus.count()
	keyAndTextTest = test_corpus.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))
	regex = re.compile('[^a-zA-Z]')
	keyAndListOfWordsTest = keyAndTextTest.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))
	allWordsWithDocIDTest = keyAndListOfWordsTest.flatMap(lambda x: ((j, x[0]) for j in x[1]))
	allDictionaryWordsTest =  dictionary.join(allWordsWithDocIDTest)
	justDocAndPosTest = allDictionaryWordsTest.map(lambda x: (x[1][1],x[1][0]))
	allDictionaryWordsInEachDocTest = justDocAndPosTest.groupByKey()
	TF_test = allDictionaryWordsInEachDocTest.map(lambda x: (1 if x[0].startswith('AU') else 0, np.append(buildArray(x[1]),1)))
	test_num = TF_test.count()
	predictions = TF_test.map(lambda x: (x[0], 1 if np.dot(x[1],parameter_vector_sgd_w)>0 else 0))
	true_positive = predictions.map(lambda x: 1 if (x[0]== 1) and (x[1]==1) else 0).reduce(lambda x,y:x+y)
	false_positive = predictions.map(lambda x: 1 if (x[0]== 0) and (x[1]==1) else 0).reduce(lambda x,y:x+y)
	true_negative = predictions.map(lambda x: 1 if (x[0]== 0) and (x[1]==0) else 0).reduce(lambda x,y:x+y)
	false_negative = predictions.map(lambda x: 1 if (x[0]== 1) and (x[1]==0) else 0).reduce(lambda x,y:x+y)

	# Print the Contingency matrix
	print("--Contingency matrix--")
	print(f" TP:{true_positive:6}  FP:{false_positive:6}")
	print(f" FN:{false_negative:6}  TN:{true_negative:6}")
	print("----------------------")

	# Calculate the Accuracy and the F1
	accuracy = (true_positive+true_negative)/(test_num)
	f1 = true_positive/(true_positive+0.5*(false_positive+false_negative))
	print(f"Accuracy = {accuracy}  \nF1 = {f1}")



	sc.stop()
