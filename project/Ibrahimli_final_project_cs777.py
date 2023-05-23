"""
Name: Eljan Ibrahimli
Class: CS 777 
Date: 10-12-2022
Final Project
Description: In am going to analyze Credit Approval data set as my final project. This dataset is available publicly, however, all attribute names and values 
have been changed to meaningless symbols to protect confidentiality of the data. This dataset is interesting because there is a good mix of
15 + class attributes — continuous, nominal with small numbers ofvalues, and nominal with larger numbers of values. There are also a few missing values.
In this dataset, I will manipulate the data, then preprocessing. After having clean data, I will split the data into training and testing data sets, scale 
them using Standard Scaler. My goal is to build a machine learning model that predicts whether a credit card has to be approved or not. 
I will implement several classifiers  to find out which classifier provides better accuracies in this dataset. 
    
 
"""

from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans
#!pip install xgboost==0.90

col_names = ['Gender', "Age", 'Debt', "Marital_status", "Bank_Customer", "Education", 'Ethnicity', "Year_of_Employment", "Prior_Default", "Employed",
                    "Credit_Score", "Drivers_License", "Citizen", "Zip_Code", "Income", "class"]
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data', names= col_names)

print(df.tail(20))
print(df.info())
print(df.describe())
col_names1 = ["Gender", "Age", 'Debt', "Marital_status", "Bank_Customer", "Education", 'Ethnicity', "Year_of_Employment", "Prior_Default", "Employed",
                    "Credit_Score", "Drivers_License", "Citizen", "Zip_Code", "Income"]

#Cleaning
df = df.replace('?', np.nan)
df.fillna(df.mean(), inplace=True)
print(df.isna().sum())

features_plot = ['Credit_Score', 'Income', 'Debt', 'Year_of_Employment']
for feat in features_plot:
    sns.distplot(df[feat])
    plt.show()
for col in df:
    if df[col].dtype == 'object':
        df = df.fillna(df[col].value_counts().index[0])
print(df.isna().sum())

le = LabelEncoder()
for col in df:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])
print(df.head())     
#Adding status column, approved = 0, denied  = 1
df['pred'] = np.where(df['class'] == 0, 'approved', 'denied')
approved_app = df[df['class'] == 0]
rejected_app = df[df['class'] == 1]
#Checking the data to see the distribution of the application status
y = np.array([len(approved_app), len(rejected_app)])
mylabels = ["Approved", "Denied"]
colors = ['#ff9999','#66b3ff']
plt.pie(y, labels = mylabels, colors = colors,  autopct='%1.1f%%',shadow=True, startangle=90)
plt.legend()
plt.show()

#Prior Default study
prior_def_class = df.groupby(['Prior_Default', 'class']).size()
x = ['Prior Default', 'No Prior Default']
y1 = prior_def_class[0]
y2 = prior_def_class[1]
# plot bars in stack manner
plt.bar(x, y1, color='r')
plt.bar(x, y2, bottom=y1, color='b')
plt.xlabel("Default Status")
plt.ylabel("Number of Applications")
plt.legend(["Approved", 'Denied'])
plt.show()
cMatrix = df.corr()
sns.heatmap(cMatrix, annot = False, cmap = 'BuPu')
plt.show()
#The above plot shows that Prior Default, Year of Employment, Debt, Income and Credit Score are highly correlated
#We can drop the rest features
cols = ['Marital_status','Employed','Prior_Default','Credit_Score', 'Income']
data = df[df.columns[df.columns.isin(['Marital_status','Employed','Prior_Default','Credit_Score', 'Income', 'pred'])]]
def con_mat(list1, list2):
    cnf_matrix = metrics.confusion_matrix(list1, list2)
    tp = cnf_matrix[0][0]
    tn = cnf_matrix[1][1]
    fp = cnf_matrix[1][0]
    fn = cnf_matrix[0][1]
    tpr = round(tp/(tp+fn),4)
    tnr = round(tn/(tn + fp),4)
    ac = round(accuracy_score(list1, list2),4)*100
    stat_list = [tp, fp,tn,fn,ac,tpr,tnr]
    return stat_list
headers = ["TP", "FP", "TN", "FN", "Accuracy", "TPR", "TNR"]
indeces = ['KNN best k value', 'SVM', 'Logistic Regression','Naive Bayesian','XGBoost' ,'Decision Tree' ,'Random Forest (N-d)*', 'Kmeans']
data_stat = pd.DataFrame(columns = headers, index = indeces )
def plot_feature_importance(importance,names,model_type):
    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)
    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.show()
#Lets implement XGBoost
df = df.drop('pred',1)
X = df.loc[:, ~df.columns.isin(['class'])]
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42, stratify=y, shuffle = True)
model = XGBClassifier(eval_metric='mlogloss')
model.fit(X_train, y_train)     
plot_feature_importance(model.feature_importances_, col_names1, 'XGBoost')
# USING KNN for k = 3,5,7,9,11
X = data.loc[:, ~data.columns.isin(['pred'])]
y = data['pred']
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30,
                             random_state = 42, stratify=y, shuffle = True)
#lets scale the data using Sdandard Scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
new_x = scaler.fit_transform(X_test)
k_list = [3,5,7,9,11]
ac_list =[]
stat_list = []
for index, k in enumerate(k_list):
# Training the K-NN model on the Training set
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(new_x)
    ac = round(accuracy_score(y_test, y_pred),4)*100
    ac_list.append(ac)
    stat_list.append(con_mat(y_test, y_pred))
data_stat.loc[indeces[0]] = stat_list[-1]
plt.figure()
plt.plot(k_list,ac_list,"r-")
plt.xlabel("k")
plt.ylabel("Accuracy of Predictions(%)")
plt.title("Accuracy vs k value for kNN")
plt.show()

#SVM
svm_classifier = svm.SVC(kernel='rbf', probability=True)
svm_classifier.fit(X_train,y_train)
y_pred = svm_classifier.predict(new_x)
data_stat.loc[indeces[1]] = con_mat(y_test, y_pred)
cf_matrix = confusion_matrix(y_test, y_pred)
#Heatmap - SVM model
ax = plt.subplot()
sns.heatmap(cf_matrix, annot=True, fmt='', cmap='BuPu')
ax.set_xlabel("Predicted Label")
ax.set_ylabel('True Label')
ax.set_title("Confusion Matrix - SVM model")
plt.show()

#Logistic Regression
X = data.loc[:, ~data.columns.isin(['pred'])]
y = data['pred']
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42, stratify=y, shuffle = True)
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
data_stat.loc[indeces[2]] = con_mat(y_test, y_pred)
cf_matrix = confusion_matrix(y_test, y_pred)
#Heatmap - Logistic Regression
ax = plt.subplot()
sns.heatmap(cf_matrix, annot=True, fmt='', cmap='BuPu')
ax.set_xlabel("Predicted Label")
ax.set_ylabel('True Label')
ax.set_title("Confusion Matrix - Logistic Regression")
plt.show()


#Naive Bayesian
NB_classifier = GaussianNB().fit(X_train,y_train)
y_pred = NB_classifier.predict(X_test)
data_stat.loc[indeces[3]] = con_mat(y_test, y_pred)
cf_matrix = confusion_matrix(y_test, y_pred)
#Heatmap - Naive Bayesian
ax = plt.subplot()
sns.heatmap(cf_matrix, annot=True, fmt='', cmap='BuPu')
ax.set_xlabel("Predicted Label")
ax.set_ylabel('True Label')
ax.set_title("Confusion Matrix - Naive Bayesian")
plt.show()

# Using XGBoost
model = XGBClassifier(objective="binary:logistic",eval_metric='rmse')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
data_stat.loc[indeces[4]] = con_mat(y_test, y_pred)
cf_matrix = confusion_matrix(y_test, y_pred)
#Heatmap that shows the confusion matrix for SVM model
ax = plt.subplot()
sns.heatmap(cf_matrix, annot=True, fmt='', cmap='BuPu')
ax.set_xlabel("Predicted Label")
ax.set_ylabel('True Label')
ax.set_title("Confusion Matrix - XGBoost")
plt.show()
plot_feature_importance(model.feature_importances_, cols, 'XGBoost')

#Decision Tree 
clf = tree.DecisionTreeClassifier(criterion = 'entropy', random_state=42)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
data_stat.loc[indeces[5]] = con_mat(y_test, y_pred)
cf_matrix = confusion_matrix(y_test, y_pred)
#Heatmap that shows the confusion matrix for SVM model
ax = plt.subplot()
sns.heatmap(cf_matrix, annot=True, fmt='', cmap='BuPu')
ax.set_xlabel("Predicted Label")
ax.set_ylabel('True Label')
ax.set_title("Confusion Matrix - Decision Tree Classifier")
plt.show()

#Random Forest Classification
N_list = [1,2,3,4,5,6,7,8,9,10]
d_list = [1,2,3,4,5]
ac_list = []
error_list = []
stat_list = []
for d in d_list:
    for n in N_list:
        model = RandomForestClassifier(n_estimators =n, max_depth =d,
                               criterion ='gini', random_state= 42)
        model.fit (X_train, y_train)
        y_pred = model.predict(X_test)
        ac = round(accuracy_score(y_test, y_pred),4)*100
        error_rate = np.mean(y_pred != y_test)*100
        error_list.append(error_rate)
        ac_list.append(ac)
        stat_list.append(con_mat(y_test, y_pred))
data1 = pd.DataFrame(np.array(error_list).reshape(5,10), columns=N_list)
print(data1)
for row in range(len(data1)):
    plt.plot(N_list, data1.iloc[row])
    plt.xlabel("Number of Estimators")
    plt.ylabel("Error Rate(%)")
    plt.title("Random Forest Error rate in different hyperparameters")
    plt.legend(d_list)
    plt.xticks(N_list)
    plt.ylim(15,25)
plt.show()
data_stat.loc[indeces[6]] = stat_list[21]

#Lets apply feature_importances on the best N-d Combination
model = RandomForestClassifier(n_estimators =2, max_depth =2, random_state=42)
model.fit(X_train, y_train)
plot_feature_importance(model.feature_importances_, cols, 'Random Forest')

#Using kmean clusters
#Creating an empty inertia list
inertia_list = []
#Loop through all k values(from 1 to 10)
scaler = StandardScaler()
X = scaler.fit_transform(X)
for k in range(1,11):
    #Calculating the inertia using predicted labels for all the features
    kmeans_classifier = KMeans(n_clusters=k, init='random', random_state=42)
    y_kmeans = kmeans_classifier.fit_predict(X)
    inertia = kmeans_classifier.inertia_
    inertia_list.append(inertia)
#Plotting the inertia for different k values
fig,ax = plt.subplots(1,figsize =(7,5))
plt.plot(range(1, 11), inertia_list, marker='o',
        color='grey')
plt.legend()
plt.xlabel('number of clusters: k')
plt.ylabel('inertia')
plt.tight_layout()
plt.show()

#The elbow shows that the best k value is 2
#Lets pick up 2 features 
df3 = data[['Credit_Score', 'Prior_Default']].copy()
#Choose the Prior Default and Employed as features for the clustering method.
X = df3.values
X2 = scaler.fit_transform(X)
#Predicting labels for my features using kmeans classifier with number of clusters = 2
kmeans_classifier = KMeans(n_clusters=2, random_state=42)
y_kmeans = kmeans_classifier.fit_predict(X2)
centroids = kmeans_classifier.cluster_centers_
#Plotting the 2 clusters with the assigned labels 
fig, ax = plt.subplots(1,figsize =(7,5))
plt.scatter(X2[y_kmeans == 0, 0], X2[y_kmeans == 0, 1],
                s = 75, c ='red', label = 'Approved')
plt.scatter(X2[y_kmeans == 1, 0], X2[y_kmeans == 1, 1],
                s = 75, c = 'blue', label = 'Denied')
plt.scatter(centroids[:, 0], centroids[:,1] ,
                s = 200 , c = 'black', label = 'Centroids')
plt.legend()
plt.xlabel('Credit_Score')
plt.ylabel('Prior_Default')
plt.tight_layout()
plt.show()
df3['true label'] = df['class']
df3['kmeans_label'] = y_kmeans
true_label = df3['true label'].to_numpy()
kmeans_label = df3['kmeans_label'].to_numpy()
data_stat.loc[indeces[7]] = con_mat(true_label, kmeans_label)
print(data_stat)