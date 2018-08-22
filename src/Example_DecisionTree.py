# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 22:55:10 2018

@author: gilsonsf
"""

import pandas as pd
import os
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics

#muda o diretorio corrente
os.chdir("/home/gilsonsf/Documents/Mestrado/ML/out")

#Carrega dataset
AH_data = pd.read_csv("tree_addhealth.csv")

#drop todos NaN
data_clean = AH_data.dropna()

data_clean.dtypes
data_clean.describe()


"""
Modeling and Prediction
"""
#Split into training and testing sets

predictors = data_clean[['BIO_SEX','HISPANIC','WHITE','BLACK','NAMERICAN','ASIAN',
'age','ALCEVR1','ALCPROBS1','marever1','cocever1','inhever1','cigavail','DEP1',
'ESTEEM1','VIOL1','PASSIST','DEVIANT1','SCHCONN1','GPA1','EXPEL1','FAMCONCT','PARACTV',
'PARPRES']]

#predictors = data_clean[['marever1','ALCEVR1']]

targets = data_clean.TREG1

X_train, X_test, Y_train, Y_test  =   train_test_split(predictors, targets, test_size=.4)

X_train.shape
X_test.shape
Y_train.shape
Y_test.shape

#Construindo modelo e treinando
classifier=DecisionTreeClassifier()
classifier=classifier.fit(X_train,Y_train)

#Obtendo predicoes
predictions=classifier.predict(X_test)

sklearn.metrics.confusion_matrix(Y_test,predictions)
sklearn.metrics.accuracy_score(Y_test, predictions)

#Displaying 
from sklearn import tree
from io import StringIO
import graphviz

out = StringIO()
dot_data = tree.export_graphviz(classifier, out_file=None)
graph = graphviz.Source(dot_data) 
graph.render("decision_tree") 



