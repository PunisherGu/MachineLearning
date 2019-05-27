# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 18:57:21 2018

@author: Gustavo
"""
import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt


series = pd.read_csv('Petr.csv')
fechamento = series['Close'].copy()

series = series.drop('Close',axis = 1)
series = series.drop('Date',axis = 1)

X_treino, X_teste, y_treino, y_teste = train_test_split(series, fechamento, random_state=42)

# walk-forward validation
history = [x for x in X_treino]
predictions = list()
for i in range(len(X_teste)):
	# make prediction
	predictions.append(history[-1])
	# observation
	history.append(X_treino[i])
# report performance
rmse = sqrt(mean_squared_error(X_treino, predictions))
print('RMSE: %.3f' % rmse)
# line plot of observed vs predicted
pyplot.plot(X_teste)
pyplot.plot(predictions)
pyplot.show()   
