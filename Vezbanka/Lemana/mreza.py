# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from keras.layers import Dense 
from keras import Sequential
import keras
from sklearn.ensemble import RandomForestClassifier
import eli5
from eli5.sklearn import PermutationImportance
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

data = pd.read_excel(r'covid.xlsx', sheet_name="Pyton")

y = data['targets']
for i in range(len(y)):
    y.iloc[i]=y[i]-1
   
ytrain=y[0:int(len(y)*0.8)]
ytrain=np.array(ytrain)
ytest=y[len(ytrain):len(y)]    
ytest=np.array(ytest)   


x=data[['SPOL', 
        'LE WBC', 
        'Limf%', 
        'Mid%', 
        'Gran%', 
        'HGB', 
        'ER RBC', 
        'HCT',
       'MCV', 
       'TR PLT', 
       'SE', 
       'INR', 
       'APTT', 
       'D DIMER', 
       'GLUKOZA', 
       'UREA',
       'KREATININ', 
       'AC URICUM', 
       'AST', 
       'ALT', 
       'LDH', 
       'CK', 
       'KALIJ', 
       'NATRIJ',
       'UK PROTEINI', 
       'ALBUMIN', 
       'CRP', 
       'pH', 
       'pCO2', 
       'pO2', 
       'ctHb', 
       'sO2',
       'FO2Hb', 
       'FCOHb', 
       'FHHb', 
       'FMetHb', 
       'Hctc', 
       'ctO2c', 
       'p50c',
       'cBase(Ecf)c', 
       'cHCO3(P.st)c', 
       'cHCO3(P)c', 'ABEc']]

xtrain=y[0:int(len(x)*0.8)]
xtest=y[len(xtrain):len(x)]   

mreza=Sequential()
mreza.add(Dense(20,activation='tanh'))
mreza.add(Dense(50,activation='tanh'))
mreza.add(Dense(3,activation='softmax'))
opt = keras.optimizers.Adam(learning_rate=0.1)
mreza.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['accuracy'])

mreza.fit(xtrain,ytrain.reshape(-1,1),epochs=50,batch_size=100)
output=mreza.predict(xtest)
temp=[]
for i in range(len(output)):
    temp.append(np.argmax(output[i]))
    
counter=0
for i in range(len(ytest)):
    if(ytest[i]==temp[i]):
        counter+=1
        
print(counter/len(ytest))
##########################################################################




# perm = PermutationImportance(mreza, random_state=1).fit(x, y)
# eli5.show_weights(perm, feature_names=x.columns.to_list())
# input_layer = mreza.weights[0]
# importances = np.abs(input_layer).sum(axis=1)
# # importances = np.abs(mreza.layers[0].get_weights()[0]).sum(axis=1)
# print(importances)
# # Create a dataframe to store the results
# importances_df = pd.DataFrame({'Features': (x.columns), 'Importance': importances})
#
# # Sort the dataframe by importance score in descending order
# importances_df = importances_df.sort_values('Importance', ascending=False)

# Print the feature importances
# print(importances_df)

# rf=RandomForestClassifier(n_estimators=100)
# rf.fit(xtrain,ytrain.reshape(-1,1))
# pred=rf.predict(xtest)

# counter=0
# for i in range(len(ytest)):
#     if(ytest[i]==pred[i]):
#         counter+=1
        
# print(counter/len(ytest))

#from sklearn.preprocessing import StandardScaler
#from sklearn.neural_network import MLPRegressor
#scaler = StandardScaler()
#x_scaled = scaler.fit_transform(x)
#ann_model = MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', max_iter=1000)
#mreza.fit(x_scaled, y)
#input_layer = mreza.coefs_[0]
#importances = np.abs(input_layer).sum(axis=1)
#importances_df = pd.DataFrame({'Feature': x.columns, 'Importance': importances})
#importances_df = importances_df.sort_values('Importance', ascending=False)
#print(importances_df)