import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#load dataset
data = pd.read_csv('1_data_working.csv',encoding='utf-8',sep=',')

one = np.ones((data.shape[0],1))
data.insert(loc=0,column='A',value=one)
data_X = data[["A","BMI","Smoking","AlcoholDrinking","Stroke","PhysicalHealth","MentalHealth","DiffWalking","Sex","AgeCategory","Race","Diabetic","PhysicalActivity","GenHealth","SleepTime","Asthma","KidneyDisease","SkinCancer"]]
data_Y = data["HeartDisease"]
Y_train, Y_test, X_train, X_test = train_test_split(data_Y, data_X, test_size=0.2, random_state=50)

# khớp vào mẫu bằng cây phân lớp
dtc = DecisionTreeClassifier(max_depth=5)
dtc.fit(X_train, Y_train)
all_pred = dtc.predict(X_test)

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import v_measure_score
from sklearn.metrics import accuracy_score
print("Recall Score of Decision Tree Classifier    : ",round(recall_score(Y_test,all_pred), 4)*100,"%")
print("Precision Score of Decision Tree Classifier : ",round(precision_score(Y_test,all_pred), 4)*100,"%")
print("Measure Score of Decision Tree Classifier   : ",round(v_measure_score(Y_test,all_pred), 4)*100,"%")
print("Accuracy Score of Decision Tree Classifier  : ",round(accuracy_score(Y_test,all_pred), 4)*100,"%")

import matplotlib.pyplot as plt 
from sklearn import tree
# tree.plot_tree(dtc,max_depth=5,fontsize=10)
# plt.show() 
#ahihi