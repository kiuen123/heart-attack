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
dtc = DecisionTreeClassifier(max_depth=10)
dtc.fit(X_train, Y_train)
all_pred = dtc.predict(X_test)

def acc(Y_test, all_pred):
    correct = np.sum(Y_test == all_pred)
    return float(correct)/Y_test.shape[0]
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(Y_test, all_pred)
print('Confusion matrix:')
print(cnf_matrix)
TP=cnf_matrix[0][0]
FN=cnf_matrix[0][1]
TN=cnf_matrix[1][1]
FP=cnf_matrix[1][0]


Recall=(TP/(TP+FN))
Precision=(TP/(TP+FP))
F1_Measure=((2*Recall*Precision)/(Recall+Precision))
print("Recall Score of Decision Tree Classifier  : ",round(Recall, 4)*100,"%")
print("Precision Score of Decision Tree Classifier  : ",round(Precision, 3)*100,"%")
print("F1_Measure Score of Decision Tree Classifier : ",round(F1_Measure, 4)*100,"%")
print("Accuracy Score of Decision Tree Classifier  : ",round(acc(Y_test,all_pred), 4)*100,"%")


