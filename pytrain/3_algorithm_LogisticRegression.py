from matplotlib import pyplot as plt
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split


#load dataset
data = pd.read_csv('1_data_working.csv',encoding='utf-8',sep=',')
one = np.ones((data.shape[0],1))
data.insert(loc=0,column='A',value=one)
data_X = data[["A","BMI","Smoking","AlcoholDrinking","Stroke","PhysicalHealth","MentalHealth","DiffWalking","Sex","AgeCategory","Race","Diabetic","PhysicalActivity","GenHealth","SleepTime","Asthma","KidneyDisease","SkinCancer"]]
data_Y = data["HeartDisease"]
Y_train, Y_test, X_train, X_test = train_test_split(data_Y, data_X, test_size=0.2, random_state=50)


#khớp vào mẫu bằng hồi quy logic
from sklearn.linear_model import LogisticRegression
# Tạo Regression Model
lr = LogisticRegression(solver='lbfgs', max_iter=1000)
# Train the model
lr.fit(X_train,Y_train)
# Sử dụng mô hình hồi quy để đưa ra dự đoán
all_pred = lr.predict(X_test)
# Độ đo accuracy
def accuracy(Y_test, all_pred):
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
print("Recall Score of Logistic Regression  : ",round(Recall, 4)*100,"%")
print("Precision Score of Logistic Regression  : ",round(Precision, 3)*100,"%")
print("F1_Measure Score of Logistic Regression  : ",round(F1_Measure, 4)*100,"%")
print("Accuracy Score of Logistic Regression  : ",round(accuracy(Y_test,all_pred), 4)*100,"%")
print('Các hệ số của mô hình : \n',lr.coef_)


score2 = accuracy(Y_test,all_pred)

from urllib import response
import requests
import json

host = 'http://localhost:3001/api/add'
params = {
    "a":lr.coef_[0][0],
    "BMI":lr.coef_[0][1],
    "Smoking":lr.coef_[0][2],
    "AlcoholDrinking":lr.coef_[0][3],
    "Stroke":lr.coef_[0][4],
    "PhysicalHealth":lr.coef_[0][5],
    "MentalHealth":lr.coef_[0][6],
    "DiffWalking":lr.coef_[0][7],
    "Sex":lr.coef_[0][8],
    "AgeCategory":lr.coef_[0][9],
    "Race":lr.coef_[0][10],
    "Diabetic":lr.coef_[0][11],
    "PhysicalActivity":lr.coef_[0][12],
    "GenHealth":lr.coef_[0][13],
    "SleepTime":lr.coef_[0][14],
    "Asthma":lr.coef_[0][15],
    "KidneyDisease":lr.coef_[0][16],
    "SkinCancer":lr.coef_[0][17],
    "per":round(score2, 4)*100
}
response_code = requests.get(host, params=params)
response_result = (json.dumps(response_code.json(), indent=4))
print(response_code)
print(response_result)

