from urllib import response
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import requests
import json

#load dataset
data = pd.read_csv('1_data_working.csv',encoding='utf-8',sep=',')

one = np.ones((data.shape[0],1))
data.insert(loc=0,column='A',value=one)
data_X = data[["A","BMI","Smoking","AlcoholDrinking","Stroke","PhysicalHealth","MentalHealth","DiffWalking","Sex","AgeCategory","Race","Diabetic","PhysicalActivity","GenHealth","SleepTime","Asthma","KidneyDisease","SkinCancer"]]
data_Y = data["HeartDisease"]
Y_train, Y_test, X_train, X_test = train_test_split(data_Y, data_X, test_size=0.2, random_state=50)

#khớp vào mẫu bằng hồi quy logic
lr = LogisticRegression(solver='lbfgs', max_iter=1000)
lr.fit(X_train,Y_train)
all_pred = lr.predict(X_test)
score2 = lr.score(X_test,Y_test)
print("Score of Logistic Regression : ",round(score2, 4)*100,"%")
print(lr.coef_)
import math

test=0
a = [1,0.75,1,0,0,0.3,0,1,0,1,1,1,0,0,1,1,0,0]
for i in range(len(lr.coef_[0])):
    test +=lr.coef_[0][i]*a[i]
print(math.log10(test))

# host = 'http://118.71.64.144:3001/api/add'
# params = {
#     "a":lr.coef_[0][0],
#     "BMI":lr.coef_[0][1],
#     "Smoking":lr.coef_[0][2],
#     "AlcoholDrinking":lr.coef_[0][3],
#     "Stroke":lr.coef_[0][4],
#     "PhysicalHealth":lr.coef_[0][5],
#     "MentalHealth":lr.coef_[0][6],
#     "DiffWalking":lr.coef_[0][7],
#     "Sex":lr.coef_[0][8],
#     "AgeCategory":lr.coef_[0][9],
#     "Race":lr.coef_[0][10],
#     "Diabetic":lr.coef_[0][11],
#     "PhysicalActivity":lr.coef_[0][12],
#     "GenHealth":lr.coef_[0][13],
#     "SleepTime":lr.coef_[0][14],
#     "Asthma":lr.coef_[0][15],
#     "KidneyDisease":lr.coef_[0][16],
#     "SkinCancer":lr.coef_[0][17],
#     "per":round(score2, 4)*100
# }
# response_code = requests.get(host, params=params)
# response_result = (json.dumps(response_code.json(), indent=4))
# print(response_code)
# print(response_result)