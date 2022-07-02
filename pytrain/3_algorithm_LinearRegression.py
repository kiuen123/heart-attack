import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
#load dataset
data = pd.read_csv('1_data_working.csv',encoding='utf-8',sep=',')

one = np.ones((data.shape[0],1))
data.insert(loc=0,column='A',value=one)
data_X = data[["A","BMI","Smoking","AlcoholDrinking","Stroke","PhysicalHealth","MentalHealth","DiffWalking","Sex","AgeCategory","Race","Diabetic","PhysicalActivity","GenHealth","SleepTime","Asthma","KidneyDisease","SkinCancer"]]
data_Y = data["HeartDisease"]
Y_train, Y_test, X_train, X_test = train_test_split(data_Y, data_X, test_size=0.2, random_state=50)

#khớp vào mẫu bằng hồi quy tuyến tính
lr = LinearRegression()
lr.fit(X_train,Y_train)
all_pred = lr.predict(X_test)
score2 = lr.score(X_test,Y_test)
print("Accuracy Score of Linear Regression  : ",score2*100,"%")

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# In ra các chỉ số liên quan
print("Model Coefficients:", lr.coef_)
print("Mean Absolute Error:", mean_absolute_error(Y_test, all_pred))
print("Accuracy Score of Linear Regression  :", r2_score(Y_test, all_pred))


# from sklearn.metrics import recall_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import v_measure_score
from sklearn.metrics import accuracy_score
# print("Recall Score of Linear Regression    : ",round(recall_score(Y_test,all_pred), 4)*100,"%")
# print("Precision Score of Linear Regression : ",round(precision_score(Y_test,all_pred), 4)*100,"%")
# print("Measure Score of Linear Regression   : ",round(v_measure_score(Y_test,all_pred), 4)*100,"%")
print("Accuracy Score of Linear Regression  : ",round(r2_score(Y_test,all_pred), 6)*100,"%")


