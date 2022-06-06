import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#load data
#đọc dữ liệu từ tệp heart_2020_cleaned.csv
data = pd.read_csv('heart_2020_cleaned.csv',encoding='utf-8',sep=',')

# Converting Gender type to Integers
data.iloc[:,8].replace("Female",1,inplace=True)
data.iloc[:,8].replace("Male",0,inplace=True)
# Categorizing Age values
data.iloc[:,9].replace("18-24",1,inplace=True)
data.iloc[:,9].replace("25-29",2,inplace=True)
data.iloc[:,9].replace("30-34",3,inplace=True)
data.iloc[:,9].replace("35-39",4,inplace=True)
data.iloc[:,9].replace("40-44",5,inplace=True)
data.iloc[:,9].replace("45-49",6,inplace=True)
data.iloc[:,9].replace("50-54",7,inplace=True)
data.iloc[:,9].replace("55-59",8,inplace=True)
data.iloc[:,9].replace("60-64",9,inplace=True)
data.iloc[:,9].replace("65-69",10,inplace=True)
data.iloc[:,9].replace("70-74",11,inplace=True)
data.iloc[:,9].replace("75-79",12,inplace=True)
data.iloc[:,9].replace("80 or older",13,inplace=True)

# Categorize Race of the person 
data.iloc[:,10].replace("White",1,inplace=True)
data.iloc[:,10].replace("Black",2,inplace=True)
data.iloc[:,10].replace("Asian",3,inplace=True)
data.iloc[:,10].replace("American Indian/Alaskan Native",4,inplace=True)
data.iloc[:,10].replace("Other",5,inplace=True)
data.iloc[:,10].replace("Hispanic",6,inplace=True)

# Catgorize if the person is diabetic or not 
data.iloc[:,11].replace("Yes",4,inplace=True)
data.iloc[:,11].replace("Yes (during pregnancy)",3,inplace=True)
data.iloc[:,11].replace("No, borderline diabetes",2,inplace=True)
data.iloc[:,11].replace("No",1,inplace=True)

# Categorize the Health of the person into integers values
data.iloc[:,13].replace("Excellent",4,inplace=True)
data.iloc[:,13].replace("Very good",3,inplace=True)
data.iloc[:,13].replace("Good",2,inplace=True)
data.iloc[:,13].replace("Fair",1,inplace=True)
data.iloc[:,13].replace("Poor",0,inplace=True)

#chuyển kết quả thành dạng số
data.replace("Yes",1,inplace=True)
data.replace("No",0,inplace=True)

# print(data)

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