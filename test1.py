import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree

#đọc dữ liệu từ tệp heart_2020_cleaned.csv
data = pd.read_csv('heart_2020_cleaned.csv',encoding='utf-8',sep=',')

# chuyển đổi BMI
for i in range(len(data)):
    if data.iloc[i,1] < 16: # gầy độ 3
        data.iloc[i,1] = 1/8
    elif data.iloc[i,1] >= 16 and data.iloc[i,1] < 17: # gầy độ 2
        data.iloc[i,1] = 2/8
    elif data.iloc[i,1] >= 17 and data.iloc[i,1] < 18.5: # gầy độ 1
        data.iloc[i,1] = 3/8
    elif data.iloc[i,1] >= 18.5 and data.iloc[i,1] < 25: # bình thường
        data.iloc[i,1] = 4/8
    elif data.iloc[i,1] >= 25 and data.iloc[i,1] < 30: # tiền béo phì
        data.iloc[i,1] = 5/8
    elif data.iloc[i,1] >= 30 and data.iloc[i,1] < 35: # béo phì độ 1
        data.iloc[i,1] = 6/8
    elif data.iloc[i,1] >= 35 and data.iloc[i,1] < 40: # béo phì độ 2
        data.iloc[i,1] = 7/8
    elif data.iloc[i,1] >= 40: # béo phì độ 3
        data.iloc[i,1] = 8/8

# Converting Gender type to Integers
data.iloc[:,8].replace("Female",1,inplace=True)
data.iloc[:,8].replace("Male",0,inplace=True)
# Categorizing Age values
data.iloc[:,9].replace("18-24",1/13,inplace=True)
data.iloc[:,9].replace("25-29",2/13,inplace=True)
data.iloc[:,9].replace("30-34",3/13,inplace=True)
data.iloc[:,9].replace("35-39",4/13,inplace=True)
data.iloc[:,9].replace("40-44",5/13,inplace=True)
data.iloc[:,9].replace("45-49",6/13,inplace=True)
data.iloc[:,9].replace("50-54",7/13,inplace=True)
data.iloc[:,9].replace("55-59",8/13,inplace=True)
data.iloc[:,9].replace("60-64",9/13,inplace=True)
data.iloc[:,9].replace("65-69",10/13,inplace=True)
data.iloc[:,9].replace("70-74",11/13,inplace=True)
data.iloc[:,9].replace("75-79",12/13,inplace=True)
data.iloc[:,9].replace("80 or older",13/13,inplace=True)

# Categorize Race of the person 
data.iloc[:,10].replace("White",1/6,inplace=True)
data.iloc[:,10].replace("Black",2/6,inplace=True)
data.iloc[:,10].replace("Asian",3/6,inplace=True)
data.iloc[:,10].replace("American Indian/Alaskan Native",4/6,inplace=True)
data.iloc[:,10].replace("Other",5/6,inplace=True)
data.iloc[:,10].replace("Hispanic",6/6,inplace=True)

# Catgorize if the person is diabetic or not 
data.iloc[:,11].replace("Yes",1,inplace=True)
data.iloc[:,11].replace("Yes (during pregnancy)",2/3,inplace=True)
data.iloc[:,11].replace("No, borderline diabetes",1/3,inplace=True)
data.iloc[:,11].replace("No",0,inplace=True)

# Categorize the Health of the person into integers values
data.iloc[:,13].replace("Excellent",1,inplace=True)
data.iloc[:,13].replace("Very good",4/3,inplace=True)
data.iloc[:,13].replace("Good",2/4,inplace=True)
data.iloc[:,13].replace("Fair",1/4,inplace=True)
data.iloc[:,13].replace("Poor",0,inplace=True)

#chuyển kết quả thành dạng số
data.replace("Yes",1,inplace=True)
data.replace("No",0,inplace=True)

print(data)

one = np.ones((data.shape[0],1))
data.insert(loc=0,column='A',value=one)
data_X = data[["A","BMI","Smoking","AlcoholDrinking","Stroke","PhysicalHealth","MentalHealth","DiffWalking","Sex","AgeCategory","Race","Diabetic","PhysicalActivity","GenHealth","SleepTime","Asthma","KidneyDisease","SkinCancer"]]
data_Y = data["HeartDisease"]
Y_train, Y_test, X_train, X_test = train_test_split(data_Y, data_X, test_size=0.2, random_state=50)

# khớp vào mẫu bằng hồi quy tuyến tính
regr = LinearRegression()
regr.fit(X_train,Y_train)
Y_pred = regr.predict(X_test)
score1 = regr.score(X_test,Y_test)
print("Score of Linear Regression        : ",round(score1, 4)*100,"%")
# print(regr.coef_)

# khớp vào mẫu bằng hồi quy logic
lr = LogisticRegression(solver='lbfgs', max_iter=1000)
lr.fit(X_train,Y_train)
all_pred = lr.predict(X_test)
score2 = lr.score(X_test,Y_test)
print("Score of Logistic Regression : ",round(score2, 4)*100,"%")
# print(lr.coef_)

# khớp vào mẫu bằng cây phân lớp
clf5 = DecisionTreeClassifier(max_depth=5,max_leaf_nodes=2)
clf5.fit(X_train, Y_train)
y_pred5 = clf5.predict(X_test)
score3 = clf5.score(X_test,Y_test)
print("Score of Decision Tree Regressor  : ",round(score3, 4)*100,"%")
# tree.plot_tree(clf5,max_depth=10,fontsize=10)
# plt.show()

params = {
    "n_estimators": 90,  # Number of trees in the forest
    "max_depth": 5,  # Max depth of the tree
    "min_samples_split": 12,  # Min number of samples required to split a node
    "min_samples_leaf": 2,  # Min number of samples required at a leaf node
    "ccp_alpha": 0,  # Cost complexity parameter for pruning
    "random_state": 42,
}
r_clf = RandomForestRegressor(**params)
r_clf = r_clf.fit(X_train, Y_train)
r_pred =r_clf.predict(X_test)
score4 = r_clf.score(X_test,Y_test)
print("Score of Random Forest Regressor  : ",round(score4, 4)*100,"%")