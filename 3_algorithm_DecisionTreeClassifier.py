import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

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
y_pred5 = dtc.predict(X_test)
score3 = dtc.score(X_test,Y_test)
print("Score of Decision Tree Regressor  : ",round(score3, 4)*100,"%")
tree.plot_tree(dtc,max_depth=5,fontsize=10)
plt.show()
