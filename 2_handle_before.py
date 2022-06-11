import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

#đọc dữ liệu từ tệp heart_2020_cleaned.csv
df = pd.read_csv('1_data_heart_2020_cleaned.csv',encoding='utf-8',sep=',')

sns.set_style("darkgrid", {"grid.color": ".6"})
colRange = [['Smoking','AlcoholDrinking','Stroke'],['DiffWalking','Sex','PhysicalActivity'],['Asthma','KidneyDisease','SkinCancer']]
def printCount(cols):
    fig, axes = plt.subplots(3, 3, figsize=(16, 9))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    row=0
    col=0
    p_count=1
    for row in range(3):
        for col in range(3):
            column = colRange[row][col]
            sns.countplot(ax=axes[row,col],x=df[column],hue=df['HeartDisease'])
            axes[row,col].set_title("Counts of {} (Plot {})".format(column,p_count))
            p_count += 1
printCount(colRange)

plt.figure(figsize=(12,6))
sns.countplot(df['Race'],hue=df['HeartDisease'])
plt.title('Variation of Heart Disease amoung Races')

plt.figure(figsize=(12,6))
sns.countplot(df['Diabetic'],hue=df['HeartDisease'])
plt.title('Variation of Heart Disease among Diabetic People')

plt.figure(figsize=(12,6))
sns.histplot(data=df[df['HeartDisease']=='Yes'],x='BMI',kde=True,color='red')
sns.histplot(data=df[df['HeartDisease']=='No'],x='BMI',kde=True,color='blue')
plt.title('Distribution of BMI Among People')

plt.figure(figsize=(12,6))
sns.kdeplot(df[df['HeartDisease']=='Yes']['PhysicalHealth'],shade=True,color='red')
sns.kdeplot(df[df['HeartDisease']=='No']['PhysicalHealth'],shade=True,color='blue')
plt.title('Physical Health Pattern')

plt.figure(figsize=(12,6))
sns.kdeplot(df[df['HeartDisease']=='Yes']['MentalHealth'],shade=True,color='red')
sns.kdeplot(df[df['HeartDisease']=='No']['MentalHealth'],shade=True,color='blue')
plt.title('Variation of Mental Health')

plt.show()