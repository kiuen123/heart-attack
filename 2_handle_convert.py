import pandas as pd
from pandas import DataFrame

#đọc dữ liệu từ tệp heart_2020_cleaned.csv
data = pd.read_csv('1_data_heart_2020_cleaned.csv',encoding='utf-8',sep=',')
print(data.head())

# chuyển đổi BMI
# https://yhoccongdong.com/thongtin/bang-phan-loai-tinh-trang-dinh-duong/
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
print("chuyển đổi BMI thành công")

# chuyển đổi giới tính sang số
data.iloc[:,8].replace("Female",1,inplace=True)
data.iloc[:,8].replace("Male",0,inplace=True)
print("chuyển đổi giới tính sang dạng số thành công")

# xét giờ ngủ 
# https://www.vinmec.com/vi/tin-tuc/thong-tin-suc-khoe/suc-khoe-tong-quat/thoi-luong-ngu-theo-tung-do-tuoi/
for i in range(len(data)):
    if data.iloc[i,9] in ["18-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64"]:
        if data.iloc[i,14] < 7:
            data.iloc[i,14] = 0 # ngủ thiếu
        elif data.iloc[i,14] == 7 or data.iloc[i,14] == 8:
            data.iloc[i,14] = 0.5 # ngủ đủ
        elif data.iloc[i,14] > 8:
            data.iloc[i,14] = 1 # ngủ quá
    elif data.iloc[i,9] in ["65-69","70-74","75-79","80 or older"]:
        if data.iloc[i,14] < 7:
            data.iloc[i,14] = 0 # ngủ thiếu
        elif data.iloc[i,14] == 7 or data.iloc[i,14] == 8:
            data.iloc[i,14] = 0.5 # ngủ đủ
        elif data.iloc[i,14] > 8:
            data.iloc[i,14] = 1 # ngủ quá
print("chuyển đổi thời gian ngủ thành công")

# xét độ tuổi 
# https://phcn-online.com/2015/10/16/bai-1-dai-cuong-ve-qua-trinh-phat-trien-con-nguoi/ 
# https://vie.encyclopedia-titanica.com/etapas-del-desarrollo-humano
data.iloc[:,9].replace("18-24",1/3,inplace=True)# thanh niên
data.iloc[:,9].replace("25-29",1/3,inplace=True)# thanh niên
data.iloc[:,9].replace("30-34",1/3,inplace=True)# thanh niên
data.iloc[:,9].replace("35-39",1/3,inplace=True)# thanh niên
data.iloc[:,9].replace("40-44",2/3,inplace=True)# trung niên
data.iloc[:,9].replace("45-49",2/3,inplace=True)# trung niên
data.iloc[:,9].replace("50-54",2/3,inplace=True)# trung niên
data.iloc[:,9].replace("55-59",2/3,inplace=True)# trung niên
data.iloc[:,9].replace("60-64",3/3,inplace=True)# người già
data.iloc[:,9].replace("65-69",3/3,inplace=True)# người già
data.iloc[:,9].replace("70-74",3/3,inplace=True)# người già
data.iloc[:,9].replace("75-79",3/3,inplace=True)# người già
data.iloc[:,9].replace("80 or older",3/3,inplace=True)# người già
print("chuyển đổi nhóm tuổi thành công")

# chuyển đổi nhóm sắc tộc sang số 
data.iloc[:,10].replace("White",1,inplace=True)
data.iloc[:,10].replace("Black",4/5,inplace=True)
data.iloc[:,10].replace("Asian",3/5,inplace=True)
data.iloc[:,10].replace("American Indian/Alaskan Native",2/5,inplace=True)
data.iloc[:,10].replace("Other",1/5,inplace=True)
data.iloc[:,10].replace("Hispanic",0,inplace=True)
print("chuyển đổi nhóm sắc tộc sang dạng số thành công")

# chuyển đồi bệnh tiểu đường sang số
data.iloc[:,11].replace("Yes",1,inplace=True)
data.iloc[:,11].replace("Yes (during pregnancy)",2/3,inplace=True)
data.iloc[:,11].replace("No, borderline diabetes",1/3,inplace=True)
data.iloc[:,11].replace("No",0,inplace=True)
print("chuyển đổi bệnh tiểu dg sang dạng số thành công")

# Categorize the Health of the person into integers values
data.iloc[:,13].replace("Excellent",1,inplace=True)
data.iloc[:,13].replace("Very good",3/4,inplace=True)
data.iloc[:,13].replace("Good",2/4,inplace=True)
data.iloc[:,13].replace("Fair",1/4,inplace=True)
data.iloc[:,13].replace("Poor",0,inplace=True)
print("chuyển đổi chỉ số sức khỏe thành công")

#chuyển các giá trị y/n thành số
data.replace("Yes",1,inplace=True)
data.replace("No",0,inplace=True)
print("chuyển đổi các giá trị y/n sang dạng số thành công")

# chuyển về vùng xử lí 0 - 1
for i in range(len(data)):
    data.iloc[i,5] = data.iloc[i,5]/100
    data.iloc[i,6] = data.iloc[i,6]/100
print("chuyển giá trị về vùng xử lí {0,1} thành công")

print(data.head())

df = DataFrame(data)
export_csv = df.to_csv(r'1_data_working.csv', index = None, header=True) 
print("Exported to CSV successfully")
