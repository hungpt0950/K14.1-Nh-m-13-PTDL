import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from PIL import ImageTk, Image
import tkinter
from tkinter import *
from PIL import Image, ImageTk

path = "/Users/anhctl/Documents/Cao học/ProjectBTL/VN_housing_dataset.csv"
df = pd.read_csv(path)
pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width",None)
pd.set_option("display.max_colwidth",None)
# pd.set_option("", None)
# df.head(1000)
# print(df)


a = Tk()
a.title('Phân tích giá nhà Hà Nội')
a.geometry('500x500')
# top['bg'] = 'red'
a.attributes('-topmost',False)
name = Label(a,text = 'Nhóm 13', font = ('Times New Roman',32), bg='black', fg='white')
name.place(x=185,y=15)

n_rows, n_col = df.shape
print(n_rows, n_col)

# df.sample(10)
df[df.duplicated(keep=False)]
df = df.drop_duplicates()
df.drop(df.tail(1).index,inplace=True)
# df
# list(df.columns.values)
# df.dtypes
df['Ngày'] = pd.to_datetime(df['Ngày'])
# column Số tầng, Số phòng ngủ
# đối với số tầng nếu giá trị là nhiều hơn 10 ta mặc định là 11
df['Số tầng'][df['Số tầng'] == 'Nhiều hơn 10'] = 11
df['Số tầng'] = pd.to_numeric(df['Số tầng'])

df['Số phòng ngủ'][df['Số phòng ngủ'] == 'nhiều hơn 10 phòng'] = '11 phòng'
df['Số phòng ngủ'] = df['Số phòng ngủ'].str.split()
df['Số phòng ngủ'] = df['Số phòng ngủ'].str[0]
df['Số phòng ngủ'] = pd.to_numeric(df['Số phòng ngủ'])
# column Diện tích, Dài, Rộng, Giá
df['Diện tích'] = df['Diện tích'].str.split().str[0]
df['Dài'] = df['Dài'].str.split().str[0]
df['Rộng'] = df['Rộng'].str.split().str[0]
# df['Giá/m2'] = df['Giá/m2'].str.split().str[0]
# df['Giá/m2'] = df['Giá/m2'].str.replace(',', '')
df['Diện tích'] = pd.to_numeric(df['Diện tích'])
df['Dài'] = pd.to_numeric(df['Dài'])
df['Rộng'] = pd.to_numeric(df['Rộng'])
# df['Giá/m2'] = pd.to_numeric(df['Giá/m2'])
def transform(x):
    #print(x)
    if 'triệu/m²' in str(x):
        if ',' in str(x):
            number0 = len(x.split(',')[1].split(' ')[0])
            x = str(x).replace(',','').replace(' triệu/m²',(9-number0)*'0')
        else:
            x = str(x).replace(' triệu/m²',9*'0')
    if 'tỷ/m²' in str(x):
        if ',' in str(x):
            number0 = len(x.split(',')[1].split(' ')[0])
            x = str(x).replace(',','').replace(' tỷ/m²',(12-number0)*'0')
        else:
            x = str(x).replace(' tỷ/m²',12*'0')
    x = str(x).replace(' đ/m²','')
    return float(x)
df['Giá'] = df['Giá/m2'].apply(transform)
unique_counts = df.nunique() < 25
categorical  = unique_counts[unique_counts == True].index.tolist()
categorical.extend(df.select_dtypes(exclude=["number","bool_",]).columns.tolist())
# categorical
numeric = [x for x in df.columns.tolist() if x not in categorical]
# numeric
def calculate_quartile(data):
    nume_col_info_df = pd.DataFrame()
    for col in data.keys():
        missing = data[col].isnull().sum()
        missing_percentage = round(data[col].isnull().sum() * 100 / len(data[col]), 1)
        mean_value = data[col].mean()
        min_value = data[col].min()
        lower_quartile = data[col].quantile(0.25)
        median = data[col].median()
        upper = data[col].quantile(0.75)
        max_value = data[col].max()
        row_line = pd.Series([missing,missing_percentage,mean_value, min_value, lower_quartile, median, upper, max_value],
                            index = ['num_missing','missing_percentage', 'mean','min', 'lower_quartile', 'median', 'upper_quartile', 'max'])
        nume_col_info_df[col] = row_line
    return nume_col_info_df
# calculate_quartile(df[numeric])
missing = [];missing_percentage=[];num_values=[];value_percentages=[]

for column in categorical:
    missing.append(df[column].isnull().sum())
    missing_percentage.append((df[column].isnull().sum() * 100 / len(df)).round(1))
    temp = df[column].dropna()
    num_values.append(len(temp.unique()))
    value_percentages.append(((temp.value_counts(normalize=True)*100).round(1)).to_dict())
cat_info_df = pd.DataFrame([missing,missing_percentage,num_values,value_percentages],
                            index=['num_missing','missing_percentage','num_values','value_percentages'],
                            columns=list(categorical))
# cat_info_df
from matplotlib import cycler
colors = cycler('color', ['#EE6666', '#3388BB', '#9988DD', '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True, grid=True, prop_cycle=colors)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('xtick', direction='out', color='gray')
plt.rc('ytick', direction='out', color='gray')
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)

def ClickBtn():
    fig, axis = plt.subplots(1,2,figsize=(15,10))
    fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
    df['Loại hình nhà ở'].value_counts().plot(kind='pie', autopct='%1.1f%%',ax = axis[0], fontsize=13)
    df['Giấy tờ pháp lý'].value_counts().plot(kind='pie', autopct='%1.1f%%',ax = axis[1], fontsize=13)
    plt.title("Biểu Đồ Tròn");
    plt.show()

def ClickBtnHeatMap():
    df_corr = df.corr()
    mask = np.zeros_like(df_corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    plt.subplots(figsize=(18,12))
    sns.heatmap(df_corr, cmap="coolwarm", annot=True, mask=mask)
    plt.title("Biểu Đồ Nhiệt Tương Quan");
    plt.show()

but = Button(a,text = 'Biểu Đồ Dạng Tròn', width=15, height=1,font=('Times New Roman',13),command=ClickBtn)
but.place(x= 20,y = 30)
# butHeatmap = Button(a,text = 'Biểu Đồ Nhiệt Tương Quan', width=21, height=1,font=('Times New Roman',13),command=ClickBtnHeatMap)
# butHeatmap.place(x= 170,y = 120)


NhapTTSLQuan = Entry(a,width=10,font=('Times new Roman',10))
NhapTTSLQuan.place(x=20,y=80)

def SlQuanLayTT():
    x = df['Quận'].value_counts().keys()
    if(int(NhapTTSLQuan.get())>0):
        plt.barh(x[:int(NhapTTSLQuan.get())], df['Quận'].value_counts()[:int(NhapTTSLQuan.get())])
    plt.show()

butLayTTQuan = Button(a,text = 'Tt SL Nhà In Quận!', width=15, height=1,font=('Times New Roman',13),command=SlQuanLayTT)
butLayTTQuan.place(x= 90,y = 75)

def CacKieuNha():
    x = df['Loại hình nhà ở'].value_counts()
    plt.barh(x.keys(), x)
    plt.show()

butLoaiNha = Button(a,text = 'Các Loại Nhà Ở', width=15, height=1,font=('Times New Roman',13),command=CacKieuNha)
butLoaiNha.place(x= 20,y = 120)

def GiayToPhapLy():
    x = df['Giấy tờ pháp lý'].value_counts()
    plt.barh(x.keys(), x)
    plt.show()
butLoaiNha = Button(a,text = 'Tính Pháp Lý', width=15, height=1,font=('Times New Roman',13),command=GiayToPhapLy)
butLoaiNha.place(x= 20,y = 170)

def SLTANG():
    x = df['Số tầng'].value_counts()
    plt.bar(x.keys()[:10], x[:10])
    plt.show()
butSLTANG = Button(a,text = 'Thống Kê Tầng', width=15, height=1,font=('Times New Roman',13),command=SLTANG)
butSLTANG.place(x= 20,y = 220)

def SLPHONGNGU():
    x = df['Số phòng ngủ'].value_counts()
    plt.bar(x.keys(), x)
    plt.show()
butSLTANG = Button(a,text = 'SL Phòng Ngủ', width=15, height=1,font=('Times New Roman',13),command=SLPHONGNGU)
butSLTANG.place(x= 20,y = 270)

def DIENTICH():
    x = df['Diện tích']
    plt.hist(x, edgecolor='black', color='red', bins=np.arange(0, 150+1))
    plt.show()
butSLTANG = Button(a,text = 'Diện Tích', width=15, height=1,font=('Times New Roman',13),command=DIENTICH)
butSLTANG.place(x= 20,y = 320)

def GIA():
    # x = df['Giá'].value_counts()[:10]
    # plt.hist(x, edgecolor='black', color='red', bins=np.arange(0, 200+1));
    x = df['Giá'].value_counts()[:100]
    x = x.sort_index()
    plt.bar(x, x.keys());
    plt.xlim([0, 200]);
    # plt.hist(x, edgecolor='black', color='red', bins=np.arange(100000000, 15000000000+1));
    plt.show()
butSLTANG = Button(a,text = 'Giá', width=15, height=1,font=('Times New Roman',13),command=GIA)
butSLTANG.place(x= 20,y = 320)




NhapTTSLQuanCanTim = Entry(a,width=10,font=('Times new Roman',10))
NhapTTSLQuanCanTim.place(x=20,y=370)
def TILEPHANBONHAOCACPHUONG():
    fig, axis = plt.subplots(2,2,figsize=(15,10))
    fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
    df['Huyện'][df['Quận'] == NhapTTSLQuanCanTim.get()].value_counts().plot(kind='pie', ax=axis[0][0] , autopct='%1.1f%%', fontsize=13)
    plt.show()
butTILEPHANBONHAOCACPHUONG = Button(a,text = 'PB Nhà Trong Quận', width=15, height=1,font=('Times New Roman',13),command=TILEPHANBONHAOCACPHUONG)
butTILEPHANBONHAOCACPHUONG.place(x= 90,y = 365)

NhapGIATRI1 = Entry(a,width=10,font=('Times new Roman',10))
NhapGIATRI1.place(x=20,y=410)
NhapGIATRI2 = Entry(a,width=10,font=('Times new Roman',10))
NhapGIATRI2.place(x=20,y=450)
def BIEUDOPHANBO2THANHPHAN():
    # y = df[NhapGIATRI1.get()].sort_values(ascending=False)[:100]
    y = df[NhapGIATRI1.get()][:1500]
    x = df[NhapGIATRI2.get()][:1500]
    plt.scatter(x, y);
    # df['Diện tích']
    plt.ylabel(NhapGIATRI1.get());
    plt.xlabel(NhapGIATRI2.get());
    plt.show()
butBIEUDOPHANBO2THANHPHAN = Button(a,text = 'Biểu Đồ Phân Tán', width=15, height=1,font=('Times New Roman',13),command=BIEUDOPHANBO2THANHPHAN)
butBIEUDOPHANBO2THANHPHAN.place(x= 90,y = 420)

sub_df = df[['Địa chỉ','Quận','Huyện', 'Loại hình nhà ở', 'Giấy tờ pháp lý', 'Số tầng', 'Số phòng ngủ', 'Diện tích','Dài','Rộng', 'Giá']]
sub_df = sub_df.dropna()
# print(sub_df)
sub_df['Địa chỉ'] = pd.factorize(sub_df['Địa chỉ'])[0]
sub_df['Quận'] = pd.factorize(sub_df['Quận'])[0]
sub_df['Huyện'] = pd.factorize(sub_df['Huyện'])[0]
sub_df['Loại hình nhà ở'] = pd.factorize(sub_df['Loại hình nhà ở'])[0]
sub_df['Giấy tờ pháp lý'] = pd.factorize(sub_df['Giấy tờ pháp lý'])[0]
# print(sub_df)
# print(str(sub_df))

def HamXuatFileSauXL():
    file_object = open("Hieu.txt","w", encoding='utf-8')
    file_object.write(str(sub_df))

butXUATFILEDATAHAUXULY = Button(a,text = 'Xuất file data hậu xử lý', width=20, height=1,font=('Times New Roman',13),command=HamXuatFileSauXL)
butXUATFILEDATAHAUXULY.place(x= 250,y = 75)

from sklearn.decomposition import PCA

train_df = sub_df.drop('Giá', axis=1)

model = PCA(n_components=1)
model.fit(train_df)
info_2D =  model.transform(train_df)

train_df['PCA1'] = info_2D[:, 0]
# train_df['PCA2'] = info_2D[:, 1]


# plt.scatter(train_df['PCA1'][:10000], sub_df['Giá'][:10000])
# # plt.show()

# pair_df = sub_df
# g = sns.pairplot(pair_df[:100], height=2.5)
# # plt.show()

x = pd.concat([train_df[['PCA1']], sub_df[['Giá']]], axis=1)
x['Giá'].value_counts(sort=False)

# x = x.sample(100)
# sns.lmplot(x="PCA1",y="Giá", data=x);
# plt.show()


NhapGIATRI12 = Entry(a,width=10,font=('Times new Roman',10))
NhapGIATRI12.place(x=250,y=125)
NhapGIATRI23 = Entry(a,width=10,font=('Times new Roman',10))
NhapGIATRI23.place(x=250,y=165)

def FUNCTIONPCA():
    _x = sub_df.sample(100)
    sns.lmplot(x=NhapGIATRI12.get(), y=NhapGIATRI23.get(), data=_x);
    plt.show()

butPCA = Button(a,text = 'PCA', width=15, height=1,font=('Times New Roman',13),command=FUNCTIONPCA)
butPCA.place(x= 320,y = 135)


image1 = Image.open('Logo.jpg')
image2 = image1.resize((225, 250))
img = ImageTk.PhotoImage(image2)

# hinhanh = Button(a,text = '',font=('Time New Roman',11),image = img)
# hinhanh.place(x = 250,y = 200)

label1 = tkinter.Label(image=img)
label1.image = image1
label1.place(x = 250,y = 200)

a.mainloop()
