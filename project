# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 17:42:07 2022

@author: Dipali.Badgujar
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv(r"C:\360 CLASSES\Project_71\edx_courses.csv",encoding = 'unicode_escape')
df=df.drop(['title','summary','instructors','subtitles'], axis=1)
df.describe()
df.info()
df.columns

df.isnull()
df.isnull().sum()

# changing datatype of n_enrolled column
df.n_enrolled= df.n_enrolled.str.replace(',','')
df.n_enrolled=df.n_enrolled.astype('float32')
df.info()

## imputation for columns having null values
from sklearn.impute import SimpleImputer
df.n_enrolled.mean()
df.n_enrolled.median()
df.n_enrolled.mode()

# mean imputation for n_enrolled column
mean_imputer=SimpleImputer(missing_values=np.nan, strategy='mean')
df["n_enrolled"]=pd.DataFrame(mean_imputer.fit_transform(df[["n_enrolled"]]))
df.isnull().sum()

df.n_enrolled=df.n_enrolled.astype('int64')
df.info()
df.price=df.price.astype('int64')
df.info()

# checking for duplicate values
duplicate= df.duplicate()
duplicate.value_count()
duplicate
df=df.drop_duplicates()
duplicate=df.duplicated().sum()
duplicate

## Finding Outlier 
sns.boxplot(data= df)
sns.boxplot(df.n_enrolled)
sns.boxplot(df.course_hours)
sns.boxplot(df.price)

## Outlier treatment
## n_enrolled
IQR = df['n_enrolled'].quantile(0.75) - df['n_enrolled'].quantile(0.25)
lower_limit = df['n_enrolled'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['n_enrolled'].quantile(0.75) + (IQR * 1.5)

outliers = np.where(df['n_enrolled'] > upper_limit, True, np.where(df['n_enrolled'] < lower_limit, True, False))
sum(outliers)
outliers_values = df.n_enrolled[outliers]
outliers_values

## trimming values
df=df.iloc[(~outliers)]
sns.boxplot(df.n_enrolled);plt.title("Trimmed boxplot for n_enrolled");plt.show()

# course_hours
sns.boxplot(df.course_hours)
IQR = df['course_hours'].quantile(0.75) - df['course_hours'].quantile(0.25)
lower_limit = df['course_hours'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['course_hours'].quantile(0.75) + (IQR * 1.5)

outliers = np.where(df['course_hours'] > upper_limit, True, np.where(df['course_hours'] < lower_limit, True, False))
sum(outliers)
outliers_values = df.course_hours[outliers]
outliers_values

## trimming values
df=df.iloc[(~outliers)]
sns.boxplot(df.course_hours);plt.title("Trimmed boxplot for course_hours");plt.show()

## Replacing remaining values
IQR = df['course_hours'].quantile(0.75) - df['course_hours'].quantile(0.25)
lower_limit = df['course_hours'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['course_hours'].quantile(0.75) + (IQR * 1.5)

outliers = np.where(df['course_hours'] > upper_limit, True, np.where(df['course_hours'] < lower_limit, True, False))
sum(outliers)
outliers_values = df.course_hours[outliers]
outliers_values

df.course_hours = np.where(df['course_hours'] > upper_limit,upper_limit, np.where(df['course_hours'] < lower_limit,lower_limit,df['course_hours']))
sns.boxplot(df.course_hours);plt.title("Boxplot course_hours after replacing");plt.show()

## price
sns.boxplot(df.price)
IQR = df['price'].quantile(0.75) - df['price'].quantile(0.25)
lower_limit = df['price'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['price'].quantile(0.75) + (IQR * 1.5)

outliers = np.where(df['price'] > upper_limit, True, np.where(df['price'] < lower_limit, True, False))
sum(outliers)
outliers_values = df.price[outliers]
outliers_values

# trimming technique
df=df.iloc[(~outliers)]
sns.boxplot(df.price);plt.title("Trimmed boxplot for price");plt.show()

sns.heatmap(df.corr())

## exploring data
sns.distplot(df["price"])

df.course_type.unique()
df.course_type.value_counts()
df['institution'].value_counts()
df['Level'].value_counts()
df['subject'].value_counts()

y = df.price
x= df.drop('price', axis=1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error

#from sklearn.linear_model import LinearRegression,Ridge,Lasso
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
#from sklearn.svm import SVR
#from xgboost import XGBRegressor
# pip install xgboost

####################### Gradient Boost ###############


step1 = ColumnTransformer(transformers=[
('col_tnf',OneHotEncoder(sparse=False,handle_unknown = 'ignore'),[1,2,3,4,5])
],remainder='passthrough')

step2 = GradientBoostingRegressor(n_estimators=500)
pipe = Pipeline([('step1',step1),('step2',step2)])
pipe.fit(x_train,y_train)
y_pred = pipe.predict(x_test)
R2_GBoost = r2_score(y_test,y_pred)
R2_GBoost # 0.68155
MAE_gboost = mean_absolute_error(y_test,y_pred) # 1850.740
MAE_gboost

# save the model_ar to disk
import pickle
filename= "Final_code_edx.pkl"
pickle.dump(pipe,open(filename,"wb"))
Final_code_edx=pickle.load(open("Final_code_edx.pkl","rb"))
