import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
import joblib 
import warnings
warnings.filterwarnings('ignore')

loan_data = pd.read_csv('loan.csv')

loan_data = loan_data.drop(['Loan_ID'], axis=1)

loan_data=loan_data.dropna()

from sklearn.preprocessing import LabelEncoder
lr=LabelEncoder()
loan_data['Gender']=lr.fit_transform(loan_data['Gender'])
loan_data['Married']=lr.fit_transform(loan_data['Married'])
loan_data['Education']=lr.fit_transform(loan_data['Education'])
loan_data['Self_Employed']=lr.fit_transform(loan_data['Self_Employed'])
loan_data['Property_Area']=lr.fit_transform(loan_data['Property_Area'])
loan_data['Loan_Status']=lr.fit_transform(loan_data['Loan_Status'])
loan_data['Dependents'] = loan_data['Dependents'].replace('3+', '3')
loan_data['Dependents'] = pd.to_numeric(loan_data['Dependents'])

x=loan_data.iloc[:,:-1].values
y=loan_data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=20)

from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)
#inp=[[1,1,1,0,0,4583,1508.0,128.0,360.0,1.0,0]]
#yp=nb.predict(inp)
#print(yp)

joblib.dump(nb,"loan.joblib")