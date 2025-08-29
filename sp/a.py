import numpy as np
import pandas as pd
import joblib

data=pd.read_csv("Salary_Data.csv")


avgsal=data['Salary'].mean()
data.fillna({'Salary':avgsal}, inplace=True)

x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x,y)

inp=[[2.5]]
yp=model.predict(inp)
print(yp)

joblib.dump(model,"ml38.joblib")