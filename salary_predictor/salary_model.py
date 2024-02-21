import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import joblib
dataset=pd.read_csv("SalaryData.csv")
x=dataset['YearsExperience']
y=dataset['Salary']

x=x.values
x=x.reshape(-1,1)

model=LinearRegression()
model.fit(x,y)

print(model.predict([[0]]))

yhat = model.predict(x)
print(metrics.mean_absolute_error(y,yhat))

joblib.dump(model,'salarymodel.pk1')
